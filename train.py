# -*- coding: utf-8 -*-
import datetime
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BASE_LR,
    BATCH_SIZE,
    BETAS,
    CHECKPOINT_TAG,
    CLIP_DOWNLOAD_ROOT,
    CLIP_MODEL_NAME,
    COEFF_CORRELATION,
    COEFF_FREQ_FINAL,
    COEFF_FUSION,
    COEFF_LOCAL_CONTRAST,
    COEFF_SSIM,
    FREQUENCY_WARMUP_EPOCHS,
    GLOBAL_GRAD_CLIP,
    METRIC_WEIGHTS,
    MIN_LR,
    MODEL_DIRECTORY,
    MODEL_VERSION,
    TRAIN_H5_PATH,
    USE_CLIP_IMAGE_QUERY,
    VAL_BASELINE_CHECKPOINT,
    VAL_BASELINE_JSON,
    VAL_EXPECTED_PAIRS,
    VAL_INFRARED_DIR,
    VAL_VISIBLE_DIR,
    VAL_VISIBLE_RGB_DIR,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from net.Network import (
    DualDomainTextIntentGenerator,
    DualStreamIntentMLP,
    FSRC,
    FrequencyPyramidAdapter,
    FusionDecoder,
    SharedEncoder,
    TextConditionedSpatialFusion,
)
from net.frequency_fusion import TGSFF
from utils.clip_preprocess import preprocess_clip_rgb
from utils.dataset import H5Dataset
from utils.loss import (
    CorrelationConsistencyLoss,
    FrequencyConsistencyLoss,
    Fusionloss,
    LocalContrastLoss,
    SimpleSSIMLoss,
)
from utils.training_utils import (
    append_csv,
    build_lr_scheduler,
    compute_validation_score,
    get_frequency_weight,
    init_csv,
)
from utils.val_metrics import compute_val_metrics
from utils.validation_dataset import PairedValidationDataset


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ---------------------------------------------------------------------------
# Single build_model (shared by training and baseline)
# ---------------------------------------------------------------------------

def build_model(device: torch.device):
    encoder = nn.DataParallel(
        SharedEncoder(inp_channels=1, feature_dim=64, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0)
    ).to(device)
    if USE_CLIP_IMAGE_QUERY:
        intent_core = DualDomainTextIntentGenerator(
            intent_dim=64, clip_model_name=CLIP_MODEL_NAME,
            clip_download_root=CLIP_DOWNLOAD_ROOT, clip_device=str(device),
        )
    else:
        intent_core = DualStreamIntentMLP(
            channels=64, intent_dim=64, hidden_dim=256,
            clip_model_name=CLIP_MODEL_NAME, clip_download_root=CLIP_DOWNLOAD_ROOT, clip_device=str(device),
        )
    intent_generator = nn.DataParallel(intent_core).to(device)
    frequency_fusion = nn.DataParallel(TGSFF(
        in_channels=64, patch_size=4, amp_topk_ratio=0.30, phase_topk_ratio=0.35,
        token_embed_dim=128, num_heads=4, return_aux=True, routing_temperature=0.25,
    )).to(device)
    frequency_pyramid_adapter = nn.DataParallel(FrequencyPyramidAdapter(channels=64)).to(device)
    spatial_fusion = nn.DataParallel(TextConditionedSpatialFusion(
        channels=64, intent_dim=64, num_heads=4, ffn_expansion_factor=2.0,
        init_res_scale=0.05, use_freq_context=False,
    )).to(device)
    fsrc_l1 = nn.DataParallel(FSRC(channels=64)).to(device)
    fsrc_l2 = nn.DataParallel(FSRC(channels=64)).to(device)
    fsrc_l3 = nn.DataParallel(FSRC(channels=64)).to(device)
    fusion_decoder = nn.DataParallel(FusionDecoder(
        channels=64, out_channels=1, inner_dim=24, num_blocks=1, num_heads=1, ffn_expansion_factor=2.0,
    )).to(device)
    return encoder, intent_generator, frequency_fusion, frequency_pyramid_adapter, spatial_fusion, fsrc_l1, fsrc_l2, fsrc_l3, fusion_decoder


def unwrap(module):
    return module.module if isinstance(module, nn.DataParallel) else module


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_MODEL_KEYS = (
    'shared_encoder', 'intent_generator', 'frequency_fusion', 'frequency_pyramid_adapter',
    'spatial_fusion', 'fsrc_l1', 'fsrc_l2', 'fsrc_l3', 'fusion_decoder',
)


def save_checkpoint(
    path,
    modules,
    optimizer=None, scheduler=None,
    epoch=None, val_score=None, val_metrics=None, val_ratios=None,
    is_best=False,
):
    checkpoint = {
        'model_version': MODEL_VERSION,
        'use_clip_image_query': USE_CLIP_IMAGE_QUERY,
        'intent_generator_type': type(unwrap(modules[1])).__name__,
    }
    for key, module in zip(_MODEL_KEYS, modules):
        checkpoint[key] = module.state_dict()

    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = int(epoch)
    if val_score is not None:
        checkpoint['val_score'] = float(val_score)
    if val_metrics is not None:
        checkpoint['val_metrics'] = dict(val_metrics)
    if val_ratios is not None:
        checkpoint['val_ratios'] = dict(val_ratios)
    checkpoint['is_best'] = bool(is_best)

    # Atomic write
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def validate_checkpoint_metadata(checkpoint, intent_generator):
    """Reuse test.py metadata validation."""
    expected_type = type(unwrap(intent_generator)).__name__
    if checkpoint.get('model_version') != MODEL_VERSION:
        raise RuntimeError('Checkpoint model version mismatch.')
    if checkpoint.get('use_clip_image_query') != USE_CLIP_IMAGE_QUERY:
        raise RuntimeError('Checkpoint query variant mismatch.')
    if checkpoint.get('intent_generator_type') != expected_type:
        raise RuntimeError('Checkpoint intent generator type mismatch.')


def load_modules_from_checkpoint(modules, checkpoint):
    """Strict load with all keys required."""
    for key, module in zip(_MODEL_KEYS, modules):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing key: {key}")
        module.load_state_dict(checkpoint[key], strict=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(modules, valloader, device):
    """Run inference on all validation pairs and return averaged metrics."""
    previous_modes = [m.training for m in modules]

    try:
        for m in modules:
            m.eval()

        all_metrics = {k: [] for k in METRIC_WEIGHTS}

        with torch.no_grad():
            for (data_vis, data_ir, data_vis_rgb_raw,
                 metric_vis_u8, metric_ir_u8, _) in valloader:
                data_vis = data_vis.to(device)
                data_ir = data_ir.to(device)
                data_vis_clip = preprocess_clip_rgb(data_vis_rgb_raw.to(device))

                vis_spa, vis_freq, _ = modules[0](data_vis)
                ir_spa, ir_freq, _ = modules[0](data_ir)
                i_deg, i_fus, _ = modules[1](data_vis_clip, vis_spa, ir_spa, vis_freq, ir_freq)
                fused_freq, _ = modules[2](vis_freq, ir_freq, frequency_intent=i_deg)
                _, spatial_pyramid, _ = modules[4](vis_spa, ir_spa, i_fus, return_aux=True, return_pyramid=True)
                freq_pyramid = modules[3](fused_freq, target_pyramid=spatial_pyramid)
                d_l1, _ = modules[5](freq_pyramid['l1'], spatial_pyramid['l1'])
                d_l2, _ = modules[6](freq_pyramid['l2'], spatial_pyramid['l2'])
                d_l3, _ = modules[7](freq_pyramid['l3'], spatial_pyramid['l3'])
                fused_image, _ = modules[8](d_l1, d_l2, d_l3)

                metrics = compute_val_metrics(
                    fused=fused_image,
                    vis_u8=metric_vis_u8,
                    ir_u8=metric_ir_u8,
                )
                for k in METRIC_WEIGHTS:
                    all_metrics[k].append(metrics[k])

    finally:
        for m, mode in zip(modules, previous_modes):
            m.train(mode)

    avg_metrics = {k: float(sum(v) / len(v)) for k, v in all_metrics.items()}
    return avg_metrics


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def validate_baseline_json(json_path, val_filenames):
    """Validate baseline JSON has all required fields and matches current val set."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check required metric fields
    for k in METRIC_WEIGHTS:
        if k not in data:
            raise KeyError(f"Baseline JSON missing metric key: {k}")
        v = float(data[k])
        if not math.isfinite(v):
            raise ValueError(f"Baseline metric {k} is not finite: {v}")
        if v <= 0:
            raise ValueError(f"Baseline metric {k} is not positive: {v}")

    # Check validation filenames consistency
    saved_filenames = data.get("validation_filenames")
    if saved_filenames is None:
        raise ValueError("Baseline JSON missing validation_filenames field.")
    if saved_filenames != val_filenames:
        raise ValueError(
            "Validation filenames changed since baseline was generated. "
            "Delete the baseline JSON and re-run to regenerate."
        )

    # Check validation_count
    saved_count = data.get("validation_count")
    if saved_count is None:
        raise ValueError("Baseline JSON missing validation_count field.")
    if saved_count != len(val_filenames):
        raise ValueError(
            "Baseline validation_count does not match the current validation set."
        )

    # Check baseline checkpoint path consistency
    saved_checkpoint = data.get("baseline_checkpoint_path")
    if saved_checkpoint is None:
        raise ValueError("Baseline JSON missing baseline_checkpoint_path.")
    if os.path.abspath(saved_checkpoint) != os.path.abspath(VAL_BASELINE_CHECKPOINT):
        raise ValueError(
            "Baseline checkpoint path changed. "
            "Delete the old baseline JSON and regenerate it."
        )

    # Check that the referenced checkpoint still exists with same size/mtime
    if not os.path.isfile(saved_checkpoint):
        raise ValueError(
            f"Baseline checkpoint no longer exists: {saved_checkpoint}"
        )
    saved_size = data.get("baseline_checkpoint_size")
    if saved_size is None:
        raise ValueError(
            "Baseline JSON missing baseline_checkpoint_size."
        )
    saved_mtime = data.get("baseline_checkpoint_mtime")
    if saved_mtime is None:
        raise ValueError(
            "Baseline JSON missing baseline_checkpoint_mtime."
        )
    current_stat = os.stat(saved_checkpoint)
    if int(saved_size) != int(current_stat.st_size):
        raise ValueError(
            "Baseline checkpoint size changed. "
            "Delete the old baseline JSON and regenerate it."
        )
    if abs(float(saved_mtime) - float(current_stat.st_mtime)) > 1.0:
        raise ValueError(
            "Baseline checkpoint mtime changed. "
            "Delete the old baseline JSON and regenerate it."
        )

    return data


def generate_baseline(device, val_filenames):
    """Generate baseline metrics JSON from the fixed checkpoint."""
    if not os.path.isfile(VAL_BASELINE_CHECKPOINT):
        raise FileNotFoundError(f"Baseline checkpoint not found: {VAL_BASELINE_CHECKPOINT}")

    # Build separate baseline modules
    baseline_modules = build_model(device)

    try:
        # Load checkpoint to CPU first to reduce peak GPU memory
        checkpoint = torch.load(VAL_BASELINE_CHECKPOINT, map_location="cpu")

        # Strict metadata validation (reuse test.py logic)
        validate_checkpoint_metadata(checkpoint, baseline_modules[1])

        # Strict key validation (all 9 keys must exist) and load to modules on GPU
        load_modules_from_checkpoint(baseline_modules, checkpoint)

        # Release CPU checkpoint immediately after loading
        del checkpoint
        gc.collect()

        val_dataset = PairedValidationDataset(
            visible_dir=VAL_VISIBLE_DIR,
            infrared_dir=VAL_INFRARED_DIR,
            visible_rgb_dir=VAL_VISIBLE_RGB_DIR,
            expected_pairs=VAL_EXPECTED_PAIRS,
        )
        valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=torch.cuda.is_available())

        metrics = run_validation(baseline_modules, valloader, device)

        checkpoint_stat = os.stat(VAL_BASELINE_CHECKPOINT)
        baseline_data = {
            **metrics,
            "validation_filenames": val_filenames,
            "validation_count": len(val_filenames),
            "baseline_checkpoint_path": VAL_BASELINE_CHECKPOINT,
            "baseline_checkpoint_size": checkpoint_stat.st_size,
            "baseline_checkpoint_mtime": checkpoint_stat.st_mtime,
        }

        baseline_json_dir = os.path.dirname(VAL_BASELINE_JSON) or "."
        os.makedirs(baseline_json_dir, exist_ok=True)
        with open(VAL_BASELINE_JSON, 'w') as f:
            json.dump(baseline_data, f, indent=2)

        print(f"Baseline saved to: {VAL_BASELINE_JSON}")
        print(f"Baseline metrics: {metrics}")

    finally:
        del baseline_modules
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Training forward pass
# ---------------------------------------------------------------------------

def training_forward(modules, data_vis, data_ir, data_vis_clip):
    """Shared forward pass for training. Returns fused_image."""
    shared_encoder, intent_generator, frequency_fusion, frequency_pyramid_adapter, \
        spatial_fusion, fsrc_l1, fsrc_l2, fsrc_l3, fusion_decoder = modules

    vis_spa, vis_freq, _ = shared_encoder(data_vis)
    ir_spa, ir_freq, _ = shared_encoder(data_ir)
    i_deg, i_fus, _ = intent_generator(data_vis_clip, vis_spa, ir_spa, vis_freq, ir_freq)
    fused_freq, _ = frequency_fusion(vis_freq, ir_freq, frequency_intent=i_deg)
    _, spatial_pyramid, _ = spatial_fusion(vis_spa, ir_spa, i_fus, return_aux=True, return_pyramid=True)
    freq_pyramid = frequency_pyramid_adapter(fused_freq, target_pyramid=spatial_pyramid)
    d_l1, _ = fsrc_l1(freq_pyramid['l1'], spatial_pyramid['l1'])
    d_l2, _ = fsrc_l2(freq_pyramid['l2'], spatial_pyramid['l2'])
    d_l3, _ = fsrc_l3(freq_pyramid['l3'], spatial_pyramid['l3'])
    fused_image, _ = fusion_decoder(d_l1, d_l2, d_l3)

    return fused_image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Validation dataset (used for both baseline and per-epoch validation) ---
    val_dataset = PairedValidationDataset(
        visible_dir=VAL_VISIBLE_DIR,
        infrared_dir=VAL_INFRARED_DIR,
        visible_rgb_dir=VAL_VISIBLE_RGB_DIR,
        expected_pairs=VAL_EXPECTED_PAIRS,
    )
    val_filenames = list(val_dataset.filenames)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                           pin_memory=torch.cuda.is_available())

    # --- Baseline ---
    if not os.path.isfile(VAL_BASELINE_JSON):
        print("Baseline JSON not found, generating from checkpoint...")
        generate_baseline(device, val_filenames)

    baseline_data = validate_baseline_json(VAL_BASELINE_JSON, val_filenames)
    baseline_metrics = {k: baseline_data[k] for k in METRIC_WEIGHTS}
    print(f"Baseline metrics: {baseline_metrics}")

    # --- Loss criteria ---
    criteria_fusion = Fusionloss().to(device)
    criteria_ssim = SimpleSSIMLoss(window_size=11).to(device)
    criteria_freq = FrequencyConsistencyLoss(low_weight=1.0, high_weight=1.0).to(device)
    criteria_correlation = CorrelationConsistencyLoss().to(device)
    criteria_local_contrast = LocalContrastLoss(window_size=7, eps=1e-6).to(device)

    num_epochs = 50

    # --- Build model ---
    modules = build_model(device)

    # --- Single AdamW optimizer ---
    trainable_params = []
    for module in modules:
        trainable_params.extend([p for p in module.parameters() if p.requires_grad])

    param_ids = [id(p) for p in trainable_params]
    if len(param_ids) != len(set(param_ids)):
        raise RuntimeError("Duplicate trainable parameters were found across modules.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=BASE_LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        num_epochs=num_epochs,
        warmup_epochs=WARMUP_EPOCHS,
        base_lr=BASE_LR,
        min_lr=MIN_LR,
    )

    # --- Data ---
    trainloader = DataLoader(H5Dataset(TRAIN_H5_PATH), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # --- Checkpoint paths ---
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
    best_checkpoint_path = os.path.join(MODEL_DIRECTORY, f'{CHECKPOINT_TAG}_TextIntentDualDomainFusion_best.pth')
    latest_checkpoint_path = os.path.join(MODEL_DIRECTORY, f'{CHECKPOINT_TAG}_TextIntentDualDomainFusion_latest.pth')

    # --- CSV ---
    csv_path = os.path.join(MODEL_DIRECTORY, f'training_history_{timestamp}.csv')
    init_csv(csv_path)

    # --- Best tracking ---
    best_val_score = -float("inf")
    best_epoch = -1
    best_metrics = None
    best_ratios = None

    torch.backends.cudnn.benchmark = True
    previous_time = time.time()

    for epoch in range(num_epochs):
        # Read LR at START of epoch (before training)
        current_lr = optimizer.param_groups[0]["lr"]

        # === Training ===
        for module in modules:
            module.train()

        # Accumulators for per-batch weighted losses
        epoch_total = 0.0
        epoch_weighted_fusion = 0.0
        epoch_weighted_ssim = 0.0
        epoch_weighted_freq = 0.0
        epoch_weighted_corr = 0.0
        epoch_weighted_local_ctr = 0.0
        epoch_fusion_raw = 0.0
        epoch_ssim_raw = 0.0
        epoch_freq_raw = 0.0
        epoch_corr_raw = 0.0
        epoch_local_ctr_raw = 0.0
        grad_norm_sum = 0.0
        num_batches = 0

        for index, (data_vis, data_ir, data_vis_rgb_raw) in enumerate(trainloader):
            data_vis, data_ir = data_vis.to(device), data_ir.to(device)
            data_vis_clip = preprocess_clip_rgb(data_vis_rgb_raw.to(device))

            optimizer.zero_grad(set_to_none=True)

            fused_image = training_forward(modules, data_vis, data_ir, data_vis_clip)

            fusion_loss, _, _ = criteria_fusion(data_vis, data_ir, fused_image)
            ssim_loss = criteria_ssim(fused_image, data_vis) + criteria_ssim(fused_image, data_ir)
            freq_loss, _, _ = criteria_freq(data_vis, data_ir, fused_image)
            correlation_loss = criteria_correlation(data_vis, data_ir, fused_image)
            local_contrast_loss = criteria_local_contrast(data_vis, data_ir, fused_image)

            current_freq_weight = get_frequency_weight(
                epoch_index=epoch,
                warmup_epochs=FREQUENCY_WARMUP_EPOCHS,
                final_weight=COEFF_FREQ_FINAL,
            )

            weighted_fusion = COEFF_FUSION * fusion_loss
            weighted_ssim = COEFF_SSIM * ssim_loss
            weighted_freq = current_freq_weight * freq_loss
            weighted_corr = COEFF_CORRELATION * correlation_loss
            weighted_local_contrast = COEFF_LOCAL_CONTRAST * local_contrast_loss

            loss = weighted_fusion + weighted_ssim + weighted_freq + weighted_corr + weighted_local_contrast
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GLOBAL_GRAD_CLIP, norm_type=2.0)
            optimizer.step()

            # Accumulate per-batch values
            epoch_total += loss.item()
            epoch_weighted_fusion += weighted_fusion.item()
            epoch_weighted_ssim += weighted_ssim.item()
            epoch_weighted_freq += weighted_freq.item()
            epoch_weighted_corr += weighted_corr.item()
            epoch_weighted_local_ctr += weighted_local_contrast.item()
            epoch_fusion_raw += fusion_loss.item()
            epoch_ssim_raw += ssim_loss.item()
            epoch_freq_raw += freq_loss.item()
            epoch_corr_raw += correlation_loss.item()
            epoch_local_ctr_raw += local_contrast_loss.item()
            grad_norm_sum += float(grad_norm) if not isinstance(grad_norm, torch.Tensor) else grad_norm.item()
            num_batches += 1

            batches_done = epoch * len(trainloader) + index
            batches_left = num_epochs * len(trainloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - previous_time))
            previous_time = time.time()
            sys.stdout.write(
                '\r[Epoch %d/%d] [Batch %d/%d] [loss: %.6f] [fusion: %.4f] [ssim: %.4f] [freq: %.4f] [corr: %.4f] [ctr: %.4f] ETA: %.10s' % (
                    epoch + 1, num_epochs, index + 1, len(trainloader), loss.item(),
                    fusion_loss.item(), ssim_loss.item(), freq_loss.item(),
                    correlation_loss.item(), local_contrast_loss.item(), time_left,
                )
            )

        # scheduler step AFTER training of this epoch
        scheduler.step()

        # Compute epoch averages
        n = max(num_batches, 1)
        avg_total = epoch_total / n
        avg_fusion = epoch_fusion_raw / n
        avg_weighted_fusion = epoch_weighted_fusion / n
        avg_ssim = epoch_ssim_raw / n
        avg_weighted_ssim = epoch_weighted_ssim / n
        avg_freq = epoch_freq_raw / n
        avg_weighted_freq = epoch_weighted_freq / n
        avg_corr = epoch_corr_raw / n
        avg_weighted_corr = epoch_weighted_corr / n
        avg_local_ctr = epoch_local_ctr_raw / n
        avg_weighted_local_ctr = epoch_weighted_local_ctr / n
        avg_grad_norm = grad_norm_sum / n
        current_freq_w = get_frequency_weight(epoch_index=epoch, warmup_epochs=FREQUENCY_WARMUP_EPOCHS, final_weight=COEFF_FREQ_FINAL)

        print(f"\n[Epoch {epoch + 1}/{num_epochs}] "
              f"lr={current_lr:.6e} "
              f"grad_norm={avg_grad_norm:.4f} "
              f"total={avg_total:.6f} "
              f"fusion={avg_fusion:.6f} w_fus={avg_weighted_fusion:.6f} "
              f"ssim={avg_ssim:.6f} w_ssim={avg_weighted_ssim:.6f} "
              f"freq={avg_freq:.6f} fw={current_freq_w:.6f} w_freq={avg_weighted_freq:.6f} "
              f"corr={avg_corr:.6f} w_corr={avg_weighted_corr:.6f} "
              f"ctr={avg_local_ctr:.6f} w_ctr={avg_weighted_local_ctr:.6f}")

        # === Validation ===
        val_metrics = run_validation(modules, valloader, device)

        # Compute validation score with exception handling for non-finite metrics
        try:
            val_score, val_ratios = compute_validation_score(
                metrics=val_metrics,
                baseline_metrics=baseline_metrics,
                metric_weights=METRIC_WEIGHTS,
            )
        except (ValueError, KeyError) as error:
            print(f"[WARNING] Validation scoring failed: {error}")
            val_score = float("nan")
            val_ratios = {name: float("nan") for name in METRIC_WEIGHTS}

        worst_ratio = min(val_ratios.values())

        print(f"[Validation] EN={val_metrics['EN']:.4f} SD={val_metrics['SD']:.4f} "
              f"SCD={val_metrics['SCD']:.4f} VIF={val_metrics['VIF']:.4f} "
              f"QABF={val_metrics['QABF']:.4f} MI={val_metrics['MI']:.4f} "
              f"score={val_score:.6f} worst_ratio={worst_ratio:.4f}")

        # --- Determine best ---
        finite_validation = (
            math.isfinite(val_score)
            and all(math.isfinite(float(v)) for v in val_metrics.values())
        )

        is_best = False
        if finite_validation and val_score > best_val_score:
            is_best = True
            best_val_score = val_score
            best_epoch = epoch + 1
            best_metrics = dict(val_metrics)
            best_ratios = dict(val_ratios)

            save_checkpoint(
                path=best_checkpoint_path,
                modules=modules,
                optimizer=optimizer, scheduler=scheduler,
                epoch=best_epoch, val_score=best_val_score,
                val_metrics=best_metrics, val_ratios=best_ratios,
                is_best=True,
            )
            print(f"[Best] epoch={best_epoch} score={best_val_score:.6f} "
                  f"metrics={best_metrics} ratios={best_ratios}")
            print(f"[Best] saved to {best_checkpoint_path}")
        elif not finite_validation:
            print(f"[WARNING] Validation metrics non-finite (val_score={val_score}), "
                  f"skipping best checkpoint update.")

        # Always save latest
        save_checkpoint(
            path=latest_checkpoint_path,
            modules=modules,
            optimizer=optimizer, scheduler=scheduler,
            epoch=epoch + 1, val_score=val_score,
            val_metrics=val_metrics, val_ratios=val_ratios,
            is_best=is_best,
        )

        # --- CSV ---
        append_csv(
            csv_path,
            epoch=epoch + 1,
            learning_rate=current_lr,
            avg_grad_norm=avg_grad_norm,
            train_total=avg_total,
            train_fusion=avg_fusion,
            train_weighted_fusion=avg_weighted_fusion,
            train_ssim=avg_ssim,
            train_weighted_ssim=avg_weighted_ssim,
            train_frequency=avg_freq,
            frequency_weight=current_freq_w,
            train_weighted_frequency=avg_weighted_freq,
            train_correlation=avg_corr,
            train_weighted_correlation=avg_weighted_corr,
            train_local_contrast=avg_local_ctr,
            train_weighted_local_contrast=avg_weighted_local_ctr,
            val_EN=val_metrics['EN'],
            val_SD=val_metrics['SD'],
            val_SCD=val_metrics['SCD'],
            val_VIF=val_metrics['VIF'],
            val_QABF=val_metrics['QABF'],
            val_MI=val_metrics['MI'],
            val_score=val_score,
            worst_ratio=worst_ratio,
            is_best=1 if is_best else 0,
            best_epoch_so_far=best_epoch,
            best_score_so_far=best_val_score,
        )

    # === Training completed ===
    print(f"\n{'='*60}")
    print(f"Training completed")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation score: {best_val_score:.6f}")
    print(f"Best validation metrics: {best_metrics}")
    print(f"Best validation ratios: {best_ratios}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")
    print(f"Training history: {csv_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
