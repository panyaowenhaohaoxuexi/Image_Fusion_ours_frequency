# -*- coding: utf-8 -*-
import datetime
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
    DECODER_INNER_DIM,
    DECODER_MAX_RESIDUAL_SCALE,
    DECODER_NUM_BLOCKS,
    FREQUENCY_WARMUP_EPOCHS,
    GLOBAL_GRAD_CLIP,
    METRIC_WEIGHTS,
    MIN_LR,
    MODEL_DIRECTORY,
    TRAIN_H5_PATH,
    USE_CLIP_IMAGE_QUERY,
    VAL_EXPECTED_PAIRS,
    VAL_INFRARED_DIR,
    VAL_VISIBLE_DIR,
    VAL_VISIBLE_RGB_DIR,
    VALIDATION_START_EPOCH,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    validate_runtime_config,
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
    save_checkpoint,
    should_run_validation,
    should_update_best,
    validate_positive_baseline_metrics,
)
from utils.val_metrics import compute_val_metrics
from utils.validation_dataset import PairedValidationDataset


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ---------------------------------------------------------------------------
# Model construction
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
        channels=64, out_channels=1, inner_dim=DECODER_INNER_DIM, num_blocks=DECODER_NUM_BLOCKS,
        max_residual_scale=DECODER_MAX_RESIDUAL_SCALE, num_heads=1, ffn_expansion_factor=2.0,
    )).to(device)
    return encoder, intent_generator, frequency_fusion, frequency_pyramid_adapter, spatial_fusion, fsrc_l1, fsrc_l2, fsrc_l3, fusion_decoder


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
                 metric_vis_u8, metric_ir_u8, filenames) in valloader:
                filename = filenames[0]
                data_vis = data_vis.to(device)
                data_ir = data_ir.to(device)
                data_vis_clip = preprocess_clip_rgb(data_vis_rgb_raw.to(device))

                vis_spa, vis_freq, _ = modules[0](data_vis)
                ir_spa, ir_freq, _ = modules[0](data_ir)
                i_deg, i_fus, _ = modules[1](data_vis_clip, vis_spa, ir_spa, vis_freq, ir_freq)
                fused_freq, _ = modules[2](vis_freq, ir_freq, frequency_intent=i_deg)
                _, spatial_pyramid, spatial_aux = modules[4](vis_spa, ir_spa, i_fus, return_aux=True, return_pyramid=True)
                freq_pyramid = modules[3](fused_freq, target_pyramid=spatial_pyramid)
                d_l1, _ = modules[5](freq_pyramid['l1'], spatial_pyramid['l1'])
                d_l2, _ = modules[6](freq_pyramid['l2'], spatial_pyramid['l2'])
                d_l3, _ = modules[7](freq_pyramid['l3'], spatial_pyramid['l3'])
                fused_image, _ = modules[8](
                    d_l1, d_l2, d_l3, image_vis=data_vis, image_ir=data_ir,
                    weight_ir=spatial_aux['weight_multiscale'],
                )

                metrics = compute_val_metrics(
                    fused=fused_image,
                    vis_u8=metric_vis_u8,
                    ir_u8=metric_ir_u8,
                )
                for k in METRIC_WEIGHTS:
                    if k not in metrics:
                        raise KeyError(f"Validation metric {k} missing for {filename}")
                    value = float(metrics[k])
                    if not math.isfinite(value):
                        raise FloatingPointError(
                            f"Validation metric {k} is not finite for {filename}: {value}"
                        )
                    all_metrics[k].append(value)

    finally:
        for m, mode in zip(modules, previous_modes):
            m.train(mode)

    if any(len(values) == 0 for values in all_metrics.values()):
        raise RuntimeError("Validation loader produced no metric results.")
    avg_metrics = {k: float(sum(v) / len(v)) for k, v in all_metrics.items()}
    return avg_metrics


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
    _, spatial_pyramid, spatial_aux = spatial_fusion(vis_spa, ir_spa, i_fus, return_aux=True, return_pyramid=True)
    freq_pyramid = frequency_pyramid_adapter(fused_freq, target_pyramid=spatial_pyramid)
    d_l1, _ = fsrc_l1(freq_pyramid['l1'], spatial_pyramid['l1'])
    d_l2, _ = fsrc_l2(freq_pyramid['l2'], spatial_pyramid['l2'])
    d_l3, _ = fsrc_l3(freq_pyramid['l3'], spatial_pyramid['l3'])
    fused_image, _ = fusion_decoder(
        d_l1, d_l2, d_l3, image_vis=data_vis, image_ir=data_ir,
        weight_ir=spatial_aux['weight_multiscale'],
    )

    return fused_image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    validate_runtime_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f'[Model] version={CHECKPOINT_TAG} decoder_inner_dim={DECODER_INNER_DIM} '
        f'decoder_num_blocks={DECODER_NUM_BLOCKS} decoder_max_residual_scale={DECODER_MAX_RESIDUAL_SCALE}'
    )

    # --- Fixed validation dataset ---
    val_dataset = PairedValidationDataset(
        visible_dir=VAL_VISIBLE_DIR,
        infrared_dir=VAL_INFRARED_DIR,
        visible_rgb_dir=VAL_VISIBLE_RGB_DIR,
        expected_pairs=VAL_EXPECTED_PAIRS,
    )
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8,
                           pin_memory=torch.cuda.is_available())

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

    # --- Validation baseline and best tracking ---
    baseline_metrics = None
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

        current_epoch = epoch + 1
        val_metrics = None
        val_score = None
        val_ratios = None
        worst_ratio = None
        is_best = False

        if should_run_validation(current_epoch, VALIDATION_START_EPOCH):
            val_metrics = run_validation(modules, valloader, device)

            if baseline_metrics is None:
                try:
                    baseline_metrics = validate_positive_baseline_metrics(
                        val_metrics,
                        METRIC_WEIGHTS,
                    )
                except (KeyError, ValueError) as error:
                    raise RuntimeError(
                        f"Failed to initialize validation baseline "
                        f"at epoch {current_epoch}: {error}"
                    ) from error

                print(
                    f"[Validation Baseline] Initialized from epoch "
                    f"{current_epoch}: {baseline_metrics}"
                )

            try:
                val_score, val_ratios = compute_validation_score(
                    metrics=val_metrics,
                    baseline_metrics=baseline_metrics,
                    metric_weights=METRIC_WEIGHTS,
                )
                worst_ratio = min(val_ratios.values())
                is_best = should_update_best(
                    val_score,
                    best_val_score,
                    val_metrics,
                )
            except (KeyError, ValueError) as error:
                print(
                    f"[WARNING] Validation scoring failed "
                    f"at epoch {current_epoch}: {error}"
                )
                val_score = float("nan")
                val_ratios = {
                    name: float("nan")
                    for name in METRIC_WEIGHTS
                }
                worst_ratio = float("nan")
                is_best = False

            print(f"[Validation] EN={val_metrics['EN']:.4f} SD={val_metrics['SD']:.4f} "
                  f"SCD={val_metrics['SCD']:.4f} VIF={val_metrics['VIF']:.4f} "
                  f"QABF={val_metrics['QABF']:.4f} MI={val_metrics['MI']:.4f} "
                  f"score={val_score:.6f} worst_ratio={worst_ratio:.4f}")

            if is_best:
                best_val_score = float(val_score)
                best_epoch = current_epoch
                best_metrics = dict(val_metrics)
                best_ratios = dict(val_ratios)

                save_checkpoint(
                    path=best_checkpoint_path,
                    modules=modules,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=best_epoch,
                    val_score=best_val_score,
                    val_metrics=best_metrics,
                    val_ratios=best_ratios,
                    is_best=True,
                )
                print(f"[Best] epoch={best_epoch} score={best_val_score:.6f} "
                      f"metrics={best_metrics} ratios={best_ratios}")
                print(f"[Best] saved to {best_checkpoint_path}")
        else:
            print(
                f"[Validation] Skipped at epoch {current_epoch}; "
                f"validation starts after training epoch "
                f"{VALIDATION_START_EPOCH}."
            )

        save_checkpoint(
            path=latest_checkpoint_path,
            modules=modules,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=current_epoch,
            val_score=val_score,
            val_metrics=val_metrics,
            val_ratios=val_ratios,
            is_best=is_best,
        )

        csv_val_metrics = (
            {name: "" for name in METRIC_WEIGHTS}
            if val_metrics is None else val_metrics
        )
        csv_val_score = "" if val_score is None else val_score
        csv_worst_ratio = "" if worst_ratio is None else worst_ratio
        csv_best_score = "" if best_epoch < 0 else best_val_score

        # --- CSV ---
        append_csv(
            csv_path,
            epoch=current_epoch,
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
            val_EN=csv_val_metrics['EN'],
            val_SD=csv_val_metrics['SD'],
            val_SCD=csv_val_metrics['SCD'],
            val_VIF=csv_val_metrics['VIF'],
            val_QABF=csv_val_metrics['QABF'],
            val_MI=csv_val_metrics['MI'],
            val_score=csv_val_score,
            worst_ratio=csv_worst_ratio,
            is_best=1 if is_best else 0,
            best_epoch_so_far=best_epoch,
            best_score_so_far=csv_best_score,
        )

    # === Training completed ===
    print(f"\n{'='*60}")
    print(f"Training completed")
    if best_epoch < 0:
        print("No best checkpoint was produced because validation never ran.")
    else:
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
