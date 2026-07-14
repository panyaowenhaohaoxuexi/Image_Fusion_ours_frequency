import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BASE_DIR = Path("E:/yizuo_SCI/3_methods_pth_images")
OUTPUT_ROOT = Path("E:/yizuo_SCI/4_方案/图/结果可视化图/对比实验/对比实验图")

# Edit these defaults directly when you want to switch image, output path, crop mode, or crop size.
IMAGE_ID = "7-113"
DATASET = "datasets_ours"
CROP_MODE = "auto"  # "auto" or "manual"
CROP_WIDTH = 120
CROP_HEIGHT = 60
TARGET_METHOD = "Complete_v10"
TOP_K = 20
SELECTED_CANDIDATE_INDEX = 18
SCAN_STEP = 8
PREVIEW_SCALE = 0.5

METHODS = [
    "Complete_v10",
    "1_Tardal",
    "2_DeF",
    "3_MetaF",
    "4_Text-IF",
    "5_SFDFusion",
    "6_FISCNet",
    "7_SAGE",
    "8_SHIP++",
    "9_PSFusion",
    "10_TDFusion",
]

COMPARISON_METHODS = {
    "1_Tardal",
    "2_DeF",
    "3_MetaF",
    "4_Text-IF",
    "5_SFDFusion",
    "6_FISCNet",
    "7_SAGE",
    "8_SHIP++",
    "9_PSFusion",
    "10_TDFusion",
}

DATASET_DIRS = {
    "datasets_ours": "datasets_ours",
}

ORIGINAL_DATASET_ROOTS = {
    "datasets_ours": Path("E:/yizuo_SCI/2_Datasets/6_datasets-ours"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export full-size method comparison images and automatically choose a high-difference crop."
    )
    parser.add_argument("--image-id", default=IMAGE_ID, help="Source image stem, for example 1-036.")
    parser.add_argument("--dataset", default=DATASET, choices=sorted(DATASET_DIRS.keys()))
    parser.add_argument("--crop-width", type=int, default=CROP_WIDTH)
    parser.add_argument("--crop-height", type=int, default=CROP_HEIGHT)
    parser.add_argument("--target-method", default=TARGET_METHOD, choices=METHODS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--selected-candidate-index", type=int, default=SELECTED_CANDIDATE_INDEX)
    parser.add_argument("--scan-step", type=int, default=SCAN_STEP, help="Sliding-window stride in pixels.")
    parser.add_argument("--preview-scale", type=float, default=PREVIEW_SCALE, help="Scale used for the preview contact sheet.")
    return parser.parse_args()


def method_root(method):
    if method == "Complete_v10":
        return BASE_DIR / "Ours" / "Complete" / "v10" / "images"
    if method in COMPARISON_METHODS:
        return BASE_DIR / method / "images"
    raise ValueError(f"Unknown method: {method}")


def first_existing(root, image_id, exts):
    for ext in exts:
        candidate = root / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return root / f"{image_id}{exts[0]}"


def source_path(method, dataset, image_id):
    if dataset != "datasets_ours":
        raise ValueError("This comparison script is configured for datasets_ours only")

    if method == "Complete_v10":
        return first_existing(method_root(method) / "6_datasets-ours" / "color", image_id, [".png", ".jpg", ".jpeg", ".bmp"])

    if method == "3_MetaF":
        return first_existing(method_root(method) / "datasets_ours", image_id, [".jpg", ".png", ".jpeg", ".bmp"])

    if method == "4_Text-IF":
        return method_root(method) / "datasets_ours" / "text_fusion" / f"{image_id}.png"

    if method in COMPARISON_METHODS:
        return first_existing(method_root(method) / "datasets_ours", image_id, [".png", ".jpg", ".jpeg", ".bmp"])

    raise ValueError(f"Unknown method: {method}")


def output_name(image_id, method):
    return f"{image_id}_{method}.png"


def case_dirs(dataset, image_id):
    case_root = OUTPUT_ROOT / f"{dataset}_{image_id}"
    return {
        "case_root": case_root,
        "full_dir": case_root / "完整原图",
        "crop_dir": case_root / "裁剪对比图",
        "candidate_dir": case_root / "候选区域预览",
    }


def find_original_image(dataset, image_id, modality):
    root = ORIGINAL_DATASET_ROOTS[dataset] / modality
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        candidate = root / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return root / f"{image_id}.png"


def ensure_sources_exist(sources):
    missing = [(method, path) for method, path in sources if not path.exists()]
    print("Source check:")
    for method, path in sources:
        print(f"  {method}: {'OK' if path.exists() else 'MISSING'} - {path}")
    if missing:
        missing_text = "\n".join(f"{method}: {path}" for method, path in missing)
        raise FileNotFoundError(f"Missing source images:\n{missing_text}")


def ensure_originals_exist(dataset, image_id):
    originals = {
        "IR": find_original_image(dataset, image_id, "ir"),
        "VIS": find_original_image(dataset, image_id, "vi"),
    }
    missing = [(name, path) for name, path in originals.items() if not path.exists()]
    print("\nOriginal IR/VIS check:")
    for name, path in originals.items():
        print(f"  {name}: {'OK' if path.exists() else 'MISSING'} - {path}")
    if missing:
        missing_text = "\n".join(f"{name}: {path}" for name, path in missing)
        raise FileNotFoundError(f"Missing original images:\n{missing_text}")
    return originals


def majority_size(sources):
    sizes = []
    for _, path in sources:
        with Image.open(path) as image:
            sizes.append(image.size)

    size_counts = Counter(sizes)
    target_size, target_count = size_counts.most_common(1)[0]
    print("\nResolution counts:")
    for size, count in size_counts.most_common():
        print(f"  {size}: {count}")
    print(f"Majority resolution: {target_size} ({target_count}/{len(sources)})")
    return target_size


def save_full_images(sources, originals, image_id, target_size, full_dir):
    full_dir.mkdir(parents=True, exist_ok=True)
    full_paths = {}
    print(f"\nWriting full images: {full_dir}")
    for method, source in sources:
        with Image.open(source) as image:
            image = image.convert("RGB")
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.BICUBIC)
            save_path = full_dir / output_name(image_id, method)
            image.save(save_path)
            full_paths[method] = save_path
            print(f"  {save_path.name}: {image.size}")

    original_paths = {}
    for label, source in originals.items():
        with Image.open(source) as image:
            image = image.convert("RGB")
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.BICUBIC)
            save_path = full_dir / f"{image_id}_{label}.png"
            image.save(save_path)
            original_paths[label] = save_path
            print(f"  {save_path.name}: {image.size}")
    return full_paths, original_paths


def to_gray_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def integral_image(values):
    return values.cumsum(axis=0).cumsum(axis=1)


def rect_sum(integral, x, y, width, height):
    x2 = x + width - 1
    y2 = y + height - 1
    total = integral[y2, x2]
    if x > 0:
        total -= integral[y2, x - 1]
    if y > 0:
        total -= integral[y - 1, x2]
    if x > 0 and y > 0:
        total += integral[y - 1, x - 1]
    return float(total)


def gradient_magnitude(gray):
    gy, gx = np.gradient(gray)
    return np.sqrt(gx * gx + gy * gy)


def local_mean_std(gray, window=15):
    """Compute per-pixel local mean & std maps via integral images.

    Pure-numpy vectorised implementation — O(1) per image (no Python loops).
    """
    pad = window // 2
    padded = np.pad(gray, pad, mode="edge")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral_sq = (padded * padded).cumsum(axis=0).cumsum(axis=1)

    # Pad integral images with one row/col of zeros so that boundary queries
    # (where i-1 or j-1 would be -1) naturally return 0 without branching.
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    integral_sq = np.pad(integral_sq, ((1, 0), (1, 0)), mode="constant", constant_values=0)

    h, w = gray.shape
    area = float(window * window)

    # four-corner slicing: br - bl - tr + tl  (vectorised over all pixels)
    sums = (
        integral[window : h + window, window : w + window]   # bottom-right
        - integral[window : h + window, :w]                   # bottom-left
        - integral[:h, window : w + window]                   # top-right
        + integral[:h, :w]                                    # top-left
    )
    mean_map = sums / area

    sq_sums = (
        integral_sq[window : h + window, window : w + window]
        - integral_sq[window : h + window, :w]
        - integral_sq[:h, window : w + window]
        + integral_sq[:h, :w]
    )
    sq_mean_map = sq_sums / area

    var_map = np.maximum(sq_mean_map - mean_map * mean_map, 0.0)
    std_map = np.sqrt(var_map)
    return mean_map.astype(np.float32), std_map.astype(np.float32)


def local_contrast_normalize(gray, window=15, eps=1e-4):
    """Locally contrast-normalise an image: subtract local mean, divide by local std.

    After normalisation, flat regions (sky, wall) have values near 0 regardless
    of their absolute brightness; only structural features remain prominent.
    """
    mean, std = local_mean_std(gray, window)
    return ((gray - mean) / np.maximum(std, eps)).astype(np.float32)


def build_score_map(gray_images, target_method):
    # --- locally contrast-normalise every method before comparing ---
    # This removes global brightness offsets so the sky no longer dominates.
    lcn_images = {m: local_contrast_normalize(img) for m, img in gray_images.items()}

    target = lcn_images[target_method]
    others = np.stack([img for m, img in lcn_images.items() if m != target_method], axis=0)
    others_mean = others.mean(axis=0)

    target_diff = np.abs(target - others_mean)
    all_variance = np.stack(list(lcn_images.values()), axis=0).var(axis=0)

    # sharpness and texture are still computed on the original target
    # because they measure the presence of detail (not inter-method difference)
    orig_target = gray_images[target_method]
    sharpness = gradient_magnitude(orig_target)
    texture = np.abs(orig_target - orig_target.mean())
    saturation = ((orig_target < 8) | (orig_target > 247)).astype(np.float32)

    def normalize(values):
        low, high = np.percentile(values, [1, 99])
        if high <= low:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip((values - low) / (high - low), 0, 1).astype(np.float32)

    score = (
        0.42 * normalize(target_diff)
        + 0.25 * normalize(sharpness)
        + 0.20 * normalize(all_variance)
        + 0.13 * normalize(texture)
        - 0.25 * saturation
    )
    return score.astype(np.float32)


def choose_candidate_crops(full_paths, target_method, crop_width, crop_height, scan_step, top_k):
    gray_images = {method: to_gray_array(path) for method, path in full_paths.items()}
    height, width = next(iter(gray_images.values())).shape
    if crop_width > width or crop_height > height:
        raise ValueError(f"Crop size {crop_width}x{crop_height} exceeds image size {width}x{height}")

    score_map = build_score_map(gray_images, target_method)
    integral = integral_image(score_map)

    candidates = []
    seen = set()

    def add_candidate(x, y):
        if (x, y) in seen:
            return
        seen.add((x, y))
        score = rect_sum(integral, x, y, crop_width, crop_height) / (crop_width * crop_height)
        candidates.append(
            {
                "score": score,
                "x": x,
                "y": y,
                "box": (x, y, x + crop_width, y + crop_height),
            }
        )

    for y in range(0, height - crop_height + 1, scan_step):
        for x in range(0, width - crop_width + 1, scan_step):
            add_candidate(x, y)

    if (height - crop_height) % scan_step:
        y = height - crop_height
        for x in range(0, width - crop_width + 1, scan_step):
            add_candidate(x, y)
    if (width - crop_width) % scan_step:
        x = width - crop_width
        for y in range(0, height - crop_height + 1, scan_step):
            add_candidate(x, y)

    if not candidates:
        raise RuntimeError("No crop candidates were generated")

    def iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter = inter_w * inter_h
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-8)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    for candidate in candidates:
        if all(iou(candidate["box"], chosen["box"]) <= 0.35 for chosen in selected):
            selected.append(candidate)
        if len(selected) == top_k:
            break
    if len(selected) < top_k:
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == top_k:
                break
    print(f"\nTop-{len(selected)} crop candidates for {target_method}:")
    for index, candidate in enumerate(selected, start=1):
        print(
            f"  {index:02d}: x={candidate['x']}, y={candidate['y']}, "
            f"width={crop_width}, height={crop_height}, score={candidate['score']:.4f}"
        )
    return selected


def generate_crop_location_map(original_paths, crop_box, crop_dir, image_id):
    """Generate a full-image map highlighting where the crop region is located.

    Uses the VIS original as the base image, darkens the area outside the crop box,
    and draws a prominent border around the selected region with coordinate annotations.
    """
    vis_path = original_paths.get("VIS")
    if vis_path is None:
        print("  (no VIS original available for location map, skipping)")
        return

    x1, y1, x2, y2 = crop_box
    crop_w, crop_h = x2 - x1, y2 - y1

    with Image.open(vis_path) as vis_img:
        # --- darken everything OUTSIDE the crop box (spotlight effect) ---
        vis_rgba = vis_img.convert("RGBA")
        overlay = Image.new("RGBA", vis_img.size, (0, 0, 0, 140))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(crop_box, fill=(0, 0, 0, 0))
        vis_rgba = Image.alpha_composite(vis_rgba, overlay)
        vis_rgb = vis_rgba.convert("RGB")

        # --- draw crop box border (outer white + inner red for contrast) ---
        draw = ImageDraw.Draw(vis_rgb)
        # outer glow: white border around the crop box
        draw.rectangle(
            [x1 - 3, y1 - 3, x2 + 3, y2 + 3],
            outline=(255, 255, 255),
            width=3,
        )
        # inner line: bright red, thinner
        draw.rectangle(crop_box, outline=(255, 50, 50), width=2)

        # --- coordinate label ---
        label = f"Crop: ({x1}, {y1})  {crop_w} x {crop_h}"
        # place label above the box when there is room, otherwise below
        if y1 > 36:
            label_y = y1 - 26
        else:
            label_y = y2 + 8

        bbox = draw.textbbox((x1, label_y), label)
        pad = 4
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=(255, 50, 50),
        )
        draw.text((x1, label_y), label, fill="white")

        save_path = crop_dir / f"{image_id}_crop_location.png"
        vis_rgb.save(save_path)
        print(f"  {save_path.name}: {vis_rgb.size}  <-- location map")


def save_crops(full_paths, original_paths, image_id, crop_box, crop_dir):
    crop_dir.mkdir(parents=True, exist_ok=True)
    # remove old crop files so stale results don't accumulate
    old_files = list(crop_dir.glob(f"{image_id}_*.png"))
    for path in old_files:
        path.unlink()

    x1, y1, x2, y2 = crop_box
    crop_w, crop_h = x2 - x1, y2 - y1
    coord_suffix = f"_x{x1}_y{y1}_w{crop_w}_h{crop_h}"

    print(f"\nWriting crops: {crop_dir}")
    for method, full_path in full_paths.items():
        with Image.open(full_path) as image:
            crop = image.crop(crop_box)
            save_path = crop_dir / f"{image_id}_{method}{coord_suffix}.png"
            crop.save(save_path)
            print(f"  {save_path.name}: {crop.size}")
    for label, full_path in original_paths.items():
        with Image.open(full_path) as image:
            crop = image.crop(crop_box)
            save_path = crop_dir / f"{image_id}_{label}{coord_suffix}.png"
            crop.save(save_path)
            print(f"  {save_path.name}: {crop.size}")

    # --- location map so you can see where the crop came from ---
    generate_crop_location_map(original_paths, crop_box, crop_dir, image_id)


def make_crop_grid(full_paths, crop_box, target_method, scale=1.0):
    images = []
    for method, path in full_paths.items():
        with Image.open(path) as image:
            image = image.convert("RGB").crop(crop_box)
            scaled = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.BICUBIC,
            ) if scale != 1.0 else image
            images.append((method, scaled))

    tile_w = max(image.width for _, image in images)
    tile_h = max(image.height for _, image in images)

    # --- location inset: shrunk full image with crop box highlighted ---
    target_path = full_paths.get(target_method)
    if target_path is not None:
        with Image.open(target_path) as full_img:
            full_img = full_img.convert("RGB")
            # scale full image to the same height as crop tiles
            loc_h = tile_h
            loc_w = int(round(full_img.width * tile_h / full_img.height))
            loc_img = full_img.resize((loc_w, loc_h), Image.Resampling.BICUBIC)
            # draw the scaled crop box
            sx = loc_w / full_img.width
            sy = loc_h / full_img.height
            x1, y1, x2, y2 = crop_box
            scaled_box = (
                int(round(x1 * sx)), int(round(y1 * sy)),
                int(round(x2 * sx)), int(round(y2 * sy)),
            )
            loc_draw = ImageDraw.Draw(loc_img)
            loc_draw.rectangle(
                [scaled_box[0] - 1, scaled_box[1] - 1, scaled_box[2] + 1, scaled_box[3] + 1],
                outline=(255, 255, 255),
                width=2,
            )
            loc_draw.rectangle(scaled_box, outline=(255, 50, 50), width=1)
            # prepend as the first tile so it acts as a visual key
            images.insert(0, ("[ Location ]", loc_img))
            tile_w = max(tile_w, loc_w)

    label_h = 28
    cols = 4
    rows = (len(images) + cols - 1) // cols
    preview = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(preview)

    for index, (method, image) in enumerate(images):
        row, col = divmod(index, cols)
        x = col * tile_w
        y = row * (tile_h + label_h)
        preview.paste(image, (x, y))
        text_color = (180, 30, 30) if method.startswith("[") else "black"
        draw.text((x + 4, y + tile_h + 6), method, fill=text_color)

    return preview


def save_candidate_previews(full_paths, image_id, candidates, target_method, candidate_dir):
    candidate_dir.mkdir(parents=True, exist_ok=True)
    old_files = list(candidate_dir.glob(f"{image_id}_candidate_*.png"))
    for path in old_files:
        path.unlink()

    print(f"\nWriting candidate previews: {candidate_dir}")
    for index, candidate in enumerate(candidates, start=1):
        preview = make_crop_grid(full_paths, candidate["box"], target_method, scale=1.0)
        crop_width = candidate["box"][2] - candidate["box"][0]
        crop_height = candidate["box"][3] - candidate["box"][1]
        preview_path = candidate_dir / (
            f"{image_id}_candidate_{index:02d}_"
            f"x{candidate['x']}_y{candidate['y']}_"
            f"w{crop_width}_h{crop_height}_"
            f"score{candidate['score']:.4f}.png"
        )
        preview.save(preview_path)
        print(f"  {preview_path.name}")


def save_preview(full_paths, image_id, crop_box, target_method, preview_scale, case_root):
    x1, y1, x2, y2 = crop_box
    crop_w, crop_h = x2 - x1, y2 - y1
    images = []
    for method, path in full_paths.items():
        with Image.open(path) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            color = "yellow" if method == target_method else "red"
            draw.rectangle(crop_box, outline=color, width=6)
            # annotate the target method tile with crop coordinates
            if method == target_method:
                coord_label = f"({x1},{y1}) {crop_w}x{crop_h}"
                draw.rectangle(crop_box, outline=color, width=6)
                bbox = draw.textbbox((x1 + 6, y1 - 18), coord_label)
                draw.rectangle(
                    [bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2],
                    fill=color,
                )
                draw.text((x1 + 6, y1 - 18), coord_label, fill="black")
            else:
                draw.rectangle(crop_box, outline=color, width=6)
            scaled = image.resize(
                (int(image.width * preview_scale), int(image.height * preview_scale)),
                Image.Resampling.BICUBIC,
            )
            images.append((method, scaled))

    tile_w = max(image.width for _, image in images)
    tile_h = max(image.height for _, image in images)
    label_h = 28
    cols = 4
    rows = (len(images) + cols - 1) // cols
    preview = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(preview)

    for index, (method, image) in enumerate(images):
        row, col = divmod(index, cols)
        x = col * tile_w
        y = row * (tile_h + label_h)
        preview.paste(image, (x, y))
        draw.text((x + 4, y + tile_h + 6), method, fill="black")

    preview_path = case_root / f"{image_id}_auto_box_preview.png"
    preview.save(preview_path)
    print(f"\nPreview written: {preview_path}")


def validate_outputs(image_id, target_size, crop_size, top_k, dirs):
    full_dir = dirs["full_dir"]
    crop_dir = dirs["crop_dir"]
    candidate_dir = dirs["candidate_dir"]
    full_files = sorted(full_dir.glob(f"{image_id}_*.png"))
    crop_files = sorted(crop_dir.glob(f"{image_id}_*.png"))
    candidate_files = sorted(candidate_dir.glob(f"{image_id}_candidate_*.png"))

    full_expected = len(METHODS) + 2  # methods + IR + VIS
    crop_expected = full_expected  # same set, but cropped
    crop_total = crop_expected + 1  # +1 for the crop location map

    # separate the location map from actual crops for size validation
    crop_location_file = crop_dir / f"{image_id}_crop_location.png"
    actual_crop_files = [f for f in crop_files if f != crop_location_file]
    full_sizes = Counter(Image.open(path).size for path in full_files)
    crop_sizes = Counter(Image.open(path).size for path in actual_crop_files)

    has_location_map = crop_location_file.exists()

    print("\nValidation:")
    print(f"  Full image count:   {len(full_files)}/{full_expected}")
    print(f"  Full image sizes:   {dict(full_sizes)}")
    print(f"  Candidate previews: {len(candidate_files)}/{top_k}")
    print(f"  Crop count:         {len(crop_files)}/{crop_total}  (incl. location map)")
    print(f"  Location map:       {'OK' if has_location_map else 'MISSING'}")
    print(f"  Crop sizes:         {dict(crop_sizes)}")

    if len(full_files) != full_expected:
        raise RuntimeError("Incorrect full image count")
    if len(crop_files) != crop_total:
        raise RuntimeError(f"Incorrect crop count (expected {crop_total}, got {len(crop_files)})")
    if len(actual_crop_files) != crop_expected:
        raise RuntimeError(f"Incorrect crop count excluding location map (expected {crop_expected})")
    if len(candidate_files) != top_k:
        raise RuntimeError("Incorrect candidate preview count")
    if full_sizes != Counter({target_size: full_expected}):
        raise RuntimeError("Full image sizes are inconsistent")
    if crop_sizes != Counter({crop_size: crop_expected}):
        raise RuntimeError("Crop sizes are inconsistent")
    if not has_location_map:
        raise RuntimeError("Crop location map was not created")
    if not dirs["case_root"].exists():
        raise RuntimeError("Case root was not created")


def manual_crop_selection(full_paths, target_method, image_id):
    """Let the user draw a crop rectangle on one method's full image via matplotlib.

    Displays the *target_method* full image.  Drag to draw a rectangle,
    press Enter to confirm, Esc to reset.
    Returns ``(x1, y1, x2, y2)`` in pixel coordinates.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    target_path = full_paths[target_method]
    with Image.open(target_path) as img:
        full_img = np.asarray(img.convert("RGB"))

    state = {"crop_box": None}  # mutable closure so callbacks can write to it

    def on_select(eclick, erelease):
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        # ensure left→right, top→bottom ordering
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        state["crop_box"] = (x1, y1, x2, y2)
        print(
            f"  Box: ({x1}, {y1}) → ({x2}, {y2})  "
            f"[{x2 - x1} x {y2 - y1}]"
        )

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(full_img)
    ax.set_title(
        f"Draw crop box on [{target_method}] — {image_id}\n"
        "Drag to draw · Enter = confirm · Esc = reset",
        fontsize=11,
    )
    ax.axis("off")

    rs = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],  # left mouse button only
        minspanx=4,
        minspany=4,
        spancoords="pixels",
        interactive=True,
        props=dict(edgecolor="yellow", facecolor="none", linewidth=2.5),
    )

    def on_key(event):
        if event.key == "enter":
            if state["crop_box"] is None:
                print("  (no rectangle drawn yet — draw one first)")
            else:
                print(f"\n  ✓ confirmed: {state['crop_box']}")
                plt.close(fig)
        elif event.key == "escape":
            state["crop_box"] = None
            print("  (rectangle reset — draw a new one)")

    fig.canvas.mpl_connect("key_press_event", on_key)

    print(
        f"\nOpening interactive crop window for [{target_method}] — {image_id}...\n"
        "  Drag to draw · Enter = confirm · Esc = reset\n"
        "  Close the window to confirm the current box."
    )
    plt.tight_layout()
    plt.show()

    if state["crop_box"] is None:
        raise RuntimeError("No crop box was selected (close the window after drawing).")

    x1, y1, x2, y2 = state["crop_box"]
    crop_w, crop_h = x2 - x1, y2 - y1
    print(
        f"\nManual crop selected: x={x1}, y={y1}, "
        f"width={crop_w}, height={crop_h}"
    )
    return state["crop_box"]


def main():
    mode = CROP_MODE.lower().strip()
    if mode not in ("auto", "manual"):
        raise ValueError('CROP_MODE must be "auto" or "manual"')
    if TOP_K < 1:
        raise ValueError("TOP_K must be at least 1")
    if not 1 <= SELECTED_CANDIDATE_INDEX <= TOP_K:
        raise ValueError("SELECTED_CANDIDATE_INDEX must be within 1..TOP_K")

    image_id = IMAGE_ID
    dataset = DATASET
    target_method = TARGET_METHOD
    crop_width = CROP_WIDTH
    crop_height = CROP_HEIGHT
    top_k = TOP_K
    selected_candidate_index = SELECTED_CANDIDATE_INDEX
    scan_step = SCAN_STEP
    preview_scale = PREVIEW_SCALE

    dirs = case_dirs(dataset, image_id)
    sources = [(method, source_path(method, dataset, image_id)) for method in METHODS]

    ensure_sources_exist(sources)
    originals = ensure_originals_exist(dataset, image_id)
    target_size = majority_size(sources)
    full_paths, original_paths = save_full_images(
        sources,
        originals,
        image_id,
        target_size,
        dirs["full_dir"],
    )

    if mode == "auto":
        # --- Auto mode: score-based crop search ---
        candidates = choose_candidate_crops(
            full_paths,
            target_method,
            crop_width,
            crop_height,
            scan_step,
            top_k,
        )
        save_candidate_previews(
            full_paths, image_id, candidates,
            target_method, dirs["candidate_dir"],
        )
        selected_candidate = candidates[selected_candidate_index - 1]
        crop_box = selected_candidate["box"]
        print()
        print(
            f"Selected candidate {selected_candidate_index}: "
            f"x={selected_candidate['x']}, y={selected_candidate['y']}, "
            f"width={crop_width}, height={crop_height}, "
            f"score={selected_candidate['score']:.4f}"
        )
    else:
        # --- Manual mode: user draws the box ---
        crop_box = manual_crop_selection(full_paths, target_method, image_id)

    save_crops(full_paths, original_paths, image_id, crop_box, dirs["crop_dir"])
    save_preview(full_paths, image_id, crop_box, target_method, preview_scale, dirs["case_root"])
    validate_outputs(
        image_id, target_size,
        (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
        top_k, dirs,
    )
    print()
    print("Done.")


if __name__ == "__main__":
    main()
