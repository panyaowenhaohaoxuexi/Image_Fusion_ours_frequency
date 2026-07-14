import torch
import torch.nn.functional as F


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_clip_rgb(rgb_tensor: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Resize RGB [0, 1] data and pad in CLIP-normalized space."""
    if rgb_tensor.dim() == 3:
        rgb_tensor = rgb_tensor.unsqueeze(0)
    if rgb_tensor.dim() != 4 or rgb_tensor.shape[1] != 3:
        raise ValueError("Expected RGB tensor with shape (B,3,H,W) or (3,H,W).")

    rgb_tensor = rgb_tensor.float()
    _, _, height, width = rgb_tensor.shape
    scale = image_size / max(height, width)
    new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
    resized = F.interpolate(
        rgb_tensor,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)

    mean = resized.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    normalized = (resized - mean) / std
    pad_height, pad_width = image_size - new_height, image_size - new_width
    return F.pad(
        normalized,
        [pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2],
        mode="constant",
        value=0.0,
    ).float()
