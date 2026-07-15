# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip  # type: ignore
except Exception:
    clip = None


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kernelx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernely = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class Fusionloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis: torch.Tensor, image_ir: torch.Tensor, generate_img: torch.Tensor):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)

        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)

        total = loss_in + 10.0 * loss_grad
        return total, loss_in, loss_grad


class SimpleSSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.padding = window_size // 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, self.window_size, stride=1, padding=self.padding)
        mu_y = F.avg_pool2d(y, self.window_size, stride=1, padding=self.padding)

        sigma_x = F.avg_pool2d(x * x, self.window_size, stride=1, padding=self.padding) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, self.window_size, stride=1, padding=self.padding) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, self.window_size, stride=1, padding=self.padding) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
        )
        return 1.0 - ssim_map.mean()


class FrequencyConsistencyLoss(nn.Module):
    """RPFNet-style adaptive frequency contrastive consistency loss.

    low_weight/high_weight are kept for compatibility and mapped to the
    positive and negative contrastive terms, respectively.
    """

    def __init__(
        self,
        low_weight: float = 1.0,
        high_weight: float = 1.0,
        eps: float = 1e-8,
        detach_mask: bool = True,
    ):
        super().__init__()
        self.pos_weight = low_weight
        self.neg_weight = high_weight
        self.eps = eps
        self.detach_mask = detach_mask

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std

    def _adaptive_mask(self, ir: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        diff = ir - vis
        ir_saliency = torch.sigmoid(self._standardize(ir))
        diff_saliency = torch.sigmoid(self._standardize(diff))

        response = (
            ir_saliency * (1.0 - diff_saliency)
            + diff_saliency * (1.0 - ir_saliency)
        ) * 0.5

        mean = response.mean(dim=(-2, -1), keepdim=True)
        std = response.std(dim=(-2, -1), keepdim=True, unbiased=False)
        threshold = mean + std
        mask = (response > threshold).to(response.dtype)

        if self.detach_mask:
            mask = mask.detach()
        return mask

    @staticmethod
    def _fft_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        fft_a = torch.fft.fft2(a, norm='ortho')
        fft_b = torch.fft.fft2(b, norm='ortho')
        return torch.mean(torch.abs(fft_a - fft_b))

    def forward(self, image_vis: torch.Tensor, image_ir: torch.Tensor, fused: torch.Tensor):
        vis = image_vis[:, :1, :, :]
        ir = image_ir[:, :1, :, :]

        mask = self._adaptive_mask(ir, vis).clamp(min=0.0, max=1.0)
        inv_mask = 1.0 - mask

        f_ir_region = fused * mask
        ir_region = ir * mask

        f_vis_region = fused * inv_mask
        vis_region = vis * inv_mask

        ir_mismatch = ir * inv_mask
        vis_mismatch = vis * mask

        pos_ir = self._fft_l1(f_ir_region, ir_region)
        pos_vis = self._fft_l1(f_vis_region, vis_region)
        pos_loss = pos_ir + pos_vis

        neg_ir_cross = self._fft_l1(f_ir_region, ir_mismatch)
        neg_vis_cross = self._fft_l1(f_ir_region, vis_mismatch)
        neg_bg_ir = self._fft_l1(f_vis_region, ir_region)
        neg_bg_vis = self._fft_l1(f_vis_region, vis_region)
        neg_loss = neg_ir_cross + neg_vis_cross + neg_bg_ir + neg_bg_vis

        denom = (self.neg_weight * neg_loss).clamp_min(self.eps)
        total = self.pos_weight * pos_loss / denom
        return total, pos_loss.detach(), neg_loss.detach()


def cc(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> torch.Tensor:
    if img1.shape != img2.shape:
        raise ValueError(
            f"cc input shapes must match, "
            f"got {img1.shape} and {img2.shape}."
        )
    if img1.ndim != 4:
        raise ValueError(
            "cc expects tensors with shape (N, C, H, W)."
        )
    if not img1.is_floating_point() or not img2.is_floating_point():
        raise TypeError("cc expects floating-point tensors.")

    eps = torch.finfo(img1.dtype).eps
    n, c, _, _ = img1.shape

    img1 = img1.reshape(n, c, -1)
    img2 = img2.reshape(n, c, -1)

    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)

    numerator = torch.sum(img1 * img2, dim=-1)

    norm1 = torch.sqrt(torch.sum(img1 ** 2, dim=-1) + eps)
    norm2 = torch.sqrt(torch.sum(img2 ** 2, dim=-1) + eps)
    denominator = norm1 * norm2

    corr = numerator / denominator
    return torch.clamp(corr, -1.0, 1.0).mean()


class IntentAlignmentLoss(nn.Module):
    """Contrastively align free intent vectors to fixed prompt anchors."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def _loss_one(self, z: torch.Tensor, prompt_bank: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        prompt_bank = F.normalize(prompt_bank.detach().to(z.device, z.dtype), dim=-1)
        logits = z.matmul(prompt_bank.t()) / max(float(self.temperature), 1e-6)
        target = torch.argmax(logits.detach(), dim=1)
        return F.cross_entropy(logits, target)

    def forward(self, z_deg: torch.Tensor, z_fus: torch.Tensor,
                degradation_prompt_bank: torch.Tensor, fusion_prompt_bank: torch.Tensor):
        deg_loss = self._loss_one(z_deg, degradation_prompt_bank)
        fus_loss = self._loss_one(z_fus, fusion_prompt_bank)
        total = deg_loss + fus_loss
        return total, {'align_deg': deg_loss.detach(), 'align_fus': fus_loss.detach()}


class CLIPSemanticConsistencyLoss(nn.Module):
    """Frozen CLIP image-feature consistency for VIS/IR/Fused grayscale tensors.

    CLIPSemanticConsistencyLoss is used only as a training-time semantic loss.
    It is not part of the inference forward path and does not generate text intents.
    """

    def __init__(self, clip_model_name: str = 'ViT-B/32', download_root: str = None,
                 image_size: int = 224):
        super().__init__()
        if clip is None:
            raise ImportError('openai-clip is required for CLIPSemanticConsistencyLoss.')
        clip_model, _ = clip.load(clip_model_name, device='cpu', download_root=download_root)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model
        self.image_size = image_size
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :1].clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        image = self._prep(x)
        dtype = next(self.clip_model.parameters()).dtype
        feat = self.clip_model.encode_image(image.to(dtype=dtype))
        return F.normalize(feat.float(), dim=-1)

    def forward(self, image_vis: torch.Tensor, image_ir: torch.Tensor, fused: torch.Tensor):
        fused_feat = self._encode(fused)
        with torch.no_grad():
            vis_feat = self._encode(image_vis)
            ir_feat = self._encode(image_ir)
        loss_vis = 1.0 - F.cosine_similarity(fused_feat, vis_feat, dim=-1).mean()
        loss_ir = 1.0 - F.cosine_similarity(fused_feat, ir_feat, dim=-1).mean()
        total = 0.5 * (loss_vis + loss_ir)
        return total, {'sem_vis': loss_vis.detach(), 'sem_ir': loss_ir.detach()}


class CorrelationConsistencyLoss(nn.Module):
    """Preserve correlation between fused and source images."""

    def forward(
        self,
        image_vis: torch.Tensor,
        image_ir: torch.Tensor,
        fused: torch.Tensor,
    ) -> torch.Tensor:
        vis = image_vis[:, :1]
        ir = image_ir[:, :1]
        fused_y = fused[:, :1]

        loss_vis = 1.0 - cc(fused_y, vis)
        loss_ir = 1.0 - cc(fused_y, ir)

        loss = loss_vis + loss_ir

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "CorrelationConsistencyLoss produced NaN or Inf."
            )

        return loss


class LocalContrastLoss(nn.Module):
    """Match fused local contrast to a stable source-derived target."""

    def __init__(
        self,
        window_size: int = 7,
        eps: float = 1e-6,
    ):
        super().__init__()

        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError(
                "window_size must be a positive odd integer."
            )

        self.window_size = window_size
        self.padding = window_size // 2
        self.eps = eps

    def _local_std(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(
                "LocalContrastLoss expects input with shape (N, C, H, W)."
            )
        height, width = image.shape[-2:]
        if height <= self.padding or width <= self.padding:
            raise ValueError(
                f"Input spatial size {(height, width)} must be larger than "
                f"reflect padding {self.padding}."
            )

        # Use reflect padding to avoid edge artifacts from zero-padding
        image_padded = F.pad(
            image,
            [self.padding] * 4,
            mode="reflect",
        )
        image_sq_padded = image_padded * image_padded

        local_mean = F.avg_pool2d(
            image_padded,
            kernel_size=self.window_size,
            stride=1,
            padding=0,
        )

        local_square_mean = F.avg_pool2d(
            image_sq_padded,
            kernel_size=self.window_size,
            stride=1,
            padding=0,
        )

        local_variance = (
            local_square_mean
            - local_mean * local_mean
        ).clamp_min(0.0)

        return torch.sqrt(
            local_variance + self.eps
        )

    def forward(
        self,
        image_vis: torch.Tensor,
        image_ir: torch.Tensor,
        fused: torch.Tensor,
    ) -> torch.Tensor:
        vis = image_vis[:, :1]
        ir = image_ir[:, :1]
        fused_y = fused[:, :1]

        std_vis = self._local_std(vis)
        std_ir = self._local_std(ir)
        std_fused = self._local_std(fused_y)

        std_max = torch.maximum(std_vis, std_ir)
        std_mean = 0.5 * (std_vis + std_ir)

        target_std = (
            0.7 * std_max + 0.3 * std_mean
        ).detach()

        loss = F.l1_loss(std_fused, target_std)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "LocalContrastLoss produced NaN or Inf."
            )

        return loss
