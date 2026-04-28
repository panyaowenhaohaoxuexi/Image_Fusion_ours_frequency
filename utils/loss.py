# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, low_weight: float = 1.0, high_weight: float = 1.0):
        super().__init__()
        self.low_weight = low_weight
        self.high_weight = high_weight

    def forward(self, image_vis: torch.Tensor, image_ir: torch.Tensor, fused: torch.Tensor):
        vis = image_vis[:, :1, :, :]
        ir = image_ir[:, :1, :, :]

        fused_amp = torch.abs(torch.fft.fft2(fused, norm='ortho'))
        vis_amp = torch.abs(torch.fft.fft2(vis, norm='ortho'))
        ir_amp = torch.abs(torch.fft.fft2(ir, norm='ortho'))

        low_target = 0.5 * (vis_amp + ir_amp)
        low_loss = F.l1_loss(torch.log1p(fused_amp), torch.log1p(low_target))

        fused_hp = fused - F.avg_pool2d(fused, 3, 1, 1)
        vis_hp = vis - F.avg_pool2d(vis, 3, 1, 1)
        ir_hp = ir - F.avg_pool2d(ir, 3, 1, 1)

        high_target = torch.max(torch.abs(vis_hp), torch.abs(ir_hp))
        high_loss = F.l1_loss(torch.abs(fused_hp), high_target)

        total = self.low_weight * low_loss + self.high_weight * high_loss
        return total, low_loss, high_loss


def cc(img1: torch.Tensor, img2: torch.Tensor):
    eps = torch.finfo(torch.float32).eps
    n, c, _, _ = img1.shape

    img1 = img1.reshape(n, c, -1)
    img2 = img2.reshape(n, c, -1)

    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)

    corr = torch.sum(img1 * img2, dim=-1) / (
        eps
        + torch.sqrt(torch.sum(img1 ** 2, dim=-1))
        * torch.sqrt(torch.sum(img2 ** 2, dim=-1))
    )
    return torch.clamp(corr, -1.0, 1.0).mean()

class TokenRoutingRankingLoss(nn.Module):
    """频率 token 路由评分辅助损失。

    目的：让 TokenScoreNet 学会“哪些 tokens 应排在 Top-K 前面”，而不是把 score
    当成频谱值缩放系数。该损失作用在 amp_score / phase_score 上，前向路由仍由
    hard Top-K 完成：高分 tokens 进入 strong interaction，低分 tokens 进入 weak path。
    """

    def __init__(self,
                 bce_weight: float = 1.0,
                 rank_weight: float = 0.5,
                 separation_margin: float = 0.35,
                 amp_weight: float = 1.0,
                 phase_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.rank_weight = rank_weight
        self.separation_margin = separation_margin
        self.amp_weight = amp_weight
        self.phase_weight = phase_weight

    @staticmethod
    def _normalize_score(score: torch.Tensor) -> torch.Tensor:
        mean = score.mean(dim=1, keepdim=True)
        std = score.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        return (score - mean) / std

    def _branch_loss(self, score: torch.Tensor, target: torch.Tensor, hard_mask: torch.Tensor):
        score_norm = self._normalize_score(score)
        target = target.detach().clamp(0.0, 1.0)
        hard_mask = hard_mask.detach().float()

        # soft target 约束：让排序分数贴近由 token 信息和跨模态差异得到的软重要性。
        bce = F.binary_cross_entropy_with_logits(score_norm, target)

        # selected/unselected 间隔约束：鼓励 Top-K 内平均分高于 Top-K 外平均分。
        selected_count = hard_mask.sum(dim=1).clamp_min(1.0)
        unselected_count = (1.0 - hard_mask).sum(dim=1).clamp_min(1.0)
        selected_mean = (score_norm * hard_mask).sum(dim=1) / selected_count
        unselected_mean = (score_norm * (1.0 - hard_mask)).sum(dim=1) / unselected_count
        rank = F.relu(self.separation_margin - (selected_mean - unselected_mean)).mean()

        total = self.bce_weight * bce + self.rank_weight * rank
        return total, bce.detach(), rank.detach(), selected_mean.detach().mean(), unselected_mean.detach().mean()

    def forward(self, aux: dict):
        amp_loss, amp_bce, amp_rank, amp_sel, amp_unsel = self._branch_loss(
            aux['amp_score'], aux['amp_score_target'], aux['amp_mask']
        )
        phase_loss, phase_bce, phase_rank, phase_sel, phase_unsel = self._branch_loss(
            aux['phase_score'], aux['phase_score_target'], aux['phase_mask']
        )
        total = self.amp_weight * amp_loss + self.phase_weight * phase_loss
        log = {
            'score_loss': total.detach(),
            'amp_score_bce': amp_bce,
            'amp_score_rank': amp_rank,
            'phase_score_bce': phase_bce,
            'phase_score_rank': phase_rank,
            'amp_selected_score_mean': amp_sel,
            'amp_unselected_score_mean': amp_unsel,
            'phase_selected_score_mean': phase_sel,
            'phase_unselected_score_mean': phase_unsel,
        }
        return total, log
