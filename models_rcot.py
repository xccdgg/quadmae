"""RCOT two-stage image restoration modules built on DMAE with a Restormer-style
residual refinement branch."""

from __future__ import annotations

import math
from functools import partial
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL

from util.quadatasetgpu import QuaternionWaveletNoise
from util.smooth import _sigma_total_from_pixel

from models_dmae import DenoisingMaskedAutoencoderViT


# -----------------------------------------------------------------------------#
# Helper blocks copied / simplified from RCOT-main
# -----------------------------------------------------------------------------#

def conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class CALayer(nn.Module):
    """Channel attention layer as used in RCAN / RCOT."""

    def __init__(self, channel: int, reduction: int = 16, bias: bool = True) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    """Channel attention block."""

    def __init__(self, n_feat: int, kernel_size: int = 3, reduction: int = 16, bias: bool = True) -> None:
        super().__init__()
        self.body = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, bias=bias),
            nn.PReLU(),
            conv(n_feat, n_feat, kernel_size, bias=bias),
        )
        self.ca = CALayer(n_feat, reduction, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res = self.ca(res)
        return res + x


class SAM(nn.Module):
    """Spatial attention module used to generate the stage image."""

    def __init__(self, n_feat: int, kernel_size: int = 3, bias: bool = True) -> None:
        super().__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x: torch.Tensor, x_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        attn = torch.sigmoid(self.conv3(img))
        x1 = x1 * attn
        x1 = x1 + x
        return x1, img


class UpSample(nn.Module):
    """Upsample by a factor of 2 using PixelShuffle."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# -----------------------------------------------------------------------------#
# Feature pyramids
# -----------------------------------------------------------------------------#

class TokenPyramid(nn.Module):
    """Project ViT patch tokens into a 3-scale convolutional pyramid."""

    def __init__(self, in_dim: int, base_dim: int = 64) -> None:
        super().__init__()
        self.base_dim = base_dim

        self.level1 = nn.Sequential(
            nn.Conv2d(in_dim, base_dim * 16, kernel_size=1),
            nn.PixelShuffle(4),
            CAB(base_dim),
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(in_dim, base_dim * 4, kernel_size=1),
            nn.PixelShuffle(2),
            CAB(base_dim),
        )
        self.level3 = nn.Sequential(
            nn.Conv2d(in_dim, base_dim, kernel_size=1),
            CAB(base_dim),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """tokens: (B, N, C) without CLS token."""
        b, n, c = tokens.shape
        side = int(math.sqrt(n))
        feat = tokens.transpose(1, 2).reshape(b, c, side, side)
        level1 = self.level1(feat)
        level2 = self.level2(feat)
        level3 = self.level3(feat)
        return level1, level2, level3


class ResidualPyramidEncoder(nn.Module):
    """Encode residual images into a multi-scale pyramid aligned with the token pyramid."""

    def __init__(self, in_channels: int = 3, base_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
        )
        self.level1 = nn.Sequential(CAB(base_dim), CAB(base_dim))
        self.down1 = nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1)
        self.level2 = nn.Sequential(CAB(base_dim), CAB(base_dim))
        self.down2 = nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1)
        self.level3 = nn.Sequential(CAB(base_dim), CAB(base_dim))

    def forward(self, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(residual)
        level1 = self.level1(x)            # 56×56
        x = self.down1(level1)
        level2 = self.level2(x)            # 28×28
        x = self.down2(level2)
        level3 = self.level3(x)            # 14×14
        return level1, level2, level3


class RCOTDecoder(nn.Module):
    """Restormer-style decoder with SAM refinement."""

    def __init__(self, base_dim: int = 64) -> None:
        super().__init__()
        self.latent = nn.Sequential(CAB(base_dim), CAB(base_dim))
        self.up_level3 = UpSample(base_dim)
        self.reduce_level2 = nn.Sequential(
            conv(base_dim * 2, base_dim, kernel_size=3, bias=True),
            CAB(base_dim),
        )
        self.up_level2 = UpSample(base_dim)
        self.reduce_level1 = nn.Sequential(
            conv(base_dim * 2, base_dim, kernel_size=3, bias=True),
            CAB(base_dim),
            CAB(base_dim),
        )
        self.refine = nn.Sequential(CAB(base_dim), CAB(base_dim))
        self.sam = SAM(base_dim, kernel_size=1, bias=True)

    def forward(
        self,
        fused_levels: Sequence[torch.Tensor],
        base_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        level1, level2, level3 = fused_levels
        x = self.latent(level3)
        x = self.up_level3(x)
        x = torch.cat([x, level2], dim=1)
        x = self.reduce_level2(x)
        x = self.up_level2(x)
        x = torch.cat([x, level1], dim=1)
        x = self.reduce_level1(x)
        x = self.refine(x)
        if x.shape[-2:] != base_image.shape[-2:]:
            base_resized = F.interpolate(base_image, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            base_resized = base_image

        feat, refined = self.sam(x, base_resized)

        if refined.shape[-2:] != base_image.shape[-2:]:
            refined = F.interpolate(refined, size=base_image.shape[-2:], mode="bilinear", align_corners=False)

        return refined, feat


class ClassificationHead(nn.Module):
    """Fuse ViT CLS token with multi-scale features for classification."""

    def __init__(
        self,
        cls_dim: int,
        feat_dim: int,
        num_levels: int,
        num_classes: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cls_dim + feat_dim * num_levels)
        self.mlp = nn.Sequential(
            nn.Linear(cls_dim + feat_dim * num_levels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cls_token: torch.Tensor, fused_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        pooled_feats = [feat.mean(dim=[2, 3]) for feat in fused_levels]
        concat = torch.cat([cls_token] + pooled_feats, dim=1)
        logits = self.mlp(self.norm(concat))
        return logits


# -----------------------------------------------------------------------------#
# Two-stage DMAE with RCOT-style refinement
# -----------------------------------------------------------------------------#

class TwoStageDMAE(nn.Module):
    """DMAE backbone + RCOT residual refinement + optional classifier."""

    def __init__(
        self,
        base_model: DenoisingMaskedAutoencoderViT,
        *,
        freeze_base: bool = True,
        use_quaternion_noise: bool = False,
        levels: int = 1,
        ratio: float = 3.0,
        num_classes: int = 10,
        use_head: bool = False,
        base_dim: int = 64,
        head_hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.use_quaternion_noise = bool(use_quaternion_noise)
        self.levels = int(levels)
        self.ratio = float(ratio)
        self.use_head = use_head
        self.base_dim = base_dim

        embed_dim = base_model.patch_embed.proj.out_channels
        self.token_pyramid = TokenPyramid(embed_dim, base_dim)
        self.residual_encoder = ResidualPyramidEncoder(in_channels=3, base_dim=base_dim)
        self.fusion_alpha = nn.Parameter(torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32))
        self.decoder = RCOTDecoder(base_dim)

        if use_head:
            self.classifier = ClassificationHead(
                cls_dim=embed_dim,
                feat_dim=base_dim,
                num_levels=3,
                num_classes=num_classes,
                hidden_dim=head_hidden_dim,
            )
            if isinstance(self.classifier.mlp, nn.Sequential):
                for idx in range(len(self.classifier.mlp) - 1, -1, -1):
                    if isinstance(self.classifier.mlp[idx], nn.Linear):
                        self._modules['head'] = self.classifier.mlp[idx]
                        break
        else:
            self.classifier = None

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
            for m in self.base.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
                    m.eval()
        else:
            for p in self.base.parameters():
                p.requires_grad_(True)

        print(
            f"[Model] freeze_base={freeze_base}, "
            f"trainable_base_params={sum(p.requires_grad for p in self.base.parameters())}"
        )

    # ------------------------------------------------------------------#
    def _decoder_tokens(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Return decoder token features of the first stage before prediction."""
        x = self.base.decoder_embed(latent)
        mask_tokens = self.base.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.base.decoder_pos_embed
        for blk in self.base.decoder_blocks:
            x = blk(x)
        x = self.base.decoder_norm(x)
        return x

    # ------------------------------------------------------------------#
    def _apply_noise(self, imgs_norm: torch.Tensor) -> torch.Tensor:
        if not self.use_quaternion_noise:
            noise_norm = torch.randn_like(imgs_norm) * (self.base.sigma / self.base.std)
            return imgs_norm + noise_norm
        sigma_pix_norm = (self.base.sigma / self.base.std.mean()).item()
        sigma_total = _sigma_total_from_pixel(sigma_pix_norm, self.ratio)
        return QuaternionWaveletNoise.apply_noise(
            imgs_norm,
            sigma=sigma_total,
            filter_name="haar",
            levels=self.levels,
            ratio=self.ratio,
            device=imgs_norm.device,
        )

    # ------------------------------------------------------------------#
    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return normalized stage-1/2 outputs, residual, losses, and optional logits."""

        target_size = self.base.patch_embed.img_size
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        elif isinstance(target_size, tuple):
            if len(target_size) == 1:
                target_size = (target_size[0], target_size[0])
        else:
            raise TypeError(f"Unsupported img_size type: {type(target_size)}")

        imgs = transforms.Resize(
            target_size,
            interpolation=PIL.Image.BICUBIC,
        )(imgs)

        if self.base.mean.device != imgs.device:
            self.base.mean = self.base.mean.to(imgs.device)
            self.base.std = self.base.std.to(imgs.device)

        imgs_norm = (imgs - self.base.mean) / self.base.std
        imgs_noised_norm = self._apply_noise(imgs_norm)

        latent, mask, ids_restore = self.base.forward_encoder(imgs_noised_norm, mask_ratio)
        cls_token = latent[:, 0, :]
        patch_tokens = latent[:, 1:, :]

        decoder_tokens = self._decoder_tokens(latent, ids_restore)
        pred_tokens = self.base.decoder_pred(decoder_tokens)[:, 1:, :]
        loss1 = self.base.forward_loss(imgs_norm, pred_tokens, mask)
        x_stage1_norm = self.base.unpatchify(pred_tokens)
        residual_norm = imgs_noised_norm - x_stage1_norm

        trunk_feats = self.token_pyramid(patch_tokens)
        res_feats = self.residual_encoder(residual_norm)

        fused_levels: List[torch.Tensor] = []
        for idx, (t_feat, r_feat) in enumerate(zip(trunk_feats, res_feats)):
            if r_feat.shape[-2:] != t_feat.shape[-2:]:
                r_feat = F.interpolate(r_feat, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
            alpha = torch.clamp(self.fusion_alpha[idx], 0.0, 1.5)
            fused_levels.append(t_feat + alpha * r_feat)

        x_stage2_norm, _ = self.decoder(fused_levels, x_stage1_norm)
        loss2 = self.base.forward_loss(imgs_norm, self.base.patchify(x_stage2_norm), mask)

        cls_logits: Optional[torch.Tensor] = None
        if self.classifier is not None:
            cls_logits = self.classifier(cls_token, fused_levels)

        return x_stage1_norm, residual_norm, x_stage2_norm, loss1, loss2, cls_logits

    # ------------------------------------------------------------------#
    @torch.no_grad()
    def restore(
        self,
        x_noisy: torch.Tensor,
        *,
        use_rcot: bool = True,
    ) -> torch.Tensor:
        """Restore noisy input.  Returns pixel-domain images."""

        target_size = self.base.patch_embed.img_size
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        elif isinstance(target_size, tuple):
            if len(target_size) == 1:
                target_size = (target_size[0], target_size[0])
        else:
            raise TypeError(f"Unsupported img_size type: {type(target_size)}")

        if x_noisy.shape[-2] != target_size[0] or x_noisy.shape[-1] != target_size[1]:
            x_noisy = transforms.Resize(
                target_size,
                interpolation=PIL.Image.BICUBIC,
            )(x_noisy)

        if self.base.mean.device != x_noisy.device:
            self.base.mean = self.base.mean.to(x_noisy.device)
            self.base.std = self.base.std.to(x_noisy.device)

        x_norm = (x_noisy - self.base.mean) / self.base.std
        latent, _, ids_restore = self.base.forward_encoder(x_norm, mask_ratio=0.0)
        decoder_tokens = self._decoder_tokens(latent, ids_restore)
        pred_tokens = self.base.decoder_pred(decoder_tokens)[:, 1:, :]
        x_stage1_norm = self.base.unpatchify(pred_tokens)

        if not use_rcot:
            return (x_stage1_norm * self.base.std + self.base.mean).clamp(0.0, 1.0)

        patch_tokens = latent[:, 1:, :]
        residual_norm = x_norm - x_stage1_norm
        trunk_feats = self.token_pyramid(patch_tokens)
        res_feats = self.residual_encoder(residual_norm)
        fused_levels = []
        for alpha, t_feat, r_feat in zip(self.fusion_alpha, trunk_feats, res_feats):
            if r_feat.shape[-2:] != t_feat.shape[-2:]:
                r_feat = F.interpolate(r_feat, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
            fused_levels.append(t_feat + torch.clamp(alpha, 0.0, 1.5) * r_feat)
        x_stage2_norm, _ = self.decoder(fused_levels, x_stage1_norm)
        return (x_stage2_norm * self.base.std + self.base.mean).clamp(0.0, 1.0)

    # ------------------------------------------------------------------#
    def no_weight_decay(self) -> List[str]:
        tags = []
        if hasattr(self.base, "no_weight_decay"):
            tags.extend(self.base.no_weight_decay())
        return tags

    # Compatibility shim ------------------------------------------------#
    @property
    def blocks(self):
        return self.base.blocks

    @property
    def patch_embed(self):
        return self.base.patch_embed

    @property
    def norm(self):
        return self.base.norm

    @property
    def pos_embed(self):
        return self.base.pos_embed


# -----------------------------------------------------------------------------#
# Factory
# -----------------------------------------------------------------------------#

def rcot_dmae_vit_base_patch16(
    *,
    freeze_base: bool = True,
    dmae_ckpt: Optional[str] = None,
    use_quaternion_noise: bool = False,
    levels: int = 1,
    ratio: float = 3.0,
    num_classes: int = 10,
    use_head: bool = False,
    head_hidden_dim: int = 1024,
    base_dim: int = 64,
    **kwargs,
) -> TwoStageDMAE:
    kwargs.pop("num_classes", None)
    kwargs.pop("use_head", None)
    base = DenoisingMaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    if dmae_ckpt:
        ckpt = torch.load(dmae_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        clean_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[7:]
            clean_state[k] = v.float() if isinstance(v, torch.Tensor) else v
        missing, unexpected = base.load_state_dict(clean_state, strict=False)
        print(f"Loaded DMAE weights from {dmae_ckpt} (missing {len(missing)}, unexpected {len(unexpected)})")

    model = TwoStageDMAE(
        base_model=base,
        freeze_base=freeze_base,
        use_quaternion_noise=use_quaternion_noise,
        levels=levels,
        ratio=ratio,
        num_classes=num_classes,
        use_head=use_head,
        base_dim=base_dim,
        head_hidden_dim=head_hidden_dim,
    )
    return model
