"""RCOT two-stage image restoration modules built on DMAE."""

import torch
import torch.nn as nn
from functools import partial
import torchvision.transforms as transforms
import PIL

from models_dmae import DenoisingMaskedAutoencoderViT


class ResidualEncoder(nn.Module):
    """Encode residual image to an embedding vector."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embed_dim)
        self.act = nn.GELU()

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(r)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)
        e = self.act(self.fc(x))
        return e


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation block."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feat_dim)
        self.beta_fc = nn.Linear(cond_dim, feat_dim)

    def forward(self, cond: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_fc(cond)
        beta = self.beta_fc(cond)
        if features.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return features * gamma + beta


class ConditionalTransformerBlock(nn.Module):
    """Transformer block with FiLM conditioning."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, cond_dim: int | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.film = FiLMBlock(cond_dim or embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        if cond is not None:
            x = self.film(cond, x)
        return x


class ConditionalDecoder(nn.Module):
    """Decoder composed of conditional transformer blocks."""

    def __init__(self, embed_dim: int = 512, num_layers: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4.0, cond_dim: int | None = None,
                 patch_size: int = 16, image_size: int = 224):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            ConditionalTransformerBlock(embed_dim, num_heads, mlp_ratio, cond_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, patch_size * patch_size * 3)
        self.patch_size = patch_size

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = tokens + self.pos_embed[:, :tokens.size(1), :]
        for blk in self.blocks:
            x = blk(x, cond)
        x = self.norm(x)
        patch_pixels = self.out_proj(x)
        b, n, _ = patch_pixels.shape
        side = int(n ** 0.5)
        patches = patch_pixels.view(b, side, side, self.patch_size, self.patch_size, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patches.view(b, 3, side * self.patch_size, side * self.patch_size)


class TwoStageDMAE(nn.Module):
    """Two-stage DMAE with residual conditioning."""

    def __init__(self, base_model: DenoisingMaskedAutoencoderViT,
                 decoder2: ConditionalDecoder, res_encoder: ResidualEncoder,
                 freeze_base: bool = True):
        super().__init__()
        self.base = base_model
        self.decoder2 = decoder2
        self.res_encoder = res_encoder

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75,
                *, use_rcot: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pretrain forward that supports optional RCOT refinement."""

        noise = torch.randn_like(imgs) * self.base.sigma
        imgs_noised = imgs + noise
        imgs = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs)
        imgs_noised = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs_noised)

        if self.base.mean.device != imgs.device:
            self.base.mean = self.base.mean.to(imgs.device)
            self.base.std = self.base.std.to(imgs.device)

        imgs_norm = (imgs - self.base.mean) / self.base.std
        imgs_noised = (imgs_noised - self.base.mean) / self.base.std

        latent, mask, ids_restore = self.base.forward_encoder(imgs_noised, mask_ratio)
        pred_tokens = self.base.forward_decoder(latent, ids_restore)
        loss1 = self.base.forward_loss(imgs_norm, pred_tokens, mask)

        if not use_rcot:
            return loss1, pred_tokens, mask

        x_hat = self.base.unpatchify(pred_tokens)
        r = imgs_norm - x_hat
        cond = self.res_encoder(r)
        x_refined = self.decoder2(latent, cond)
        loss2 = ((x_refined - imgs_norm) ** 2).mean()
        loss = loss1 + loss2
        pred_refined = self.base.patchify(x_refined)
        return loss, pred_refined, mask

    # ------------------------------------------------------------------
    @torch.no_grad()
    def restore(self, x_noisy: torch.Tensor, *, use_rcot: bool = True,
                x_clean: torch.Tensor | None = None) -> torch.Tensor:
        """Run the two-stage restoration on noisy inputs."""

        latent, _, ids_restore = self.base.forward_encoder(x_noisy, mask_ratio=0.0)
        pred_tokens = self.base.forward_decoder(latent, ids_restore)
        x_hat = self.base.unpatchify(pred_tokens)

        if not use_rcot:
            return x_hat

        r = (x_clean - x_hat) if x_clean is not None else (x_noisy - x_hat)
        cond = self.res_encoder(r)
        x_refined = self.decoder2(latent, cond)
        return x_refined


def rcot_dmae_vit_base_patch16(*, freeze_base: bool = True, **kwargs) -> TwoStageDMAE:
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
    cond_decoder = ConditionalDecoder(
        embed_dim=base.decoder_embed_dim,
        num_layers=len(base.decoder_blocks),
        num_heads=base.decoder_blocks[0].attn.num_heads if base.decoder_blocks else 8,
        mlp_ratio=4.0,
        cond_dim=base.decoder_embed_dim,
        patch_size=base.patch_embed.patch_size[0],
        image_size=base.patch_embed.img_size,
    )
    res_enc = ResidualEncoder(in_channels=3, embed_dim=base.decoder_embed_dim)
    return TwoStageDMAE(base, cond_decoder, res_enc, freeze_base=freeze_base)
