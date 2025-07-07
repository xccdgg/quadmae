"""RCOT two-stage image restoration modules built on DMAE."""

import torch
import torch.nn as nn
from functools import partial
import torchvision.transforms as transforms
import PIL
from typing import Optional, Tuple, Union

from util.quadatasetgpu import QuaternionWaveletNoise
from util.smooth import _sigma_total_from_pixel


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

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, cond_dim: Optional[int] = None):
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

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        if cond is not None:
            x = self.film(cond, x)
        return x


class ConditionalDecoder(nn.Module):
    """Transformer decoder with FiLM conditioning producing images."""

    def __init__(self, embed_dim: int = 512, num_layers: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4.0, cond_dim: Optional[int] = None,
                 patch_size: int = 16, image_size: Union[int, Tuple[int, int]] = 224):
        super().__init__()

        if isinstance(image_size, (tuple, list)):
            h, w = image_size
        else:
            h = w = image_size
        num_patches = (h // patch_size) * (w // patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            ConditionalTransformerBlock(embed_dim, num_heads, mlp_ratio, cond_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, patch_size * patch_size * 3)
        self.patch_size = patch_size

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Decode token features into an image conditioned on ``cond``."""

        x = tokens + self.pos_embed[:, : tokens.size(1), :]
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
                 *,
                 freeze_base: bool = True,
                 use_quaternion_noise: bool = False,
                 levels: int = 1,
                 ratio: float = 3.0):
        super().__init__()
        self.base = base_model
        self.decoder2 = decoder2
        self.res_encoder = res_encoder
        self.use_quaternion_noise = bool(use_quaternion_noise)
        self.levels = int(levels)
        self.ratio = float(ratio)

        if freeze_base:
            # Freeze parameters of the base encoder and first decoder
            for p in self.base.parameters():
                p.requires_grad_(False)

            # Put BatchNorm/Dropout layers in evaluation mode so that running stats stay frozen
            for m in self.base.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
                    m.eval()
        else:
            # Ensure all base parameters are trainable
            for p in self.base.parameters():
                p.requires_grad_(True)

        print(
            f"[Model] freeze_base={freeze_base}, "
            f"trainable_params_base={sum(p.requires_grad for p in self.base.parameters())}"
        )

        # Initialize conditional decoder positional embedding from the base decoder
        if self.decoder2.pos_embed.shape == (1, self.base.decoder_pos_embed.shape[1] - 1, self.base.decoder_pos_embed.shape[2]):
            with torch.no_grad():
                self.decoder2.pos_embed.copy_(self.base.decoder_pos_embed[:, 1:, :])

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

    # ------------------------------------------------------------------
    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return intermediate tensors and stage losses for RCOT pretraining."""

        imgs = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs)

        if self.base.mean.device != imgs.device:
            self.base.mean = self.base.mean.to(imgs.device)
            self.base.std = self.base.std.to(imgs.device)

        imgs_norm = (imgs - self.base.mean) / self.base.std

        if self.use_quaternion_noise:
            sigma_pix_norm = (self.base.sigma / self.base.std.mean()).item()
            sigma_total = _sigma_total_from_pixel(sigma_pix_norm, self.ratio)
            imgs_noised_norm = QuaternionWaveletNoise.apply_noise(
                imgs_norm,
                sigma=sigma_total,
                filter_name="haar",
                levels=self.levels,
                ratio=self.ratio,
                device=imgs_norm.device,
            )
        else:
            noise_norm = torch.randn_like(imgs_norm) * (self.base.sigma / self.base.std)
            imgs_noised_norm = imgs_norm + noise_norm

        latent, mask, ids_restore = self.base.forward_encoder(imgs_noised_norm, mask_ratio)
        features = self._decoder_tokens(latent, ids_restore)
        pred_tokens = self.base.decoder_pred(features)
        pred_tokens = pred_tokens[:, 1:, :]
        loss1 = self.base.forward_loss(imgs_norm, pred_tokens, mask)

        x_hat = self.base.unpatchify(pred_tokens)
        r = imgs_norm - x_hat
        cond = self.res_encoder(r)
        tokens2 = features[:, 1:, :]
        x_refined = self.decoder2(tokens2, cond)
        loss2 = ((x_refined - imgs_norm) ** 2).mean()
        return x_hat, r, x_refined, loss1, loss2

    # ------------------------------------------------------------------
    @torch.no_grad()
    def restore(
        self,
        x_noisy: torch.Tensor,
        *,
        use_rcot: bool = True,
        x_clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Restore noisy input.  Returns pixel-domain images."""

        if x_noisy.shape[-1] != self.base.patch_embed.img_size:
            x_noisy = transforms.Resize(
                self.base.patch_embed.img_size,
                interpolation=PIL.Image.BICUBIC,
            )(x_noisy)

        if self.base.mean.device != x_noisy.device:
            self.base.mean = self.base.mean.to(x_noisy.device)
            self.base.std = self.base.std.to(x_noisy.device)

        x_norm = (x_noisy - self.base.mean) / self.base.std

        latent, _, ids_restore = self.base.forward_encoder(x_norm, mask_ratio=0.0)
        features = self._decoder_tokens(latent, ids_restore)
        pred_tokens = self.base.decoder_pred(features)
        pred_tokens = pred_tokens[:, 1:, :]
        x_hat = self.base.unpatchify(pred_tokens)

        if not use_rcot:
            return (x_hat * self.base.std + self.base.mean).clamp(0.0, 1.0)

        if x_clean is not None:
            if x_clean.shape[-1] != self.base.patch_embed.img_size:
                x_clean = transforms.Resize(
                    self.base.patch_embed.img_size,
                    interpolation=PIL.Image.BICUBIC,
                )(x_clean)
            x_clean = (x_clean - self.base.mean) / self.base.std
            r = x_clean - x_hat
        else:
            r = x_norm - x_hat

        cond = self.res_encoder(r)
        tokens2 = features[:, 1:, :]
        x_refined = self.decoder2(tokens2, cond)
        return (x_refined * self.base.std + self.base.mean).clamp(0.0, 1.0)


def rcot_dmae_vit_base_patch16(
    *,
    freeze_base: bool = True,
    dmae_ckpt: Optional[str] = None,
    use_quaternion_noise: bool = False,
    levels: int = 1,
    ratio: float = 3.0,
    **kwargs,
) -> TwoStageDMAE:
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
        print(
            f"Loaded DMAE weights from {dmae_ckpt} (missing {len(missing)}, unexpected {len(unexpected)})"
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
    # initialize output projection and normalization from the first stage decoder
    cond_decoder.norm.load_state_dict(base.decoder_norm.state_dict())
    cond_decoder.out_proj.weight.data.copy_(base.decoder_pred.weight.data)
    cond_decoder.out_proj.bias.data.copy_(base.decoder_pred.bias.data)
    res_enc = ResidualEncoder(in_channels=3, embed_dim=base.decoder_embed_dim)
    return TwoStageDMAE(
        base,
        cond_decoder,
        res_enc,
        freeze_base=freeze_base,
        use_quaternion_noise=use_quaternion_noise,
        levels=levels,
        ratio=ratio,
    )
