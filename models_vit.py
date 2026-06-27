from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

from util.qwt_prior import DeterministicQWTPrior


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(
        self,
        global_pool=False,
        use_qwt_prior_adapter=False,
        qwt_prior_levels=1,
        **kwargs,
    ):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.use_qwt_prior_adapter = bool(use_qwt_prior_adapter)
        self.qwt_prior_levels = int(qwt_prior_levels)
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        if self.use_qwt_prior_adapter:
            patch_size = self.patch_embed.patch_size[0]
            self.qwt_prior_extractor = DeterministicQWTPrior(
                patch_size=patch_size,
                levels=self.qwt_prior_levels,
            )
            self.prior_ln = nn.LayerNorm(4)
            self.prior_proj = nn.Linear(self.prior_ln.normalized_shape[0], self.embed_dim, bias=True)
            self.prior_gate = nn.Parameter(torch.zeros(1))
            nn.init.normal_(self.prior_proj.weight, std=1e-3)
            nn.init.constant_(self.prior_proj.bias, 0.0)
        else:
            self.qwt_prior_extractor = None
        
        # normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    def no_weight_decay(self):
        names = set(super().no_weight_decay())
        if self.use_qwt_prior_adapter:
            names.add("prior_gate")
        return names

    def _apply_qwt_prior(self, tokens, images):
        if not self.use_qwt_prior_adapter:
            return tokens

        prior = self.qwt_prior_extractor.extract_patch_prior(images)
        if prior.shape[1] != tokens.shape[1]:
            raise ValueError(
                f"QWT prior token count {prior.shape[1]} does not match patch tokens {tokens.shape[1]}"
            )
        prior = prior.to(device=tokens.device, dtype=tokens.dtype)
        prior = self.prior_proj(self.prior_ln(prior))
        return tokens + self.prior_gate.view(1, 1, 1) * prior

    def forward_features(self, x):
        prior_input = x
        # normalization for randomized smoothing
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        x = (x - self.mean) / self.std
    
        B = x.shape[0]
        x = self.patch_embed(x)
        x = self._apply_qwt_prior(x, prior_input)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
