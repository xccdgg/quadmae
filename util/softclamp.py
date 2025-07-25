# util/softclamp.py
import torch
import torch.nn as nn

class SoftClamp(nn.Module):
    r"""y = limit * tanh(x / limit). 近似线性区间 (-limit, limit)。"""
    def __init__(self, limit: float):
        super().__init__()
        self.limit = float(limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.limit * torch.tanh(x / self.limit)
