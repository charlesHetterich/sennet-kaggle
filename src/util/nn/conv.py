import torch
import torch.nn as nn

import numpy as np
from typing import Optional

from .bayes import Bayes

class Conv3DNormed(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            'norm_term',
            torch.tensor(np.sqrt(
                np.max([self.in_channels, self.out_channels])
            ))
        )
    
    def forward(self, x: torch.Tensor):
        return super().forward(x) / self.norm_term

class Conv2DNormed(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            'norm_term',
            torch.tensor(np.sqrt(
                np.max([self.in_channels, self.out_channels])
            ))
        )
    
    def forward(self, x: torch.Tensor):
        return super().forward(x) / self.norm_term

class ConvBlock(Bayes):
    def __init__(
            self,
            in_f: int,
            out_f: int,
            kernel_size: int = 3,

            conv: nn.Module = nn.Conv2d,
            activation: nn.Module = nn.GELU,
            norm_fn: Optional[nn.Module] = None,
            dropout: Optional[tuple[nn.Module, float]] = None,

            padding: int = 1,
            stride: int = 1,
            block_depth: int = 4,
            **kwargs
        ):
        super().__init__()
        self.out_f = out_f
        def build_block(in_f, out_f, stride=1):
            L = []
            if dropout is not None:
                L.append(dropout[0](dropout[1]))
            L.append(conv(in_f, out_f, kernel_size, padding=padding, stride=stride, **kwargs))
            if norm_fn is not None:
                L.append(norm_fn(out_f))
            L.append(activation())
            return nn.Sequential(*L)

        self.layers = nn.ModuleList([
            build_block(
                in_f if i == 0 else out_f,
                out_f,
                stride=stride if i == 0 else 1
                ) for i in range(block_depth)
        ])
        for l in self.layers: self.register_bayes(l)

    def forward(self, x: torch.Tensor):
        zz = self.layers[0](x)
        z = zz
        for l in self.layers[1:]:
            z = l(z) + z # Residual connection
        return z + zz # Jump connection