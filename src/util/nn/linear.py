import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
from .bayes import Bayes

class LinearBlock(Bayes):
    def __init__(
            self,
            in_f: int,
            out_f: int,
            linear: nn.Module = nn.Linear,
            activation: nn.Module = nn.ReLU,
            block_depth: int = 4,
            dropout: float = 0,
            batch_norm: bool = False,
            **kwargs
        ):
        super().__init__()
        def build_block(in_f, out_f):
            L = []
            if dropout > 0:
                L.append(nn.Dropout(dropout))
            L.append(linear(in_f, out_f, **kwargs))
            if batch_norm:
                L.append(nn.BatchNorm1d(out_f))
            L.append(activation())
            return nn.Sequential(*L)

        self.layers = nn.ModuleList([
            build_block(in_f if i == 0 else out_f, out_f) for i in range(block_depth)
        ])
        for l in self.layers: self.register_bayes(l)

    def forward(self, x: torch.Tensor):
        zz = self.layers[0](x)
        z = zz
        for l in self.layers[1:]:
            z = l(z) + z # Residual
        return z + zz # Jump

class MLP(Bayes):
    def __init__(
            self,
            layers: list[int],
            linear: nn.Module = nn.Linear,
            activation: nn.Module = nn.ReLU,
            block_depth: int = 4,
            dropout: float = 0,
            batch_norm: bool = False,
            **kwargs
        ):
        super().__init__()
        assert len(layers) >= 2, "Must be at least 2 Linear layers"

        L = []
        c = layers[0]
        for l in layers[1:-1]:
            L.append(LinearBlock(
                c, l, 
                linear=linear, 
                activation=activation, 
                block_depth=block_depth, 
                dropout=dropout, 
                batch_norm=batch_norm, 
                **kwargs
            ))
            c = l
        L.append(linear(c, layers[-1], **kwargs))
        self.net = nn.Sequential(*L)
        self.register_bayes(self.net)

    def forward(self, x: torch.Tensor):
        return self.net(x)

class ACDC(nn.Module):
    def __init__(
            self,
            in_f: int,
            out_f: int,
        ):
        super().__init__()
        self.out_f = out_f
        self.register_buffer('perm', torch.randperm(out_f))
        self.register_buffer(
            'norm_term',
            torch.tensor(np.sqrt(
                np.max([in_f, out_f])
            ))
        )

        self.A = nn.Parameter(torch.rand(1, in_f))
        self.D = nn.Parameter(torch.rand(1, out_f // 2 + 1))
        self.bias = nn.Parameter(torch.rand(1, out_f))

    def forward(self, x):
        z = x * self.A
        z = torch.fft.rfft(z, dim=-1, n=self.out_f)
        z = z * self.D
        z = torch.fft.ifft(z, dim=-1, n=self.out_f)
        z = z.real + self.bias
        return torch.gather(z, 1, getattr(self, 'perm').repeat(z.shape[0], 1)) / self.norm_term