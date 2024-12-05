import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from typing import Optional, Union

class SweepCube(Dataset):
    def __init__(
            self, 
            data: torch.Tensor, 
            patch_size: tuple[int, int, int],
            stride: Optional[tuple[int, int, int]] = None
        ):
        assert data.ndim == 4, "Data must be 4-dimensional"

        self.data = data
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size

        self.patches = (
            (self.data.shape[1] - self.patch_size[0]) // self.stride[0] + 1,
            (self.data.shape[2] - self.patch_size[1]) // self.stride[1] + 1,
            (self.data.shape[3] - self.patch_size[2]) // self.stride[2] + 1,
        )

    def __len__(self):
        return (self.patches[0] * self.patches[1] * self.patches[2])

    def __getitem__(self, i):
        x, y, z = np.unravel_index(i, self.patches)
        x, y, z = x * self.stride[0], y * self.stride[1], z * self.stride[2]

        return self.data[
            :,
            x:x+self.patch_size[0],
            y:y+self.patch_size[1],
            z:z+self.patch_size[2],
        ], torch.tensor([x, y, z])

class Bayes(nn.Module):
    def __init__(self):
        super().__init__()
        self._registered_bayesian_modules: list["Bayes"] = []

    def register_bayes(self, module: Union[list[nn.Module], nn.Module]):
        if 'Bayes' in [b.__name__ for b in type(module).__bases__]:
            self._registered_bayesian_modules.append(module)
        else:
            # check if iterable
            try:
                for u in module: self.register_bayes(u)
            except TypeError:
                pass

    def penalty(self):
        raise NotImplementedError()

    def penalize(self, alpha: float = 1e-3):
        """
        alpha: the penalty coefficient
        """
        return sum(u.penalize(alpha) for u in self._registered_bayesian_modules)

    def decay_var(self, gamma: float = 0.5):
        """
        gamma: variance decay rate
        """
        for u in self._registered_bayesian_modules: u.decay_var(gamma)
    
    def rebase(self):
        """
        Rebase the parameters to the current mean
        """
        for u in self._registered_bayesian_modules: u.rebase()

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

class UNetCrossBlock(Bayes):
    def __init__(
            self,
            layers: list[int],
            into_stage: int,

            block_depth: int = 1,
            connect_depth: int = 64,

            conv: nn.Module = nn.Conv3d,
            activation: nn.Module = nn.GELU,
            pool_fn: nn.Module = nn.MaxPool3d,
            resize_kernel: tuple = (2, 2, 2),
            upsample_mode: str = 'trilinear',
            norm_fn: Optional[nn.Module] = None,
            dropout: Optional[tuple[nn.Module, float]] = None,

            **kwargs
    ):
        super().__init__()
        assert 0 <= into_stage < len(layers), f"into_layer must be in [0, {len(layers)})"
        self.num_stages = len(layers)

        self.resize_blocks = nn.ModuleList([])
        for i in range(len(layers)):
            block = nn.Sequential()                
            if i < into_stage:
                block.append(pool_fn((*(k ** (into_stage - i) for k in resize_kernel),), ceil_mode=True))
            if i > into_stage:
                block.append(nn.Upsample(scale_factor=(*(k ** (i - into_stage) for k in resize_kernel),), mode=upsample_mode))
            block.append(
                ConvBlock(
                    connect_depth * self.num_stages if i > into_stage and i != self.num_stages - 1 else layers[i],
                    connect_depth,
                    conv=conv,
                    activation=activation,
                    block_depth=block_depth,
                    dropout=dropout,
                    norm_fn=norm_fn,
                    **kwargs
                ),
            )
            self.resize_blocks.append(block)
        self.full_conv_block = ConvBlock(
            connect_depth * len(layers),
            connect_depth * len(layers),
            conv=conv,
            activation=activation,
            block_depth=block_depth,
            dropout=dropout,
            norm_fn=norm_fn,
            **kwargs
        )

        for l in self.resize_blocks: self.register_bayes(l)
        self.register_bayes(self.full_conv_block)


    def forward(self, xs: list[torch.Tensor]):
        assert len(xs) == self.num_stages, f"expected input to be of length {self.num_stages}, but got {len(xs)}"
        zs = []
        for i in range(len(xs)):
            zs.append(self.resize_blocks[i](xs[i]))
        return self.full_conv_block(torch.cat(zs, dim=1))

# Architecture based on : https://arxiv.org/abs/2004.08790
class UNet3P(Bayes):
    def __init__(
            self,
            in_f: int = 1,
            layers: list[int] = [64, 128, 256, 512, 1024],
            out_f: int = 1,

            block_depth: int = 4,
            connect_depth: int = 64,

            conv: nn.Module = nn.Conv3d,
            activation: nn.Module = nn.GELU,
            pool_fn: nn.Module = nn.MaxPool3d,
            resize_kernel: tuple = (2, 2, 2),
            upsample_mode: str = 'trilinear',
            norm_fn: nn.Module = None,
            dropout: tuple[nn.Module, float] = None,
            
            input_noise: Optional[float] = 0.1,
            **kwargs
        ):
        super().__init__()
        if input_noise is not None:
            self.register_buffer('input_noise', torch.tensor(input_noise))
        c = in_f

        self.input_norm = norm_fn(in_f)
        self.down_blocks = nn.ModuleList([])
        for i, l in enumerate(layers):
            block = nn.Sequential()
            if i != 0:
                block.append(pool_fn(resize_kernel))
            block.append(
                ConvBlock(
                    c,
                    l,
                    conv=conv,
                    activation=activation,
                    block_depth=block_depth,
                    dropout=dropout,
                    norm_fn=norm_fn,
                    **kwargs
                )
            )
            self.down_blocks.append(block)
            c = l
        
        self.cross_blocks = nn.ModuleList([])
        for i in range(len(layers) - 1):
            self.cross_blocks.append(
                UNetCrossBlock(
                    layers,
                    i,
                    block_depth=block_depth,
                    connect_depth=connect_depth,
                    conv=conv,
                    activation=activation,
                    pool_fn=pool_fn,
                    resize_kernel=resize_kernel,
                    upsample_mode=upsample_mode,
                    norm_fn=norm_fn,
                    dropout=dropout,
                    **kwargs
                )
            )
    
        self.out_blocks = nn.ModuleList([])
        for i in reversed(range(len(layers) - 1)):
            L = []
            if i == 0:
                L.append(
                    ConvBlock(
                        connect_depth * len(layers),
                        connect_depth * len(layers),
                        1,
                        conv=conv,
                        padding=0,
                        activation=activation,
                        block_depth=block_depth,
                        dropout=dropout,
                        norm_fn=norm_fn,
                        **kwargs
                    )
                )
            L.append(
                conv(
                    connect_depth * len(layers),
                    out_f,
                    1,
                    **kwargs
                )
            )
            self.out_blocks.append(nn.Sequential(*L))

        self.mask_blocks = nn.ModuleList([
            pool_fn((*(k ** (i + 1) for k in resize_kernel),))
            for i in reversed(range(len(layers) - 2))
        ])

        for l in self.down_blocks: self.register_bayes(l)
        for l in self.cross_blocks: self.register_bayes(l)
        for l in self.out_blocks: self.register_bayes(l)

    def _prep_input(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_norm(x)
        if hasattr(self, 'input_noise') and self.training:
            z += torch.randn_like(z) * self.input_noise
        return z

    def _down_fwd(self, x: torch.Tensor) -> list[torch.Tensor]:
        down_agg = []
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)
            down_agg.append(x)
        return down_agg

    def forward(self, x: torch.Tensor, up_depth: Optional[int] = None) -> list[torch.Tensor]:
        up_depth = len(self.cross_blocks) if up_depth is None else up_depth
        assert up_depth <= len(self.cross_blocks), f"up_depth must be in [0, {len(self.cross_blocks)}]"

        z = self._prep_input(x)
        down_agg = self._down_fwd(z)
        
        up_agg = []
        for i in list(reversed(range(len(self.cross_blocks))))[:up_depth]:
            cross_input = down_agg[:i + 1] + list(reversed([u for u in up_agg])) + [down_agg[-1]]
            up_agg.append(self.cross_blocks[i](cross_input))

        return [
            o(x)
            # for o, x in zip(self.out_blocks, [down_agg[-1]] + up_agg)
            for o, x in zip(self.out_blocks[:up_depth], up_agg)
        ]

    def deep_masks(self, y: torch.Tensor) -> list[torch.Tensor]:
        masks = []
        for i in range(len(self.mask_blocks)):
            masks.append(self.mask_blocks[i](y))
        return masks + [y]

class Patch2p5D(nn.Module):
    def __init__(self, model: UNet3P):
        super().__init__()
        self.model = model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.view(x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4])
        return self.model(z)[-1]
