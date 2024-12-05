import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from typing import List, Type, Optional

from util import Bayes, SweepCube

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

# class ScanInferece(nn.Module):
#     def __init__(self, coarse_patcher: UNet3D, fine_patcher: UNet3D, coarse_patch_size: int = 64, fine_patch_size: int = 16, stride: Optional[int] = None, batch_size: int = 32):
#         super().__init__()
#         self.coarse_patcher = coarse_patcher
#         self.fine_patcher = fine_patcher
#         self.coarse_patch_size = coarse_patch_size
#         self.fine_patch_size = fine_patch_size
#         self.stride = stride if stride is not None else coarse_patch_size
#         self.batch_size = batch_size

#     def _to_chunks(self, x: torch.Tensor) -> torch.Tensor:
#         return x.unfold(2, self.fine_patch_size, self.fine_patch_size) \
#             .unfold(3, self.fine_patch_size, self.fine_patch_size) \
#             .unfold(4, self.fine_patch_size, self.fine_patch_size)

#     def _post_process(self, x: torch.Tensor) -> torch.Tensor:
#         cutoff = 0.3
#         k_size = 17
#         k = torch.ones(1, 1, k_size, k_size, k_size, device=x.device)
#         density = F.conv3d(x.float(), k, padding=k_size // 2) / (k_size ** 3)
#         return (density < cutoff) & x

#     def forward(self, x: torch.Tensor, working_device: str = "cuda") -> torch.Tensor:
#         agg_pred = torch.zeros_like(x)
#         scan_loader = DataLoader(
#             SweepCube(x, self.coarse_patch_size, self.stride),
#             batch_size=self.batch_size,
#             shuffle=True
#         )
    
#         for xs, positions in tqdm(scan_loader):
#             xs = xs.to(working_device).float()
#             p_y = [F.sigmoid(logits) for logits in self.coarse_patcher(xs, up_depth=2)]
#             pred_masks = [(p > 0.5) for p in p_y]

#             fine_input = self._to_chunks(xs)[pred_masks[-1]].unsqueeze(1)
#             p_y_fine = [F.sigmoid(logits) for logits in self.fine_patcher(fine_input)]
#             pred_masks_fine = [(p > 0.5) for p in p_y_fine]

#             full_pred = torch.zeros_like(xs, dtype=torch.bool)
#             self._to_chunks(full_pred)[pred_masks[-1]] = pred_masks_fine[-1].squeeze(1)
#             # full_pred = self._post_process(full_pred)

#             for p, pos in zip(full_pred.cpu(), positions):
#                 agg_pred[
#                     :,
#                     pos[0]:pos[0] + self.coarse_patch_size,
#                     pos[1]:pos[1] + self.coarse_patch_size,
#                     pos[2]:pos[2] + self.coarse_patch_size,
#                 ] = p
        
#         return agg_pred