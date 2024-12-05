import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union, Optional, Callable
from torch.utils.checkpoint import checkpoint

class Lambda(nn.Module):
    def __init__(self, func : Callable[[torch.Tensor], torch.Tensor]):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ACDC(nn.Module):
    """
    NOTE : for now prob keep in_f == out_f
    """
    def __init__(
            self,
            in_f,
            out_f
        ):
        super().__init__()
        self.out_f = out_f
        self.register_buffer('perm', torch.randperm(out_f))
        self.register_buffer('norm_term', torch.tensor(np.sqrt(out_f)))

        self.A = nn.Parameter(torch.rand(in_f))
        self.D = nn.Parameter(torch.rand(out_f))
        self.bias = nn.Parameter(torch.rand(out_f))

        # self.net = nn.Sequential(
        #     Lambda(lambda x: x.mul_(self.A.view(
        #         -1, self.A.shape[0], *([1] * (x.dim() - 2))
        #     ))),
        #     Lambda(lambda x: torch.fft.fft(x, dim=1, n=self.out_f)),
        #     Lambda(lambda x: x.mul_(self.D.view(
        #         -1, self.D.shape[0], *([1] * (x.dim() - 2))
        #     ))),
        #     Lambda(lambda x: torch.fft.ifft(x, dim=1, n=self.out_f)),
        #     # Lambda(lambda x: x + self.bias.view(
        #     #     -1, self.bias.shape[0], *([1] * (x.dim() - 2))
        #     # )),
        # )

    def _forward(self, x):
        z = x * self.A.view(
            -1, self.A.shape[0], *([1] * (x.dim() - 2))
        )
        z = torch.fft.fft(z, dim=1, n=self.out_f)
        z = z * self.D.view(
            -1, self.D.shape[0], *([1] * (x.dim() - 2))
        )
        z = torch.fft.ifft(z, dim=1, n=self.out_f)
        z = z + self.bias.view(
            -1, self.bias.shape[0], *([1] * (x.dim() - 2))
        )
        return torch.gather(z, 1, 
            self.perm.view(-1, self.perm.shape[0], *([1] * (x.dim() - 2))) \
                .expand(*x.shape)
        ) / self.norm_term

    def forward(self, x):
        return self._forward(x)
        return checkpoint(self._forward, x, use_reentrant=False)

# class ACDC(nn.Module):
#     def __init__(
#             self,
#             in_f: int,
#             out_f: int,
#             to_real: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
#         ):
#         super().__init__()
#         self.out_f = out_f
#         self.register_buffer('perm', torch.randperm(out_f))
#         self.register_buffer('norm_term', torch.tensor(np.sqrt(in_f)))

#         self.A = nn.Parameter(torch.rand(1, in_f))
#         self.D = nn.Parameter(torch.rand(1, out_f))
#         self.bias = nn.Parameter(torch.rand(1, out_f))
#         self.to_real = to_real

#     def forward(self, x):
#         z = x * self.A
#         z = torch.fft.fft(z, dim=-1, n=self.out_f)
#         z = z * self.D
#         z = torch.fft.ifft(z, dim=-1, n=self.out_f)
#         z = z + self.bias
#         if self.to_real is not None:
#             z = self.to_real(z)
#         return torch.gather(z, 1, getattr(self, 'perm').repeat(z.shape[0], 1)) / self.norm_term


class ACDC2D(nn.Module):
    def __init__(
            self,
            in_f: int,
            out_f: int,
            kernel_size: int,
            group_size: int = 8,
            stride: int = 1,
            padding: int = 0,
        ):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f

        # Kernel shape & unit depth
        self.dpc_k = group_size
        num_kernels = np.ceil(out_f / group_size).astype(int)
        in_dpc_k = np.ceil(in_f / num_kernels).astype(int)
        working_size = group_size * num_kernels
        self.working_in = in_dpc_k * num_kernels

        # Depthwise convolution parameters
        self.conv = nn.Conv2d(self.working_in, working_size, kernel_size, groups=num_kernels, stride=stride, padding=padding, bias=False)
        self.convi = nn.Conv2d(self.working_in, working_size, kernel_size, groups=num_kernels, stride=stride, padding=padding, bias=False)

        # ACDC parameters
        self.A = nn.Parameter(torch.rand(1, working_size, 1, 1))
        self.D = nn.Parameter(torch.rand(1, working_size, 1, 1))
        self.bias = nn.Parameter(torch.rand(1, working_size, 1, 1))

        # Permutation
        self.register_buffer('perm', torch.randperm(working_size))
        self.register_buffer('norm_term', torch.tensor(np.sqrt(self.out_f))) # TODO : check if this value is correct / good

    def forward(self, x):
        assert x.shape[1] == self.in_f, f"Expected input of shape (N, {self.in_f}, H, W), got {x.shape}"
        assert x.dim() == 4, f"Expected input of shape (N, {self.in_f}, H, W), got {x.shape}"
        
        z = F.pad(x, (0, 0, 0, 0, 0, self.working_in - self.in_f, 0, 0))
        z = self.conv(z)

        # ACDC along feature dimension
        z = z * self.A
        z = torch.fft.fft(z, dim=1)
        z = z * self.D
        z = torch.fft.ifft(z, dim=1)
        z = z + self.bias

        return torch.gather(
            z,
            1,
            self.perm.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) \
                .expand(z.shape[0], -1, z.shape[2], z.shape[3])
        )[:, :self.out_f, :, :] / self.norm_term