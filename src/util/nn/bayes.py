import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Union

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

        # if not isinstance(modules, list):
        #     modules = [modules]
        # self._registered_bayesian_modules.extend(
        #     u for u in modules if isinstance(u, Bayes)
        # )

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

class Conv3dBayes(Bayes):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple], 
                 stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, 
                 groups: int = 1, padding_mode: str = 'zeros', target_var=0.5,):
        super().__init__()
        # self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        # nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        # Bayes.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, ) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride, ) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding, ) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, ) * 3
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(out_channels))
        self.weightvar = torch.nn.Parameter(torch.ones_like(self.weight) * target_var)
        self.biasvar = torch.nn.Parameter(torch.ones_like(self.bias) * target_var)
        self.register_buffer('norm_term', torch.tensor(np.sqrt(np.max([self.in_channels, self.out_channels]))))
        self.register_buffer('target_var', torch.tensor(target_var))

    # def __init__(self, *args, target_var=0.3, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     # self.target_var = target_var
    #     self.weightvar = torch.nn.Parameter(torch.ones_like(self.weight) * target_var)
    #     self.biasvar = torch.nn.Parameter(torch.ones_like(self.bias) * target_var)
    #     self.register_buffer('target_var', torch.tensor(target_var))
    #     self.register_buffer(
    #         'norm_term',
    #         torch.tensor(np.sqrt(
    #             np.max([self.in_channels, self.out_channels])
    #         ))
    #     )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, 
                 self.weight + torch.randn_like(self.weight) * self.weightvar, 
                 self.bias + torch.randn_like(self.bias) * self.biasvar,
                 self.stride, self.padding, self.dilation, self.groups) / self.norm_term

    def penalty(self) -> torch.Tensor:
        return (self.weight ** 2).mean() + ((self.target_var - self.weightvar) ** 2).mean() + \
            (self.bias ** 2).mean() + ((self.target_var - self.biasvar) ** 2).mean()
    
    def penalize(self, alpha: float = 1e-3):
        p = (self.penalty() * alpha)
        p.backward()
        return p
    
    def decay_var(self, gamma: float = 0.5):
        self.weightvar.data *= gamma
        self.biasvar.data *= gamma