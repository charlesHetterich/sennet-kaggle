import torch
import torch.nn as nn
from PIL import Image

from typing import Callable
import numpy as np
import io

from matplotlib import pyplot as plt
import shutil
import tempfile
import torch.utils.tensorboard as tb
import subprocess

from ipywidgets import interact
import ipywidgets as widgets
import matplotlib.colors as mcolors

class LogBoard:
    def __init__(self, log_dir: str, port: int = 6006):
        self.log_dir = log_dir
        self.port = port

    def launch(self):
        shutil.rmtree(tempfile.gettempdir() + "/.tensorboard-info", ignore_errors=True) # sort of 'force reload' for tensorboard
        command = [
            'tensorboard',
            '--logdir', self.log_dir,
            '--reload_interval', '1',
            '--port', str(self.port)
        ]
        subprocess.Popen(command)

    def clear(self, folder: str = None):
        if folder is None:
            shutil.rmtree(self.log_dir, ignore_errors=True)
        else:
            shutil.rmtree(f"{self.log_dir}/{folder}", ignore_errors=True)

    def get_logger(self, name: str):
        return tb.SummaryWriter(f"{self.log_dir}/{name}")

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def do_analysis(f: Callable[[torch.Tensor], torch.Tensor], extra_stats) -> dict:
    """
    returns pandas dataframe of 
    """
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], profile_memory=True, with_flops=True) as prof:        
        f()

    time_values = [x.cuda_time_total for x in prof.key_averages()]
    flop_values = [x.flops for x in prof.key_averages()]
    mem_values = [x.cuda_memory_usage for x in prof.key_averages()]
    return extra_stats | {
        "time_ns": np.sum(time_values),
        "time_var": np.var(time_values),
        "flops_mean": np.mean(flop_values),
        "flops_var": np.var(flop_values),
        "mem_mb": np.max(mem_values) / (1024 ** 2),
        "profiler": prof
    }

class Display:
    def __init__(self, scan: torch.Tensor = None, mask: torch.Tensor = None):
        self.scan = scan
        self.mask = mask

    def _view_slice(self, i: int, slice_dim: int, ax: plt.Axes = None):
        ax.set_facecolor('black')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        slice_idx = [slice(None), slice(None), slice(None)]
        slice_idx[slice_dim] = i

        if self.scan is not None:
            ax.imshow(self.scan[0][tuple(slice_idx)], cmap='gray')
        if self.mask is not None:
            if self.mask.shape[0] == 1:
                ax.imshow(self.mask[0][tuple(slice_idx)], cmap=mcolors.ListedColormap(['none', 'blue']), alpha=0.5)
            else:
                ax.imshow(self.mask[0][tuple(slice_idx)], cmap=mcolors.ListedColormap(['none', 'green']), alpha=0.5)
                ax.imshow(self.mask[1][tuple(slice_idx)], cmap=mcolors.ListedColormap(['none', 'red']), alpha=0.5)
                ax.imshow(self.mask[2][tuple(slice_idx)], cmap=mcolors.ListedColormap(['none', 'yellow']), alpha=0.5)

    @staticmethod
    def _view_slices(i: int, displays: list['Display'], slice_dim: int):
        _, axs = plt.subplots(1, len(displays), figsize=(15, 15))
        if len(displays) == 1:
            axs = [axs]
        for ax, display in zip(axs, displays):
            display._view_slice(i, slice_dim, ax)

    @staticmethod
    def view(displays: list['Display'], slice_dim: int = 0):
        slider_max = displays[0].scan.shape[slice_dim+1] - 1 if displays[0].scan is not None else displays[0].mask.shape[slice_dim+1] - 1
        slider  = widgets.IntSlider(min=0, max=slider_max, step=1, value=0)
        widgets.interact(Display._view_slices, i=slider, displays=widgets.fixed(displays), slice_dim=widgets.fixed(slice_dim))

def mask_plots(scan: torch.Tensor, masks: list[torch.Tensor], pmasks: list[torch.Tensor]):
    fig, axes = plt.subplots(4, len(masks), figsize=(20, 10))
    if len(masks) == 1:
        axes = axes[:, None]
    
    for i, (mask, pmask) in enumerate(zip(masks, pmasks)):
        mask = mask[0].unsqueeze(0)
        pmask = pmask[0].unsqueeze(0)
        det_mask = torch.cat([ #collect TP, FP, FN
            mask * pmask, # true positive
            (1 - mask) * pmask, # false positive
            mask * (1 - pmask) # false negative
        ], dim=0)
        Display(scan=scan[0].unsqueeze(1).cpu())._view_slice(0, 0, axes[0][i])
        Display(mask=det_mask.cpu())._view_slice(0, 0, axes[1][i])
        Display(scan=mask.cpu())._view_slice(0, 0, axes[2][i])
        Display(scan=pmask.cpu())._view_slice(0, 0, axes[3][i])

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return np.array(Image.open(buf))

def _mask_plots(scan: torch.Tensor, masks: list[torch.Tensor], pmasks: list[torch.Tensor]):
    fig, axes = plt.subplots(3, len(masks), figsize=(20, 10))
    if len(masks) == 1:
        axes = axes[:, None]
    
    for i, (mask, pmask) in enumerate(zip(masks, pmasks)):
        axes[0][i].axis('off')
        axes[0][i].imshow(scan[0, 0, :,:, scan.shape[2] // 2].detach().cpu(), cmap='gray')
        axes[1][i].axis('off')
        axes[1][i].imshow(mask[0, 0, :,:, mask.shape[2] // 2].detach().cpu(), cmap='gray')
        axes[2][i].axis('off')
        axes[2][i].imshow(pmask[0, 0, :,:, pmask.shape[2] // 2].detach().cpu(), cmap='gray')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return np.array(Image.open(buf))

def mask_plots2d(scan: torch.Tensor, masks: list[torch.Tensor], pmasks: list[torch.Tensor]):
    fig, axes = plt.subplots(3, len(masks), figsize=(20, 10))
    if len(masks) == 1:
        axes = axes[:, None]
    
    for i, (mask, pmask) in enumerate(zip(masks, pmasks)):
        axes[0][i].axis('off')
        axes[0][i].imshow(scan[0, 0].detach().cpu(), cmap='gray')
        axes[1][i].axis('off')
        axes[1][i].imshow(mask[0, 0].detach().cpu(), cmap='gray')
        axes[2][i].axis('off')
        axes[2][i].imshow(pmask[0, 0].detach().cpu(), cmap='gray')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return np.array(Image.open(buf))