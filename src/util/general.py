import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Edge(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('kernel', torch.tensor([
            [0,1,0],
            [1,-4,1],
            [0,1,0]
        ]).float().reshape(1,1,3,3))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.conv2d(x, self.kernel, padding=1) != 0) * x

class ContEdge(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('kernel', torch.tensor([
            [0,1,0],
            [1,-4,1],
            [0,1,0]
        ]).float().reshape(1,1,3,3))
        self.register_buffer('max', torch.tensor(4.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((F.conv2d(x, self.kernel, padding=1).abs())) / self.max

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean', smooth=1e-6):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('gamma', torch.tensor(gamma))
        self.register_buffer('eps', torch.tensor(1e-6))
        self.reduction = reduction

    def forward(self, p_y: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pixel_class_weights = self.alpha * y + (1 - self.alpha) * (1 - y)
        BCE_loss = F.binary_cross_entropy(p_y, y, reduction='none') # * pixel_class_weights
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss * pixel_class_weights

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class DiceScore(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.register_buffer('smooth', torch.tensor(smooth))

    def forward(self, p_y: torch.Tensor, y: torch.Tensor, mode: str = 'together') -> torch.Tensor:
        assert mode in ['together', 'separate'], "Mode must be 'together' or 'separate'"
        flat_prob = p_y.view(p_y.shape[0], -1)
        flat_y = y.view(y.shape[0], -1)

        if mode == 'together':
            intersection = (flat_prob * flat_y).sum()
            cardinality = flat_prob.sum() + flat_y.sum()
            return (2. * intersection + self.smooth) / (cardinality + self.smooth)

        elif mode == 'separate':
            intersection = (flat_prob * flat_y).sum(1)
            cardinality = flat_prob.sum(1) + flat_y.sum(1)
            return ((2. * intersection + self.smooth) / (cardinality + self.smooth))

class EdgeWeightedDiceLoss(nn.Module):
    def __init__(self, alpha: float, smooth=1e-6):
        super().__init__()
        self.register_buffer('smooth', torch.tensor(smooth))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.edge = ContEdge()

    def forward(self, p_y: torch.Tensor, y: torch.Tensor, together: bool = True) -> torch.Tensor:
        # find edges
        y_edge = self.edge(y)
        y_interior = y - y_edge
        p_edge_weights = self.edge((p_y).float())
        p_edge = p_y * p_edge_weights
        # p_edge_and_exterior = p_y * (1 - y_interior)
        p_interior = p_y * (1 - p_edge_weights)

        # flatten
        y_edge = y_edge.view(y_edge.shape[0], -1)
        y_interior = y_interior.view(y_interior.shape[0], -1)
        p_edge = p_edge.view(p_edge.shape[0], -1)
        p_interior = p_interior.view(p_interior.shape[0], -1)
        
        # calculate dice
        if together:
            intersection = self.alpha * (p_edge * y_edge).sum() + \
                (1 - self.alpha) * (p_interior * y_interior).sum()
            cardinality = self.alpha * (p_edge.sum() + y_edge.sum()) + \
                (1 - self.alpha) * (p_interior.sum() + y_interior.sum())
            return (1 - ((2. * intersection + self.smooth) / (cardinality + self.smooth)))

        else:
            intersection = self.alpha * (p_edge * y_edge).sum(1) + \
                (1 - self.alpha) * (p_interior * y_interior).sum(1)
            cardinality = self.alpha * (p_edge.sum(1) + y_edge.sum(1)) + \
                (1 - self.alpha) * (p_interior.sum(1) + y_interior.sum(1))

            return (1 - ((2. * intersection + self.smooth) / (cardinality + self.smooth)))

class PatchAugment(nn.Module):
    def __init__(self, no_rotate: bool = False):
        super().__init__()
        self.rx, self.ry, self.rz = np.random.randint(0, 4, 3)
        if no_rotate:
            self.rx, self.ry, self.rz = None, None, None
        self.fx, self.fy, self.fz = np.random.random(3) < 0.5

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if self.rx is not None:
            u = torch.rot90(u, k=self.rx, dims=(2, 3))
            u = torch.rot90(u, k=self.ry, dims=(2, 4))
            u = torch.rot90(u, k=self.rz, dims=(3, 4))
        if self.fx:
            u = torch.flip(u, dims=(2,))
        if self.fy:
            u = torch.flip(u, dims=(3,))
        if self.fz:
            u = torch.flip(u, dims=(4,))
        return u.contiguous()

class AggDice:
        def __init__(self, scan: torch.Tensor, label: torch.Tensor, patch_size: int = 16):
            super().__init__()
            self.scan = scan
            self.label = label
            self.agg_preds = torch.full_like(scan, -1) # -1 is (unseen)
            self.agg_intersection = torch.zeros([])
            self.pred_sum = torch.zeros([])
            self.seen_label_sum = torch.zeros([])
            self.label_sum = label.sum()
            self.patch_size = patch_size

        def _cube_at_pos(self, u: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
            return u[
                :,
                pos[0]:pos[0] + self.patch_size,
                pos[1]:pos[1] + self.patch_size,
                pos[2]:pos[2] + self.patch_size
            ]

        def __call__(self, p_ys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            for p_y, pos in zip(p_ys, positions):
                agg_pred = self._cube_at_pos(self.agg_preds, pos)
                unsceen_pred = agg_pred.where(agg_pred == -1, torch.zeros_like(agg_pred)) * -1
                cur_pred = agg_pred.where(agg_pred != -1, torch.zeros_like(agg_pred))

                y = self._cube_at_pos(self.label, pos)

                # update dice stats
                self.agg_intersection += (y * p_y).sum() - (y * cur_pred).sum()
                self.seen_label_sum += (y * unsceen_pred).sum()
                self.pred_sum += (p_y.sum() - cur_pred.sum())

                # overwrite agg_pred with new prediction
                agg_pred[...] = p_y

                return (2 * self.agg_intersection + 1e-6) / (self.seen_label_sum + self.pred_sum + 1e-6), \
                    (2 * self.agg_intersection + 1e-6) / (self.label_sum + self.pred_sum + 1e-6)