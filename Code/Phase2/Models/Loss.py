import numpy as np
import torch
import torch.nn as nn

def geodesic_loss(ground_truth:torch.Tensor, measured: torch.Tensor, epsilon=1e-7, reduction='none'):
    """ Calculated Loss for quaternion/orientation error """
    R_diffs = measured @ ground_truth.permute(0, 2, 1)
    # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
    traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
    dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + epsilon, 1 - epsilon))
    if reduction == 'none':
        return dists
    elif reduction == 'mean':
        return dists.mean()
    elif reduction == 'sum':
        return dists.sum()
    else:
        raise ValueError(f"Unknown reduction setting. Currently '{reduction}'")
