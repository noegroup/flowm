import torch
import torch.nn as nn

__all__ = ["WeightedFMLoss"]

class WeightedFMLoss(torch.nn.Module):
    """Loss function for CGnet training."""
    def __init__(self, FM_coeff=1.0):
        super(WeightedFMLoss, self).__init__()
        # input check
        if FM_coeff < 0:
            raise ValueError("FM coefficient should be nonnegative.")
        self._fm_coeff = nn.Parameter(torch.tensor(FM_coeff, dtype=torch.float32), requires_grad=False)
    
    def forward(self, pred_forces, label_forces, weights=None):
        if weights is None:
            f_mse_all = (pred_forces - label_forces).square().mean()
            return self._fm_coeff * f_mse_all
        else:
            f_mse = (pred_forces - label_forces).square().mean(dim=(-2, -1))
            return self._fm_coeff * ((f_mse * weights).sum() / weights.sum())