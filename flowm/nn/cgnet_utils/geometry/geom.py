# intention: one topology, multi frames, similar to mdtraj's `compute_*` with a pregiven topology

import torch
from torch import nn
import numpy as np

from .utils import *


__all__ = ["FeatureCalc", "DistCalc", "AngleCalc", "DihedralCalc"]

class FeatureCalc(nn.Module):
    """Base class to allow direct computation on numpy arrays without boilerplate code."""
    def __init__(self):
        super(FeatureCalc, self).__init__()
        # facilitates checking on which device the module resides
        self._device_indicator = nn.Parameter(torch.empty(0), requires_grad=False)
    
    @property
    def device(self):
        return self._device_indicator.device
    
    @property
    def dtype(self):
        return self._device_indicator.dtype
    
    @property
    def dim(self):
        raise NotImplementedError(f"Please set the `dim` for class {type(self)}.")
    
    def calc(self, pos):
        """Calculate features according to `pos` in batch without gradient, 
        which does not care whether the input is a tensor or numpy array,
        and split the data into batches for large input during processing."""
        if isinstance(pos, np.ndarray):
            is_numpy = True
            pos = torch.as_tensor(pos).to(self.device)
        else:
            is_numpy = False
        results = []
        current_loc = 0
        with torch.no_grad():
            for i in range(int(np.ceil(len(pos) / 10000))):
                results.append(self.forward(pos[i*10000:(i+1)*10000]))
            results = torch.cat(results)
        if is_numpy:
            results = results.cpu().numpy()
        return results

class DistCalc(FeatureCalc):
    def __init__(self, pair_indices):
        # input shape check: `bond_indices`: [N_bonds, 2]
        if len(pair_indices.shape) != 2 or pair_indices.shape[-1] != 2:
            raise ValueError(f"Input `bond_indices` has shape {pair_indices.shape}, which does not comply with required shape [N_bonds, 2].")
        super(DistCalc, self).__init__()
        self._pair_indices = torch.as_tensor(pair_indices)
    
    @property
    def pair_indices(self):
        return self._pair_indices.clone().detach()
    
    @property
    def dim(self):
        return self._pair_indices.size(0)
    
    def forward(self, pos):
        return compute_dists(pos, self._pair_indices)

class AngleCalc(FeatureCalc):
    def __init__(self, angle_indices):
        # input shape check: `angle_indices`: [N_angles, 3]
        if len(angle_indices.shape) != 2 or angle_indices.shape[-1] != 3:
            raise ValueError(f"Input `angle_indices` has shape {angle_indices.shape}, which does not comply with required shape [N_angles, 3].")
        super(AngleCalc, self).__init__()
        self._angle_indices = torch.as_tensor(angle_indices)
    
    @property
    def angle_indices(self):
        return self._angle_indices.clone().detach()
    
    @property
    def dim(self):
        return self._angle_indices.size(0)
            
    def forward(self, pos):
        return compute_angles(pos, self._angle_indices)

class DihedralCalc(FeatureCalc):
    def __init__(self, dihedral_indices):
        # input shape check: `angle_indices`: [N_angles, 4]
        if len(dihedral_indices.shape) != 2 or dihedral_indices.shape[-1] != 4:
            raise ValueError(f"Input `dihedral_indices` has shape {dihedral_indices.shape}, which does not comply with required shape [N_angles, 4].")
        super(DihedralCalc, self).__init__()
        self._dihedral_indices = torch.as_tensor(dihedral_indices)
    
    @property
    def dihedral_indices(self):
        return self._dihedral_indices.clone().detach()
    
    @property
    def dim(self):
        return self._dihedral_indices.size(0)

    def forward(self, pos):
        return compute_dihedrals(pos, self._dihedral_indices)

