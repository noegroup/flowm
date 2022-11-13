import numpy as np
import torch
import torch.nn as nn
from .model import EnergyModel


__all__ = ["CGnetRepulPrior", "CGnetPrior"]

class EnergyModelWithRepulInfo(EnergyModel):
    """Energy unit is k_BT."""
    DEFAULT_REPUL = {"exclusion_volume": 0.45, "repulsion_exponential": 6}
    REDUCED_REPUL = {"exclusion_volume": 0.35, "repulsion_exponential": 6}
    AWSEM_REPUL = {"exclusion_volume": 0.4, "repulsion_exponential": 1, "use_awsem": True, "awsem_coeff": 400}
    SHIFTED_AWSEM_REPUL = {"exclusion_volume": 0.45, "repulsion_exponential": 1, "use_awsem": True, "awsem_coeff": 400}
    BOOSTED_AWSEM_REPUL = {"exclusion_volume": 0.4, "repulsion_exponential": 1, "use_awsem": True, "awsem_coeff": 1600}
    NO_REPUL = {"exclusion_volume": 0., "repulsion_exponential": 1}

class CGnetRepulPrior(EnergyModelWithRepulInfo):
    """Internally we always use kB_T as energy unit. Can be used in a standalone manner."""
    def __init__(self, featurizer, repulsion_dict=None, embed_feat=False):
        super(CGnetRepulPrior, self).__init__()
        if repulsion_dict is None:
            repulsion_dict = self.NO_REPUL
        dtype = featurizer._device_indicator.dtype
        ex_vol = torch.zeros(featurizer.dim)
        ex_vol[featurizer.indices("dists")] = repulsion_dict["exclusion_volume"]
        rep_exp = torch.ones_like(ex_vol, dtype=torch.int) * int(repulsion_dict["repulsion_exponential"])
        if repulsion_dict.get("use_awsem", False):
            self._use_awsem = True
            ex_vol[featurizer.indices("torsions")] = -np.pi - 1 # make sure that torsion features don't contribute! see .repul(...)
            awsem_coeff = torch.tensor(repulsion_dict["awsem_coeff"], dtype=dtype)
            self._awsem_coeff = nn.Parameter(awsem_coeff, requires_grad=False)
        else:
            self._use_awsem = False
        self._ex_vol = nn.Parameter(ex_vol, requires_grad=False)
        self._rep_exp = nn.Parameter(rep_exp, requires_grad=False)
        self._embed_feat = embed_feat
        if self._embed_feat:
            self._feat = featurizer

    def setup_special_gly_repul(self, topology):
        pair_a, pair_b = np.triu_indices(topology.n_residues, 2)
        gly_dists = []
        for i, resi in enumerate(topology.residues):
            if resi.code == "G":
                print(i, resi)
                gly_dists += np.concatenate([np.argwhere(pair_a == i)[:, 0], np.argwhere(pair_b == i)[:, 0]]).tolist()
        gly_dists = np.unique(gly_dists)
        print("Number of GLY-related non-neighboring pairwise distances:", len(gly_dists))
        print("Adapting the repulsion parameters accordingly.")
        n_atoms = topology.n_atoms
        start = (n_atoms - 1) + (n_atoms - 2) + (n_atoms - 3) # bonds, angles and dihedrals
        self._ex_vol[start:] = 0.42 # others should be stoped way earlier
        self._ex_vol[start + gly_dists] = 0.36 # LEU7 - GLY11 special
        
    def forward(self, pos_or_feat, **kwargs):
        if self._embed_feat:
            feat = self._feat(pos_or_feat)
        else:
            feat = pos_or_feat
        if self._use_awsem:
            zero_t = torch.tensor(0., dtype=torch.float32, device=feat.device)
            energy = self._awsem_coeff * torch.sum(torch.where(feat < self._ex_vol, (feat - self._ex_vol).square(), zero_t), 1, keepdim=True)
        else:
            energy = torch.sum((self._ex_vol / feat) ** self._rep_exp, 1, keepdim=True)
        return energy

class CGnetPrior(EnergyModelWithRepulInfo):
    """Internally we always use kB_T as energy unit."""
    def __init__(self, featurizer, harmonic_dict, repulsion_dict=None, embed_feat=False):
        super(CGnetPrior, self).__init__()
        assert featurizer.dim == len(harmonic_dict["harmonic_means"]), 'Given `harmonic_dict` does not corresponds to the `featurizer`.'
        dtype = featurizer._device_indicator.dtype # 
        harm_means = torch.tensor(harmonic_dict["harmonic_means"], dtype=dtype)
        harm_ks = torch.tensor(harmonic_dict["harmonic_ks"], dtype=dtype)
        self._harm_means = nn.Parameter(harm_means, requires_grad=False)
        self._harm_ks = nn.Parameter(harm_ks, requires_grad=False)
        self._embed_feat = embed_feat
        if self._embed_feat:
            self._feat = featurizer
        # dealing with repulsion
        if repulsion_dict is not None:
            self._use_repul = True
            self.repul = CGnetRepulPrior(featurizer, repulsion_dict=repulsion_dict, embed_feat=False)

    def setup_special_gly_repul(self, topology):
        self.repul.setup_special_gly_repul(topology)

    def forward(self, pos_or_feat, **kwargs):
        if self._embed_feat:
            feat = self._feat(pos_or_feat)
        else:
            feat = pos_or_feat
        energy = torch.sum(self._harm_ks * (feat - self._harm_means) ** 2, 1, keepdim=True) / 2
        if self._use_repul:
            energy += self.repul(feat)
        return energy
