import numpy as np
import torch
from .geom import FeatureCalc, DistCalc, AngleCalc, DihedralCalc
from .linear_mol_utils import get_linear_molecule_feature as get_feat


__all__ = ["LinearMolFeaturizer"]

class LinearMolFeaturizer(FeatureCalc):
    """For calculating CGnet features for a linear molecule."""
    def __init__(self, mdtraj_top, include_nonbond=True):
        super(LinearMolFeaturizer, self).__init__()
        self._top = mdtraj_top
        self._n_beads = self._top.n_atoms
        self.bond_feat = DistCalc(get_feat(self._n_beads, "bonds"))
        self.angle_feat = AngleCalc(get_feat(self._n_beads, "angles"))
        self.dihedral_feat = DihedralCalc(get_feat(self._n_beads, "dihedrals"))
        if include_nonbond:
            self.include_nonbond = True
            self.nonbond_feat = DistCalc(get_feat(self._n_beads, "nonbonds"))
        else:
            self.include_nonbond = False
    
    @property
    def dim_bonds(self):
        return self.bond_feat.dim
    
    @property
    def dim_angles(self):
        return self.angle_feat.dim
    
    @property
    def dim_torsions(self):
        return self.dihedral_feat.dim
    
    @property
    def dim_dists(self):
        if self.include_nonbond:
            return self.nonbond_feat.dim
        else:
            return 0
    
    @property
    def dim(self):
        return self.dim_bonds + self.dim_angles + self.dim_torsions + self.dim_dists
    
    def select(self, feat="bonds"):
        """Return the range selector for corresponding feature."""
        if feat == "bonds":
            return np.s_[:, :self.dim_bonds]
        elif feat == "angles":
            return np.s_[:, self.dim_bonds:(self.dim_bonds + self.dim_angles)]
        elif feat == "torsions":
            return np.s_[:, (self.dim_bonds + self.dim_angles):(self.dim_bonds + self.dim_angles + self.dim_torsions)]
        elif feat == "dists":
            return np.s_[:, -self.dim_dists:]
        elif feat == "nondists":
            return np.s_[:, :(self.dim_bonds + self.dim_angles + self.dim_torsions)]
        else:
            raise KeyError(f"Given feature `{feat}` is not computed.")
    
    def indices(self, feat="bonds"):
        """Return the indices for corresponding feature."""
        lb, ub = 0, self.dim_bonds
        if feat == "bonds":
            return np.arange(lb, ub)
        lb, ub = ub, ub + self.dim_angles
        if feat == "angles":
            return np.arange(lb, ub)
        lb, ub = ub, ub + self.dim_torsions
        if feat == "torsions":
            return np.arange(lb, ub)
        lb, ub = ub, ub + self.dim_dists
        if feat == "dists":
            return np.arange(lb, ub)
        if feat == "nondists":
            return np.arange(0, lb)
        raise KeyError(f"Given feature `{feat}` is not computed.")
    
    def forward(self, coords):
        if coords.dim() != 3 or coords.shape[1] != self._n_beads:
            raise ValueError("Unsupported `coords` shape.")
        # assuming the input coords in shape (N_frames, N_beads, 3)
        bonds = self.bond_feat(coords)
        angles = self.angle_feat(coords)
        dihedrals = self.dihedral_feat(coords)
        features = [bonds, angles, dihedrals]
        if self.include_nonbond:
            nonbond_dists = self.nonbond_feat(coords)
            features.append(nonbond_dists)
        return torch.cat(features, dim=1)
    
    def get_harmonic_stat_from_data(self, coords: np.ndarray, weights=None, target="all"):
        """Harmonic potential parameters from statistics in unit k_BT."""
        get_bonds = target in ("bonds", "all")
        get_angles = target in ("angles", "all")
        if not get_bonds and not get_angles:
            raise ValueError('`target` should be one from "bonds", "angles", "all"')
        feats = self.calc(coords)
        harm_means = np.zeros((self.dim,), dtype=np.float32)
        harm_ks = np.zeros((self.dim,), dtype=np.float32)
        if get_bonds:
            harm_indices = self.indices("bonds")
#             harm_means[harm_indices] = feats[self.select("bonds")].mean(axis=0)
#             harm_ks[harm_indices] = 1 / feats[self.select("bonds")].var(axis=0)
            bond_feats = feats[self.select("bonds")]
            bond_means = np.average(bond_feats, axis=0, weights=weights)
            harm_means[harm_indices] = bond_means
            bond_vars = np.average((bond_feats - bond_means) ** 2, axis=0, weights=weights)
            harm_ks[harm_indices] = 1 / bond_vars
        if get_angles:
            harm_indices = self.indices("angles")
#             harm_means[harm_indices] = feats[self.select("angles")].mean(axis=0)
#             harm_ks[harm_indices] = 1 / feats[self.select("angles")].var(axis=0)
            angle_feats = feats[self.select("angles")]
            angle_means = np.average(angle_feats, axis=0, weights=weights)
            harm_means[harm_indices] = angle_means
            angle_vars = np.average((angle_feats - angle_means) ** 2, axis=0, weights=weights)
            harm_ks[harm_indices] = 1 / angle_vars
        return {"harmonic_means": harm_means, "harmonic_ks": harm_ks}
    
    def get_zscores_from_data(self, coords: np.ndarray, weights=None):
        feats = self.calc(coords)
#         means = feats.mean(axis=0)
#         stds = feats.std(axis=0)
        means = np.average(feats, axis=0, weights=weights)
        vars_ = np.average((feats - means) ** 2, axis=0, weights=weights)
        stds = np.sqrt(vars_)
    
        zscores = np.vstack([means, stds])
        return zscores

