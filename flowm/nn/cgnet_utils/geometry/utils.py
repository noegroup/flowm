import torch
from torch import nn
import numpy as np

__all__ = ["compute_dists", "compute_angles", "compute_dihedrals"]

# ------------------------------------------
# Low level functions
# ------------------------------------------

def dist_vectors(pos, pair_indices):
    """Compute distance vectors from `pos` for pairs indicated by `pair_indices`.
    
    Inputs:
    pos: tensor of shape [..., N_atoms, 3]
    pair_indices: int tensor of shape [N_pairs, 2], in which the integers are within range [0, N_atoms). (won't check)

    Output:
    pair_vectors: tensor of shape [..., N_pairs, 3] for vector from point indicated by the first column pointing to those to the second column."""
    return pos[..., pair_indices[:, 1], :] - pos[..., pair_indices[:, 0], :]

# TODO: combination with dist_vectors to gain performance?
class _VectorNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_ij):
        d_ij = torch.sqrt(torch.sum(torch.square(v_ij), -1))
        ctx.save_for_backward(v_ij, d_ij)
        return d_ij

    @staticmethod
    def backward(ctx, dy):
        v_ij, d_ij = ctx.saved_tensors
        matrix = dy[..., None] * v_ij / (d_ij[..., None] + 1e-6)
        return matrix
vnorm = _VectorNormFunc.apply # a zero-safe vector norm function

def vdot(v1, v2, keepdim=False):
    """Dot product for vectors without checking dimensions."""
    return (v1 * v2).sum(dim=-1, keepdim=keepdim)

def vangle_cos(v1, v2):
    """Return the cosine of the angle formed by two vectors without checking dimensions."""
    return vdot(v1, v2) / (vnorm(v1) * vnorm(v2))

def vangle(v1, v2):
    """Return the angle formed by two vectors without checking dimensions."""
    return torch.arccos(vangle_cos(v1, v2))

# ------------------------------------------
# High level functions for computing individual internal coordinates
# ------------------------------------------

def compute_dists(pos, pair_indices):
    ab = dist_vectors(pos, pair_indices)
    l_ab = vnorm(ab)
    return l_ab

def compute_angles(pos, angle_indices):
    ba = dist_vectors(pos, angle_indices[:, [1, 0]])
    bc = dist_vectors(pos, angle_indices[:, [1, 2]])
    a_abc = vangle(ba, bc)
    return a_abc

def compute_dihedrals(pos, dihedral_indices):
    ab = dist_vectors(pos, dihedral_indices[:, [0, 1]])
    bc = dist_vectors(pos, dihedral_indices[:, [1, 2]])
    cd = dist_vectors(pos, dihedral_indices[:, [2, 3]])
    n1 = torch.cross(ab, bc, dim=-1)
    n2 = torch.cross(bc, cd, dim=-1)
    norm2 = vnorm(n2)
    m1 = torch.cross(n1, bc / torch.unsqueeze(vnorm(bc), -1), dim=-1)
    y = vdot(m1, n2) / vnorm(m1) / norm2
    x = vdot(n1, n2) / vnorm(n1) / norm2
    theta = torch.atan2(-y, x)
    return theta
