import numpy as np

def build_indices(n_atoms, tuple_len=2):
    return np.array([np.arange(i, n_atoms - tuple_len + 1 + i) for i in np.arange(tuple_len)]).T

def get_linear_molecule_feature(n_beads, feat="bonds"):
    """`feat`: "bonds", "angles", "dihedrals", "nonbonds". """
    if feat == "bonds":
        return build_indices(n_beads, 2)
    elif feat == "angles":
        return build_indices(n_beads, 3)
    elif feat == "dihedrals":
        return build_indices(n_beads, 4)
    elif feat == "nonbonds":
        return np.vstack(np.triu_indices(n_beads, 2)).T
    else:
        raise ValueError(f"Unrecognized `feat`: {feat}")

