import mdtraj as md
from glob import glob
import numpy as np

DESRES_DCD_TEMPL = "/PATH/TO/DESRES/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/2JOF-0-protein-*.dcd"

t = md.load("./trpcage.pdb")
t.save_pdb("trpcage.pdb")
ca_indices = t.top.select("name CA")
t.atom_slice(ca_indices).save_pdb("trpcage_ca.pdb")
dcd_files = sorted(glob(DESRES_DCD_TEMPL))
t_full = md.load(dcd_files, top=t.top)
ca_coords = t_full.xyz[:, ca_indices]
print(ca_coords.shape)
np.savez("trpcage_ca.npz", coords=ca_coords)

