### Note about home-brew datasets accompanying the publication:
Datasets for capped alanine (ala2) and capped chignolin (CLN) can be downloaded from FU Berlin FTP via `acquire_data.sh`.
Units:
coords: Angstrom
aaFs: kcal/mol/Angstrom
weights: (CLN only) 1

### Note about processing the DESRES data sets: 
Downloaded DESRES includes the system definition in Maestro format (system.mae), which is not supported by our Python scripts. Therefore, it needs to be transformed into the PDB format.
One tested way for this transformation is via the software VMD. After installing VMD, copy the mae file over, open a bash shell and run the following to gain a usable PDB file (Remember to change the file name of the mae input in the script):
```
cd [FOLDER_CONTAINING_???_-protein.mae]
vmd -dispdev text < example_transform_mae.tcl
```
This is a PDB file for all protein atoms. Afterwards, the full dataset can be generated via example_coords_extract.py, which also yields a corresponding PDB file that contains only alpha carbon atoms.

The detailed names and paths in `example_transform_mae.tcl` and `example_coords_extract.py` need to be adapted. For the four fast folding proteins used in our experiments, here are the correponding names:
- chignolin: CLN025-0-protein
- trpcage: 2JOF-0-protein
- bba: 1FME-0-protein, 1FME-0-protein (we concatenated the coordinates for flow training)
- villin: 2F4K-0-protein

### Info for adding new data set:
1. Ensure that the entries follow the following conventions:
coords: \[N_frames, N_CA_atoms, 3\] in unit nm, (required)
forces: \[N_frames, N_CA_atoms, 3\] in unit k_BT/mol/nm, (optional; required for CGnets)
weights: \[N_frames,\] (optional; required for reweighted CGnet training)

2. When the physical units are different, then it is required to provide a correct conversion factor via command-line arguments.
Examples dataset `ala2_raw_data.npz`:
- coords: Angstrom to nm
- forces: kcal/mol/Angstrom to k_BT/mol/nm (T = 300K)
We provide the following converting factors for CGnet training: `--entry-order coords aaFs --entry-scaling coords*0.1 aaFs*16.77398445`
