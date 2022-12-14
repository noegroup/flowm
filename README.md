# Code base for paper Flow-matching -- efficient coarse-graining of molecular dynamics without forces

## Installation
We tested the following procedure for setting up a usable `conda` environment before installing the `flowm` package:
```Bash
conda create -n flow_matching
conda activate flow_matching

conda install -c conda-forge python=3.9 numpy scipy
conda install -c pytorch -c conda-forge pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 
conda install -c conda-forge pytorch-lightning=1.6.1
conda install -c conda-forge matplotlib jupyter # recommended; necessary for making plots
conda install -c conda-forge mdtraj deeptime # for handling the molecular data

# installing bgflow
pip install git+https://github.com/noegroup/bgflow.git@34df704bbde0c90ec3497dd757c4a4b4b9b69e95 # bgflow, branch flowmatching
pip install einops nflows # dependecies for bgflow, part 1
conda install -c conda-forge openmm # dependecies for bgflow, part 2

# installing bgmol
pip install git+https://github.com/noegroup/bgmol.git@2be2ede75ce137258fc83c01b28648a8c29161de # bgmol, branch main
pip install networkx # dependency for bgmol
```


## Application
### Example commands for reproducing the experiments
#### Flow-CGnet for ala2

- Acquiring the data set
Run the `acquire_dataset.sh` in the `fetch_data` folder. Details are covered by the `README.md` file there.

Note that a special flag `--no-shuffling-before-cv-split` is used whenever the program needs to access the `ala2_cg_data.npz`, since it comprises 4 independent trajectories of equal length, and this way we can ensure all frames from each trajectory are either all used for training or all for validation.

- CGFlow training
```
python -m flowm.train.flow --data-path "./fetch_data/downloaded/ala2_cg_data.npz" \
--entry-order coords gen_Gaussian_2D --entry-scaling coords*0.1 \
--name ala2 \
--train-size 20000 --cv-fold 3 --n-cv-splits 4 \
--no-shuffling-before-cv-split \
--batch-size 256 --reload_dataloaders_every_n_epochs 1 \
--transform smooth --augmented-transform affine \
--hidden 128 1024 128 --n-torsion-blocks 2 \
--n-bond-bins 1 \
--max_epochs 100 --lr 1e-3 --lr-decay 1.0 \
--gpus 1 
```

- Drawing samples from CGFlow checkpoint and post-processing 
The raw and processed samples will be generated under the same folder where the checkpoint is located.
```
python -m flowm.sample.flow --data-path "./fetch_data/downloaded/ala2_cg_data.npz" \
--name ala2 --train-size 20000 --cv-fold 3 --n-cv-splits 4 \
--no-shuffling-before-cv-split \
--n-samples 100000

python -m flowm.sample.flow_post_process --data-path "./fetch_data/downloaded/ala2_cg_data.npz" \
--train-size 20000 --cv-fold 3 --n-cv-splits 4 \
--no-shuffling-before-cv-split --name ala2 \
--max-force-magnitude 1.5e5
```
The version above uses the train-size and fold information for automated checkpoint finding. In case of multiple checkpoints or when they are located in other folders, maybe directly pointing to an accurate path is better. Like this:
`--chkpt-path ./output/cgflow_ala2_20000_3/version_0/checkpoints/epoch=20-step=1659.ckpt`

- Flow-CGnet training
We require the same (raw) data set as well as the fold and training size settings used for training the CGFlow to be specified in the command-line arguments. The same train set will be used for fitting the parameters of the CGnet priors.
Remember to change the line for `--flow-samples-path` to corresponding post-processed sample files (npz format).
```
python -m flowm.train.flow_cgnet --data-path "./fetch_data/downloaded/ala2_cg_data.npz" \
--entry-order coords --entry-scaling coords*0.1 \
--name ala2 --pdb "./fetch_data/downloaded/ala2_cg.pdb" \
--train-size 20000 --cv-fold 3 --n-cv-splits 4 \
--no-shuffling-before-cv-split \
--flow-samples-path "./output/cgflow_ala2_20000_3/version_0/checkpoints/[YOUR_CKPT_FILE_NAME]_processed.npz" \
--batch-size 128 --val-batch-size 256 \
--prior-type NO_REPUL --activation tanh \
--num-layers 5 --width 160 \
--lipschitz_strength 4.0 --temp 300.0 \
--max_epochs 50 --lr 3e-3 --target-lr 1e-5 \
--lr-decay-freq 5 \
--gpus 1 
```
In addition, the argument `--n-flow-samples-for-training [INT]` can be used for specifying the number of flow samples used in training. The default is to take 80% as training set and the rest as validation set for Flow-CGnet training.

- Sampling with trained Flow-CGnet model
    - Lagenvin dynamics simulation
    ```
    python -m flowm.sample.simulate_cgnet --data-path "./fetch_data/downloaded/ala2_cg_data.npz" \
    --entry-order coords --entry-scaling coords*0.1 \
    --name ala2 --pdb "./fetch_data/downloaded/ala2_cg.pdb" \
    --train-size 20000 --cv-fold 3 --n-cv-splits 4 \
    --no-shuffling-before-cv-split \
    --cgnet-chkpt-path "./output/flow_cgnet_ala2_20000_3_n_flow_samples_full/version_0/checkpoints" \
    --temp-in-K 300 --n-time-steps 250000 \
    --n-indepedent-sims 100 \
    --time-step-in-ps 2e-3 --save-interval 250
    ```
    - Parallel-tempering simulation
    ```
    python -m flowm.sample.simulate_cgnet --data-path "./fetch_data/downloaded/ala2_cg_data.npz" \
    --entry-order coords --entry-scaling coords*0.1 \
    --name ala2 --pdb "./fetch_data/downloaded/ala2_cg.pdb" \
    --train-size 20000 --cv-fold 3 --n-cv-splits 4 \
    --no-shuffling-before-cv-split \
    --cgnet-chkpt-path "./output/flow_cgnet_ala2_20000_3_n_flow_samples_full/version_0/checkpoints" \
    --n-time-steps 250000 --n-indepedent-sims 100 \
    --time-step-in-ps 2e-3 --save-interval 250 \
    --use-pt --temp-in-K 300 500 --pt-exchange-interval 1000
    ```

#### Conventional CGnet for ala2
- Conventional CGnet training
```
python -m flowm.train.cgnet --data-path "./fetch_data/downloaded/ala2_cg_data.npz" \
--entry-order coords aaFs --entry-scaling coords*0.1 aaFs*16.77398445 \
--name ala2 --pdb "./fetch_data/downloaded/ala2_cg.pdb" \
--train-size 750000 --cv-fold 3 --n-cv-splits 4 \
--no-shuffling-before-cv-split \
--batch-size 128 --val-batch-size 256 \
--prior-type NO_REPUL --activation tanh \
--num-layers 5 --width 160 \
--lipschitz_strength 4.0 --temp 300.0 \
--max_epochs 50 --lr 3e-3 --target-lr 1e-5 \
--lr-decay-freq 5 \
--gpus 1 
```

- Sampling with trained CGnet model
    - Lagenvin dynamics simulation
    ```
    python -m flowm.sample.simulate_cgnet --data-path "./fetch_data/downloaded/ala2_cg_data.npz" \
    --entry-order coords --entry-scaling coords*0.1 \
    --name ala2 --pdb "./fetch_data/downloaded/ala2_cg.pdb" \
    --train-size 750000 --cv-fold 3 --n-cv-splits 4 \
    --no-shuffling-before-cv-split \
    --cgnet-chkpt-path "./output/cgnet_ala2_750000_3/version_0/checkpoints" \
    --temp-in-K 300 --n-time-steps 250000 \
    --n-indepedent-sims 100 \
    --time-step-in-ps 2e-3 --save-interval 250
    ```
    - Parallel-tempering simulation
    ```
    python -m flowm.sample.simulate_cgnet --data-path "./fetch_data/downloaded/ala2_cg_data.npz" \
    --entry-order coords --entry-scaling coords*0.1 \
    --name ala2 --pdb "./fetch_data/downloaded/ala2_cg.pdb" \
    --train-size 750000 --cv-fold 3 --n-cv-splits 4 \
    --no-shuffling-before-cv-split \
    --cgnet-chkpt-path "./output/cgnet_ala2_750000_3/version_0/checkpoints" \
    --n-time-steps 250000 --n-indepedent-sims 100 \
    --time-step-in-ps 2e-3 --save-interval 250 \
    --use-pt --temp-in-K 300 500 --pt-exchange-interval 1000
    ```

#### Flow-CGnet for fast folders
Here we take the miniprotein trpcage as an example.
- CGFlow training
```
python -m flowm.train.flow --data-path "./fetch_data/downloaded/trpcage/trpcage_ca.npz" \
--entry-order coords gen_Gaussian_2D \
--name trpcage \
--batch-size 128 --reload_dataloaders_every_n_epochs 1 \
--transform smooth \
--augmented-transform affine \
--hidden 128 1024 128 --n-torsion-blocks 4 \
--n-bond-bins 1 \
--max_epochs 25 --lr 1e-3 --lr-decay 1.0 \
--gpus 1 
```
Note: For back-compatibility with the paper results, here we disable the reloading of data loaders, i.e., the regeneration of augmented channel.

- Drawing samples from CGFlow checkpoint and post-processing 
The raw and processed samples will be generated under the same folder where the checkpoint is located.
```
python -m flowm.sample.flow --chkpt-path "./output/cgflow_trpcage_835200_80-20" \
--name trpcage --n-samples 1048576

python -m flowm.sample.flow_post_process --sample-file-path "./output/cgflow_trpcage_835200_80-20" \
--name trpcage --pdb "./fetch_data/downloaded/trpcage/trpcage_ca.pdb" \
--max-force-magnitude 8e4 \
--reweight-repul GLY_SPECIAL_REPUL
```
The version above uses the train-size and fold information for automated checkpoint finding. In case of multiple checkpoints or when they are located in other folders, maybe directly pointing to an accurate path is better. Like this:
`--chkpt-path ./output/cgflow_ala2_20000_3/version_0/checkpoints/epoch=20-step=1659.ckpt`

- Flow-CGnet training
We require the same (raw) data set as well as the fold and training size settings used for training the CGFlow to be specified in the command-line arguments. The same train set will be used for fitting the parameters of the CGnet priors.
Remember to change the line for `--flow-samples-path` to corresponding post-processed sample files (npz format).
```
python -m flowm.train.flow_cgnet --data-path "./fetch_data/downloaded/trpcage/trpcage_ca.npz" \
--entry-order coords \
--name trpcage --pdb "./fetch_data/downloaded/trpcage/trpcage_ca.pdb" \
--flow-samples-path "./output/cgflow_trpcage_835200_80-20" \
--batch-size 128 --val-batch-size 512 \
--prior-type GLY_SPECIAL_REPUL --activation silu \
--num-layers 8 --width 160 \
--lipschitz_strength 10.0 --temp 290.0 \
--max_epochs 75 --lr 3e-3 --target-lr 1e-5 \
--lr-decay-freq 15 \
--gpus 1 
```
In addition, the argument `--n-flow-samples-for-training [INT]` can be used for specifying the number of flow samples used in training. The default is to take 80% as training set and the rest as validation set for Flow-CGnet training.

- Sampling with trained Flow-CGnet model
    - Lagenvin dynamics simulation
    ```
    python -m flowm.sample.simulate_cgnet --data-path "./fetch_data/downloaded/trpcage/trpcage_ca.npz" \
    --entry-order coords \
    --name trpcage --pdb "./fetch_data/downloaded/trpcage/trpcage_ca.pdb" \
    --cgnet-chkpt-path "./output/flow_cgnet_trpcage_835200_80-20_n_flow_samples_full/version_0/checkpoints" \
    --temp-in-K 290 --n-time-steps 250000 \
    --n-indepedent-sims 100 \
    --time-step-in-ps 2e-3 --save-interval 250
    ```
    - Parallel-tempering simulation
    ```
    python -m flowm.sample.simulate_cgnet --data-path "./fetch_data/downloaded/trpcage/trpcage_ca.npz" \
    --entry-order coords \
    --name trpcage --pdb "./fetch_data/downloaded/trpcage/trpcage_ca.pdb" \
    --cgnet-chkpt-path "./output/flow_cgnet_trpcage_835200_80-20_n_flow_samples_full/version_0/checkpoints" \
    --n-time-steps 250000 --n-indepedent-sims 100 \
    --time-step-in-ps 2e-3 --save-interval 250 \
    --use-pt --temp-in-K 290 381 500 --pt-exchange-interval 1000
    ```

### Side note: be careful about the unit of inputs and outputs.
More information on this can be found in the file `fetch_data/README.md`.
#### Conventional all-atom dataset:
- coords: Angstrom (ala2) or nm
- forces: kcal/mol/A or kcal/mol/nm

#### Flow sample output:
- coords: nm
- forces: k\_BT/nm
- energy: k\_BT

#### CGnet training input:
- coords: nm
- forces: k\_BT/nm
(When the input does not correspond to this list, then a unit conversion via `--entry-scaling` arguments is necessary)
However, in order to keep the force matching error comparable with the 

#### CGnet simulation output:
- coords: nm
