# find flow model
# load flow model
# sample with loaded model


from argparse import ArgumentParser
import torch
import numpy as np
import os
from glob import glob

import pytorch_lightning as pl

from ..datasets import FlowMatchingData
from ..nn import CGFlow
from ..utils import units

def look_up_chkpt_file(possible_path):
    if os.path.isfile(possible_path):
        chkpt_file = possible_path
    else:
        versions = glob(os.path.join(possible_path, "version_*"))
        if len(versions) > 0:
            last_version = max([int(p.split("_")[-1]) for p in versions])
            possible_path = os.path.join(possible_path, f"version_{last_version}")
        if os.path.isdir(os.path.join(possible_path, "checkpoints")):
            possible_path = os.path.join(possible_path, "checkpoints")
        ckpts = glob(os.path.join(possible_path, "epoch*.ckpt"))
        assert len(ckpts) >= 1, f"No checkpoint found at path: `{possible_path}`. Please check input and consider using the absolute path to the checkpoint file."
        assert len(ckpts) <= 1, f"Multiple checkpoints found at path: `{possible_path}`. Please check input and consider using the absolute path to the checkpoint file."
        chkpt_file = ckpts[0]
    return chkpt_file

def main():
    parser = ArgumentParser()
    parser = FlowMatchingData.add_model_specific_args(parser)
    parser.add_argument("--chkpt-path", type=str, default=None)
    parser.add_argument("--name", type=str, default="prot")
    parser.add_argument("--n-samples", type=int, default=50000)
    args = parser.parse_args()
    #print(args)
    
    chkpt_path = args.chkpt_path
    if chkpt_path is None:
        # try to locate the checkpoint folder from the dataset options
        data = FlowMatchingData(**vars(args))
        # only the train_size and cv_fold_describe are required, no need to load the dataset in real
        # data.prepare_data()
        # data.setup()        
        chkpt_path = f"./output/cgflow_{args.name}_{data.train_size}_{data.cv_fold_describe}"

    chkpt_file = look_up_chkpt_file(chkpt_path)
        
    cgflow = CGFlow.load_from_checkpoint(chkpt_file)
    print(f"Flow model has been loaded from `{chkpt_file}`.")
    print(f"{next(cgflow.parameters()).device}")
    xyz, energies, forces = cgflow.sample_to_cpu(args.n_samples, batch_size=1024)
    np.savez(
        chkpt_file+".npz",
        samples=xyz,
        energies=energies,
        forces=forces,
    )

    print("All OK!")

if __name__ == "__main__":
    main()


