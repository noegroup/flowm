from argparse import ArgumentParser
import torch
import numpy as np
import os
from glob import glob

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ..datasets import FlowMatchingData
from ..nn import get_cg_top, get_cgnet_stats, CGnet
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
        ckpts = glob(os.path.join(possible_path, "epoch*.ckpt_processed.npz"))
        assert len(ckpts) >= 1, f"No sample found at path: `{possible_path}`. Please check input and consider using the absolute path to the npz-format sample file."
        assert len(ckpts) <= 1, f"Multiple sample files found at path: `{possible_path}`. Please check input and consider using the absolute path to the npz-format sample file."
        chkpt_file = ckpts[0]
    return chkpt_file

def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FlowMatchingData.add_model_specific_args(parser)
    parser = CGnet.add_model_specific_args(parser)
    parser.add_argument("--flow-samples-path", type=str)
    parser.add_argument("--n-flow-samples-for-training", type=int, default=None)
    parser.add_argument("--name", type=str, default="prot")
    parser.add_argument("--pdb", type=str)
    args = parser.parse_args()
    #print(args)
    orig_data = FlowMatchingData(**vars(args))
    orig_data.prepare_data()
    orig_data.setup()

    # get orig_data for fitting the priors & zscores
    cv_dataset = orig_data.cv_dataset
    train_size = orig_data.train_size
    fold_index = orig_data.cv_fold
    train_coords, *others = cv_dataset.get_train_set(fold_index=fold_index, train_size=train_size)
    if len(others) > 1:
        weights = others[1]
    else:
        weights = None
    cg_top = get_cg_top(args.pdb)
    zscores, harmonic_stat = get_cgnet_stats(cg_top, train_coords, weights=weights)
    
    k_BT = units.k_BT_in_kcal_per_mol(args.temp)
    loss_coeff = (0.1 * k_BT) ** 2 # loss will be in unit (kcal/mol/A) ** 2 for consistency
    cgnet = CGnet(args.pdb, zscores, harmonic_stat, loss_coeff=loss_coeff, **vars(args))
    
    dt = np.load(look_up_chkpt_file(args.flow_samples_path))
    entry_order = ["samples", "forces"]
    if "weights" in dt:
        print(f"Using weights contained in data set `args.flow_sample_path` for Flow-CGnet training.")
        entry_order.append("weights")
    del dt
    n_flow_samples = args.n_flow_samples_for_training
    flow_data = FlowMatchingData(look_up_chkpt_file(args.flow_samples_path), entry_order=entry_order, batch_size=args.batch_size, val_batch_size=args.val_batch_size, train_size=n_flow_samples)
    if n_flow_samples is None:
        n_flow_samples = "full"
    # flow_data.prepare_data()
    # flow_data.setup("train_size_compute") # here we only need to get the train_size

    # Trainer
    logger = TensorBoardLogger("output", name=f"flow_cgnet_{args.name}_{orig_data.train_size}_{orig_data.cv_fold_describe}_n_flow_samples_{n_flow_samples}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback])
    
    # Training
    trainer.fit(cgnet, flow_data)
    
    print("All OK!")

if __name__ == "__main__":
    main()


