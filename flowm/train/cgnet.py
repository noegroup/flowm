from argparse import ArgumentParser
import torch
import numpy as np
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ..datasets import FlowMatchingData
from ..nn import get_cg_top, get_cgnet_stats, CGnet
from ..utils import units

def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FlowMatchingData.add_model_specific_args(parser)
    parser = CGnet.add_model_specific_args(parser)
    parser.add_argument("--name", type=str, default="prot")
    parser.add_argument("--pdb", type=str)
    args = parser.parse_args()
    #print(args)
    data = FlowMatchingData(**vars(args))
    data.prepare_data()
    data.setup()

    # get data for fitting the priors & zscores
    cv_dataset = data.cv_dataset
    train_size = data.train_size
    fold_index = data.cv_fold
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
    
    # Trainer
    logger = TensorBoardLogger("output", name=f"cgnet_{args.name}_{data.train_size}_{data.cv_fold_describe}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback])
    
    # Training
    trainer.fit(cgnet, data)
    
    print("All OK!")

if __name__ == "__main__":
    main()


