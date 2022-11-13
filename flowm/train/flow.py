from argparse import ArgumentParser
import torch
import numpy as np
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ..datasets import FlowMatchingData
from ..nn import get_flow_marginals, CGFlow
from ..utils import units

def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FlowMatchingData.add_model_specific_args(parser)
    parser = CGFlow.add_model_specific_args(parser)
    parser.add_argument("--name", type=str, default="prot")
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
    n_particles, marginals = get_flow_marginals(train_coords, weights=weights)
    # print(n_particles, marginals)
    cgflow = CGFlow(n_particles, marginals, **vars(args))
    
    # Trainer
    logger = TensorBoardLogger("output", name=f"cgflow_{args.name}_{data.train_size}_{data.cv_fold_describe}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback])
    
    # Training
    trainer.fit(cgflow, data)
    
    print("All OK!")

if __name__ == "__main__":
    main()


