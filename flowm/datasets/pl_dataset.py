# a universal data set + loader for pytorch lightning

# plan:
# for ala2 and cgn scaling experiments: 4 folds cross-validation-style split of train-val sets
# for fast-folders: 1 fold -> 80-20 split of train-val datasets

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from .dataset_utils import CVDataset


__all__ = ["FlowMatchingData"]

def get_n_threads():
    avail_n_thread = torch.get_num_threads()
    return max(4, avail_n_thread)

def generate_data(n_frames, name="gen_Gaussian_2D"):
    if name.startswith("gen_Gaussian_") and name.endswith("D"):
        try:
            dim = int(name[13:-1])
        except:
            raise ValueError(f"Invalid dimension setting: `{name[13:-1]}`")
        return torch.randn(n_frames, dim)
    else:
        raise NotImplementedError("Currently only support `gen_Gaussian_nD`")

class FlowMatchingData(pl.LightningDataModule):

    def __init__(self, 
                 data_path, 
                 entry_order=["coords", "gen_Gaussian_2D"], 
                 entry_scaling={},
                 batch_size=128, 
                 val_batch_size=None, 
                 stride=1, 
                 n_cv_splits=1, 
                 cv_fold=0, 
                 train_size=None, 
                 no_shuffling_before_cv_split=False, 
                 **kwargs):
        """
        Initialize a pytorch lightning data module for the flow-matching method.
        Args:
        data_path:      (path-like object) can be either processed all-atom simulation data, or flow samples
        entry_order:    (a list of str) describing the entries that should be included in the batched data,
                        also defining the order of all entries. When the entry name contains "gen_" prefix, 
                        the data is generated on the fly. Otherwise the name will be treated as a key for 
                        the provided protein_data npz file.
                        Presets: 
                        - Flow training: ["coords", "gen_Gaussian_2D"]
                        - (Flow-)CGnet training: ["coords", "forces"]
                        - Flow-CGnet training on reweighted samples: ["coords", "forces", "weights"]
        entry_scaling:  (dict str -> float): the multiplicative scaling factor for correponding entries.
                        Mostly used for forces, e.g., between physical units and reduced units (k_BT).
                        Also possible for coordinates, e.g., between nm and Angstrom.
                        When omitted, then the scaling factor is by default 1.
        batch_size:     (int) number of samples in each training batch
        val_batch_size: (optional int) number of samples in each validation batch. 
                        If `None`, falls back to `batch_size`.
        stride:         (int) the stride when loading data entries from the npz file
        n_cv_splits:    (int) when > 1: split the data set to corresponding number of folds.
                        when = 1: split the data set with 80-20 train-val ratio.
        cv_fold:        (int, [0, n_cv_splits)) which fold after the cv split will be used for validation.
                        No function when n_cv_splits == 1.
        train_size:     (optional int) the maximum size of training data. When kept as `None`, all available
                        data after striding and splitting will be used.
        no_shuffling_before_cv_split: (bool) whether the data set should be shuffled previous to the train-val split. Usually kept false except for ala2 with n_cv_splits=4, since the dataset is built by concatenating 4 independent simulations.
        
        """
        super().__init__()
        self.data_path = data_path
        self.entry_order = entry_order
        self.entry_scaling = entry_scaling
        for entry in self.entry_scaling:
            assert entry in self.entry_order, (f"Input `entry_scaling` contains non existing key "
                                               f"{entry} in `entry_order`.")           
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        if val_batch_size is None:
            self.val_batch_size = self.batch_size
        self.stride = stride
        self.n_cv_splits = n_cv_splits
        self.cv_fold = cv_fold
        self.cv_fold_describe = cv_fold
        if self.n_cv_splits == 1: # 80-20 split = 5 CV folds with the last fold as val set
            self.n_cv_splits = 5
            self.cv_fold = 4
            self.cv_fold_describe = "80-20"
        self.train_size = train_size
        self.shuffle_before_split = not no_shuffling_before_cv_split
        if not self.shuffle_before_split:
            print("Warning: the data set will not be shuffled before the CV split. Please make sure the dataset composition and order correspond to the `n-cv-splits`")

    @staticmethod
    def add_model_specific_args(parent_parser):
        import argparse
        class ParseFloatDict(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest, dict())
                for value in values:
                    try:
                        key, value = value.split('*')
                        getattr(namespace, self.dest)[key] = float(value)
                    except:
                        raise ValueError("Expecting the multiplicative scaling factors "
                                         "be given according to the following form '[entry]*factor [entry2]*factor2 ...'"
                                         " e.g., 'coords*0.1 forces*10'")
        # for pl support
        parser = parent_parser.add_argument_group("FlowMatchingData")
        parser.add_argument("--data-path", type=str)
        parser.add_argument("--entry-order", nargs='+', default=["coords", "gen_Gaussian_2D"])
        parser.add_argument("--entry-scaling", nargs='*', action=ParseFloatDict, default={})
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--val-batch-size", type=int, default=None)
        parser.add_argument("--stride", type=int, default=1)
        parser.add_argument("--n-cv-splits", type=int, default=1)
        parser.add_argument("--cv-fold", type=int, default=0)
        parser.add_argument("--train-size", type=int, default=None)
        parser.add_argument("--no-shuffling-before-cv-split", action='store_true')
        return parent_parser

    @staticmethod
    def read_dataset(protein):
        print("Read dataset...", end="", flush=True)
        xyz = []
        for index in range(FastFolderDesres.n_trajectories(protein)):
            dataset = FastFolderDesres(protein, index=index, selection="ca", read=True)
            xyz.append(dataset.xyz)
        print("done.")
        return np.concatenate(xyz, axis=0)

    def prepare_data(self):
        # tasks run first on single process
        pass
        #FastFolderData.read_dataset(self.protein)

    def setup(self, stage=0):
        # tasks run after prepare_data, could be indepentdently over multiple processes in parallelization.        
        dt = np.load(self.data_path) # assuming unit: coords: nm, forces: kJ/mol/nm
        data_arrays = []
        self._dataset_size = 0
        for entry in self.entry_order:
            if entry.startswith("gen_"):
                arr = None # generate_data(self.length, entry)
            else:
                assert entry in dt, f"Required entry {entry} not found in file {self.data_path}."
                arr = dt[entry]
                self._dataset_size = max(self._dataset_size, len(arr))
                if entry in self.entry_scaling:
                    arr *= self.entry_scaling[entry]
            data_arrays.append(arr)
        # decide whether to use the minimal validation set size
        max_val_size = self._dataset_size // self.n_cv_splits
        max_train_size = max_val_size * (self.n_cv_splits - 1)
        if self.train_size is None:
            self.train_size = max_train_size
        elif self.train_size > max_train_size:
            raise ValueError(f"Given `train_size` excceeds the maximum available training data points {max_train_size}.")
        if self.train_size >= 2.5e4 * (self.n_cv_splits - 1):
            self.val_size = int(self.train_size / (self.n_cv_splits - 1))
        else:
            self.val_size = min(25000, max_val_size) # otherwise it could be too small that the validation loss is unreliable for model seelection
        if stage != "train_size_compute":
            for i, entry in enumerate(self.entry_order):
                if entry.startswith("gen_"):
                    data_arrays[i] = generate_data(self._dataset_size, entry).numpy()
            self.cv_dataset = CVDataset(*data_arrays,
                                        n_folds=self.n_cv_splits,
                                        shuffle_before=self.shuffle_before_split,
                                        random_seed=2342361,
                                       )   

    def train_dataloader(self):
        if not hasattr(self, "_train_sets"):
            self._train_sets = (
                self.cv_dataset.get_train_set(fold_index=self.cv_fold,
                                              train_size=self.train_size)
            )
            for i, entry in enumerate(self.entry_order):
                if not entry.startswith("gen_"):
                    self._actual_train_size = len(self._train_sets[i])
                    break
        # check and newly generate data for all "gen_*" entries
        train_data = []
        for i, entry in enumerate(self.entry_order):
            if entry.startswith("gen_"):
                train_data.append(generate_data(self._actual_train_size, entry))
            else:
                train_data.append(torch.as_tensor(self._train_sets[i]))

        return DataLoader(
            TensorDataset(*train_data),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=get_n_threads()
        )

    def val_dataloader(self):
        if not hasattr(self, "_val_sets"):
            self._val_sets = (
                self.cv_dataset.get_val_set(fold_index=self.cv_fold,
                                            val_size=self.val_size)
            )
        val_data = [torch.as_tensor(arr) for arr in self._val_sets]
        return DataLoader(
            TensorDataset(*val_data),
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=get_n_threads()
        )
    



