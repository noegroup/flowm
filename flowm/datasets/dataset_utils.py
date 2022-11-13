# helper functions for processing a dataset

import numpy as np

__all__ = ["CVDataset"]

def get_shuffled_indices(indices, seed):
    if isinstance(indices, int):
        indices = np.arange(indices)
    else:
        indices = np.array(indices, copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return indices

class CVDataset:
    """For splitting data set for cross validation experiments."""
    def __init__(self, *data_arrays, n_folds=5, shuffle_before=True, random_seed=42):
        self.length = len(data_arrays[0])
        assert len({len(i) for i in data_arrays}) == 1, "Input `data_arrays` have different lengths."
        self.random_seed = random_seed
        if self.length % n_folds != 0:
            discarded_frames = self.length % n_folds
            print(f"Number of input frames cannot be divided by {n_folds}, discarding the last {discarded_frames} frames.")
            self.length -= discarded_frames
        self.fold_length = self.length // n_folds
        self.n_folds = n_folds
        self.indices = np.arange(self.length)
        
        if shuffle_before:
            shuffle_indices = get_shuffled_indices(self.length, random_seed)
            self._return_arrays = [arr[shuffle_indices] for arr in data_arrays]
        else:
            self._return_arrays = data_arrays

    def _check_fold(self, fold_index):
        if type(fold_index) is not int or fold_index < 0 or fold_index >= self.n_folds:
            raise ValueError("Invalid fold_index: " + str(fold_index))

    def get_train_set(self, fold_index=0, train_size=None):
        """If train_size = None, then return the whole train set. Otherwise returns the first `train_size`
        frames after shuffling (subsampling)."""
        self._check_fold(fold_index)
        if train_size is None:
            train_size = self.fold_length * (self.n_folds - 1)
        elif type(train_size) is not int or train_size <= 0 or train_size > self.fold_length * (self.n_folds - 1):
            raise ValueError("Invalid train_size: " + str(train_size))
        train_indices = []
        for i in range(self.n_folds):
            if i != fold_index:
                train_indices.append(self.indices[i*self.fold_length:(i+1)*self.fold_length])
        train_indices = get_shuffled_indices(np.concatenate(train_indices), self.random_seed)
        return tuple(array_[train_indices[:train_size]] for array_ in self._return_arrays)

    def get_val_set(self, fold_index=0, val_size=None):
        """If val_size = None, then return the whole validation set. Otherwise returns the first `val_size`
        frames after shuffling (subsampling)."""
        self._check_fold(fold_index)
        if val_size is None:
            val_size = self.fold_length
        elif type(val_size) is not int or val_size <= 0 or val_size > self.fold_length:
            raise ValueError("Invalid val_size: " + str(val_size))
        val_indices = self.indices[fold_index*self.fold_length:(fold_index+1)*self.fold_length]
        val_indices = get_shuffled_indices(val_indices, self.random_seed)
        return tuple(array_[val_indices[:val_size]] for array_ in self._return_arrays)
