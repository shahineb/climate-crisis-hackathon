import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


class FieldsDataset(Dataset):
    """Temperature field dataset class, yields 3 consecutive temperature field
    stacked together as tensor and next temperature field used as target

    Args:
        root (str): path to netcdf fields files directory
    """

    def __init__(self, root):
        self.root = root
        files_names = os.listdir(self.root)
        files_names.sort()
        self.files_paths = [os.path.join(self.root, x) for x in files_names]
        self._set_mean_and_std()

    def _set_mean_and_std(self):
        """Computes mean and standard deviation of dataset
        """
        arrays = []
        for path in self.files_paths:
            dataarray = xr.open_dataarray(path)
            arrays += [dataarray.values]
        self.mean = np.mean(arrays)
        self.std = np.std(arrays)

    def __getitem__(self, idx):
        # Load all four arrays concerned by this item
        arrays = []
        for path in self.files_paths[idx:idx + 4]:
            dataarray = xr.open_dataarray(path)
            arrays += [np.roll(dataarray.values[::-1], 360, axis=1)]

        # Convert separately into torch tensors
        src_tensor = torch.from_numpy(np.stack(arrays[:3])).float()
        tgt_tensor = torch.from_numpy(arrays[-1][None, ...]).float()

        # Normalize
        src_tensor = (src_tensor - self.mean) / self.std
        tgt_tensor = (tgt_tensor - self.mean) / self.std
        return src_tensor, tgt_tensor

    def __len__(self):
        return len(self.files_paths) - 3
