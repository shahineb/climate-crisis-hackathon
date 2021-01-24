"""
Description : Train CNN to predict next temperature given last three ones

Usage: run_cnn_training.py --root=<path_to_dataset_directory> --o=<path_to_model_weigths>

Options:
  --root=<path_to_dataset_directory>          Path to configuration file specifying execution parameters
  --o=<path_to_model_weigths>                 Path to file where image numpy array will be dumped
"""
import os
from docopt import docopt
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from progress.bar import Bar


class Conv2d(nn.Module):
    """Conv2d + BatchNorm + Dropout + ReLU
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        relu (bool, str): if True, uses ReLU - if 'learn', uses PReLU
        bn (bool): if True, uses batch normalization
        dropout (float): dropout probability
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1,
                 bias=True, relu=False, dropout=0., bn=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True) if bn else None
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout > 0 else None

        if relu:
            if relu == 'learn':
                self.relu = nn.PReLU()
            elif relu is True:
                self.relu = nn.ReLU(inplace=True)
            else:
                raise ValueError("Uknown specification for ReLU")
        else:
            self.relu = None

        # Weights initializer
        nn.init.xavier_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        if self.relu:
            x = self.relu(x)
        return x

    def output_size(self, input_size):
        """Computes output size
        Args:
            input_size (tuple): (C_in, H_in, W_in)
        """
        _, H_in, W_in = input_size
        C_out = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]
        padding = self.conv.padding[0]
        stride = self.conv.stride[0]
        H_out = int(np.floor((H_in - kernel_size + 2 * padding) / stride + 1))
        W_out = int(np.floor((W_in - kernel_size + 2 * padding) / stride + 1))
        return (C_out, H_out, W_out)


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv_1 = Conv2d(in_channels=in_channels,
                             out_channels=hidden_channels,
                             kernel_size=3,
                             bn=True)
        self.conv_2 = Conv2d(in_channels=hidden_channels,
                             out_channels=hidden_channels,
                             kernel_size=3,
                             relu='learn',
                             dropout=0.2,
                             bn=True)
        self.conv_3 = Conv2d(in_channels=hidden_channels,
                             out_channels=out_channels,
                             kernel_size=3)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class FieldsDataset(Dataset):

    def __init__(self, root):
        self.root = root
        files_names = os.listdir(self.root)
        files_names.sort()
        self.files_paths = [os.path.join(self.root, x) for x in files_names]

    def __getitem__(self, idx):
        arrays = []
        for path in self.files_paths[idx:idx + 4]:
            dataarray = xr.open_dataarray(path)
            arrays += [np.roll(dataarray.values[::-1], 360, axis=1)]
        src_tensor = torch.from_numpy(np.stack(arrays[:3])).float()
        tgt_tensor = torch.from_numpy(arrays[-1][None, ...]).float()
        return src_tensor, tgt_tensor

    def __len__(self):
        return len(self.files_paths) - 3


def train_model(model, dataset, criterion, optimizer, epochs):
    model.train()
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)

    for i in range(epochs):
        bar = Bar(f"Epoch {i}", max=len(dataset) // 4)
        for batch_idx, (src_tensor, tgt_tensor) in enumerate(iter(dataloader)):
            optimizer.zero_grad()
            output = model(src_tensor)
            loss = criterion(output, tgt_tensor)
            loss.backward()
            optimizer.step()
            bar.suffix = f"Loss {loss.item()}"
            bar.next()
    return model


if __name__ == "__main__":
    args = docopt(__doc__)

    dataset = FieldsDataset(root=args['--root'])
    model = ConvNet(in_channels=3, out_channels=1, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    epochs = 2

    model = train_model(model, dataset, criterion, optimizer, epochs)

    torch.save(model.state_dict(), args['--o'])























######
