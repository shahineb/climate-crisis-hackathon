import numpy as np
import torch.nn as nn


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
    """3 layers convolutional neural network

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        hidden_channels (int): Number of channels of intermediate layer
    """
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
