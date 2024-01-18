import torch.nn as nn
import numpy as np
import torch

class PixelShuffle3d(nn.Module):
    """
    This class is a 3d version of pixelshuffle.
    """

    def __init__(self, upscale_factor, out_channels):
        """
        :param scale: upsample scale
        """
        super().__init__()
        self.scale = np.array(upscale_factor)
        self.out_channels = out_channels

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        n = np.sum(self.scale == 2)
        assert channels % (2 ** n) == 0
        nOut = self.out_channels

        out_depth = in_depth * self.scale[0]
        out_height = in_height * self.scale[1]
        out_width = in_width * self.scale[2]

        input_view = x.contiguous().view(batch_size, nOut, self.scale[0], self.scale[1], self.scale[2], in_depth,
                                         in_height,
                                         in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)
class ConvWithPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, scales):
        super().__init__()
        n = np.prod([scales,scales,scales])
        self.conv1 = nn.Conv3d(in_channels, 2 * in_channels, 5, 1, 2)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv3d(2 * in_channels, out_channels * n, 3, 1, 1)
        self.pixelshuffle = PixelShuffle3d(upscale_factor=[scales,scales,scales], out_channels=out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.tanh(y)
        y = self.conv2(y)
        y = self.tanh(y)
        y = self.pixelshuffle(y)
        return y