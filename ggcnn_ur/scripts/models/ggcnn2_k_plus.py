import torch
import torch.nn as nn
import torch.nn.functional as F
from .do_conv_pytorch import DOConv2d

class GGCNN2(nn.Module):
    def __init__(self, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [16,  # First set of convs
                            16,  # Second set of convs
                            32,  # Dilated convs
                            16]  # Transpose Convs

        if dilations is None:
            dilations = [2, 4]

        self.features = nn.Sequential(
            # 4 conv layers.
            DOConv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            DOConv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            DOConv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            DOConv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dilated convolutions.
            DOConv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True),
            nn.ReLU(inplace=True),
            DOConv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True),
            nn.ReLU(inplace=True),

            # Output layers
            nn.UpsamplingBilinear2d(scale_factor=2),
            DOConv2d(filter_sizes[2], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            DOConv2d(filter_sizes[3], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pos_output = DOConv2d(filter_sizes[3], 1, kernel_size=1)
        self.cos_output = DOConv2d(filter_sizes[3], 1, kernel_size=1)
        self.sin_output = DOConv2d(filter_sizes[3], 1, kernel_size=1)
        self.width_output = DOConv2d(filter_sizes[3], 1, kernel_size=1)
        self.k_output = DOConv2d(filter_sizes[3], 1, kernel_size=1)#

        for m in self.modules():
            #if isinstance(m, (DOConv2d, nn.ConvTranspose2d)): # with weight init
            if isinstance(m, (nn.ConvTranspose2d)): # no weight init
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.features(x)

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        k_output = self.k_output(x)#

        return pos_output, cos_output, sin_output, width_output, k_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width, y_k = yc#
        pos_pred, cos_pred, sin_pred, width_pred, k_pred = self(xc)#

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)
        k_loss = F.mse_loss(k_pred, y_k)#

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss + k_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss,
                'k_loss': k_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred,
                'k': k_pred
            }
        }
