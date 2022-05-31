import torch
import torch.nn as nn

class Pix2PixHD_Generator(nn.Module):
    def __init__(self, in_channles=6, ngf=64, norm_layer=nn.BatchNorm2d, n_downsampling=3):
        super().__init__()

        # First Block
        self.padd1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(6, in_channles, kernel_size=7, padding=0)
        self.norm1 = norm_layer(ngf)
        self.act1  = nn.ReLU()

        # Downsampling blocks
        self.downsampling_Blocks = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2**i
            downsampling_block = BlockDown(
                in_channles=ngf * mult, 
                out_chanlles=ngf * mult * 2,
                norm_layer=norm_layer
            )
            self.downsampling_Blocks.append(downsampling_block)

        # Resnet blocks


    def forward():

        return



class BlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(out_channels)
        self.act  = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__():
        super().__init__()

    def forward(x):
        return x