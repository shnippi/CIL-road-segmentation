import torch.nn as nn

class Pix2PixHD_Generator(nn.Module):
    def __init__(self, in_channles=3, out_channles=3, ngf=64, norm_layer=nn.BatchNorm2d, n_downsampling=3, n_resblocks=9):
        super().__init__()

        # First Block
        self.padd1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(6, in_channles, kernel_size=7, padding=0)
        self.norm1 = norm_layer(ngf)
        self.act1  = nn.ReLU()

        # Downsampling blocks
        self.downsampling_blocks = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2**i
            downsampling_block = BlockDown(
                in_channles=ngf * mult, 
                out_chanlles=ngf * mult * 2,
                norm_layer=norm_layer
            )
            self.downsampling_blocks.append(downsampling_block)

        # Resnet blocks
        self.resnet_blocks = nn.ModuleList()
        mult = 2**n_downsampling
        for i in range(n_resblocks):
            resnet_block = ResnetBlock(channels=ngf*mult, norm_layer=norm_layer)
            self.resnet_blocks.append(resnet_block)


        # Upsample blocks
        self.upsampling_blocks = nn.ModuleList()
        for i in range(n_downsampling):
            mult  =2**(n_downsampling-1)
            upsampling_block = BlockUp(
                in_channels=ngf*mult,
                out_channels=int(ngf*mult/2),
                norm_layer=norm_layer
            )
            self.upsampling_blocks.append(upsampling_block)

        # Last Block
        self.padd2 = nn.ReflectionPad2d(3)
        self.conv2 = nn.Conv2d(ngf, out_channles, kernel_size=7, padding=0)
        self.act2  = nn.Tanh()

    def forward(self, x):
        x = self.padd1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.downsampling_blocks(x)
        x = self.resnet_blocks(x)
        x = self.upsampling_blocks(x)

        x = self.padd2(x)
        x = self.conv2(x)
        x = self.act2(x)

        return x



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
    def __init__(self, n_channels, norm_layer):
        super().__init__()

        self.padd1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0)
        self.norm1 = norm_layer(n_channels)
        self.act1  = nn.ReLU()

        self.padd2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0)
        self.norm2 = norm_layer(n_channels)

    def forward(self, x):
        x_out = self.padd1(x)
        x_out = self.conv1(x_out)
        x_out = self.norm1(x_out)
        x_out = self.act1(x_out)
        x_out = self.padd2(x_out)
        x_out = self.conv2(x_out)
        x_out = self.norm2(x_out)

        return x + x_out


class BlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm1 = norm_layer(out_channels)
        self.act1  = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        return x