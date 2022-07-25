import torch 
import torch.nn as nn
import numpy as np

class Pix2PixHD_Generator(nn.Module):
    def __init__(self, in_channles=3, out_channles=3, ngf=64, norm_layer=nn.BatchNorm2d, n_downsampling=3, n_resblocks=9):
        super().__init__()

        # First Block
        self.padd1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channles, ngf, kernel_size=7, padding=0)
        self.norm1 = norm_layer(ngf)
        self.act1  = nn.ReLU()

        # Downsampling blocks
        self.downsampling_blocks = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2**i
            downsampling_block = BlockDown(
                in_channels=ngf * mult, 
                out_channels=ngf * mult * 2,
                norm_layer=norm_layer
            )
            self.downsampling_blocks.append(downsampling_block)

        # Resnet blocks
        self.resnet_blocks = nn.ModuleList()
        mult = 2**n_downsampling
        for i in range(n_resblocks):
            resnet_block = ResnetBlock(n_channels=ngf*mult, norm_layer=norm_layer)
            self.resnet_blocks.append(resnet_block)


        # Upsample blocks
        self.upsampling_blocks = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
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

        for block in self.downsampling_blocks:
            x = block(x)
        for block in self.resnet_blocks:
            x = block(x)
        for block in self.upsampling_blocks:
            x = block(x)

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



class Pix2PixHD_Descriminator(nn.Module):
    def __init__(self, in_channles=6, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, num_D=3):
        super().__init__()
        
        self.num_D = num_D
        self.n_layers = n_layers
        """
        self.descriminators = []
        for i in range(num_D):
            # We actually need num_D different Desciminators even though they have the exact
            # same architecure. But they all will get a differentely scaled input image. D1 should
            # learn somthing different than D2. Hence they also need to have different weights.
            self.descriminators.append(
                NLayer_Descriminator(in_channles, ndf, n_layers, norm_layer)
            )
        """
        self.descriminator1 = NLayer_Descriminator(in_channles, ndf, n_layers, norm_layer)
        self.descriminator2 = NLayer_Descriminator(in_channles, ndf, n_layers, norm_layer)
        self.descriminator3 = NLayer_Descriminator(in_channles, ndf, n_layers, norm_layer)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)


    def forward(self, x):
        result = []

        """
        for i in range(self.num_D):
            # Since we save all intermediate features for the feature matching loss we need
            # to also save them. Therefore features_resolution_i gets a list of all these features
            # where the last entry is the feature map which we pass to the next layer
            features_resolution_i = self.descriminators[i](x)
            result.append(features_resolution_i)
            
            # In the last layer we don't need to downsample anymore (unncessary computation)
            if i != self.num_D-1:
                x = self.downsample(x)
        """

        result.append(self.descriminator1(x))
        x = self.downsample(x)
        result.append(self.descriminator2(x))
        x = self.downsample(x)
        result.append(self.descriminator3(x))

        return result


class NLayer_Descriminator(nn.Module):
    # PatchGAN descriminator
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()

        kernel_size = 4
        padw = int(np.ceil((kernel_size-1.0)/2))

        # Inital Conv layer
        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padw)
        self.act1  = nn.LeakyReLU(0.2)

        # Other layers (stirde=2)
        nf = ndf
        self.descriminator_layers = nn.ModuleList()
        for i in range(1, n_layers):
            in_channels = nf
            nf = min(nf * 2, 512)
            descriminator_layer = DescriminatorLayer(in_channels, nf, kernel_size, 2, padw, norm_layer)
            self.descriminator_layers.append(descriminator_layer)   

        # Stride=1 Layer
        in_channels = nf
        nf = min(nf * 2, 512)
        self.stride_layer = DescriminatorLayer(in_channels, nf, kernel_size, 1, padw, norm_layer)

        # Last layer
        self.conv2 = nn.Conv2d(nf, 1, kernel_size=kernel_size, stride=1, padding=padw)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # We need to save intermediate feature for the feature matching loss
        result = []

        # First layer
        x = self.conv1(x)
        x = self.act1(x)
        result.append(x)

        # The different blocks
        for layer in self.descriminator_layers:
            x = layer.forward(x)
            result.append(x)

        # Stride layer
        x = self.stride_layer(x)
        result.append(x)

        # Last layer
        x = self.conv2(x)
        x = self.sigmoid(x)
        result.append(x)

        return result


class DescriminatorLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padw, norm_layer):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padw)
        self.norm = norm_layer(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_gen():
    x = torch.randn(1,3,512,512)
    model = Pix2PixHD_Generator()
    print(count_parameters(model))
    preds = model(x)

def test_disc():
    x = torch.randn(1,6,512,512)
    model = Pix2PixHD_Descriminator()
    print(count_parameters(model))
    preds = model(x)

if __name__ == "__main__":
    test_gen()
    test_disc()
    