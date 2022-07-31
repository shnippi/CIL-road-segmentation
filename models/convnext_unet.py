"""
# 
# This file contains our U-Net implementation that uses ConvNext as a backbone for feature extraction.
#  
"""


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.nn.functional as F
import numpy as np

#  U-Net with ConvNext as backbone
class ConvNext_Unet(nn.Module):
    def __init__(
            self, 
            in_channels=3, 
            out_channels=1, 
            features=[128, 256, 256, 256],
            convnext_depths = [3, 3, 27, 3],
            convnext_dims = [256, 512, 1024, 2048]
    ):
        super(ConvNext_Unet, self).__init__()

        self.backbone = ConvNeXt(depths=convnext_depths, dims=convnext_dims, num_classes=1000)
        checkpoint = torch.load('models/pretrained/convnext_xlarge_22k_1k_384_ema.pth', 'cuda')
        self.backbone.load_state_dict(checkpoint["model"])
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        #self.model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], num_classes=21841)
        #url = 'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth'
        #checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cuda")
        #self.model.load_state_dict(checkpoint["model"])

        self.layer4_conv = nn.Conv2d(convnext_dims[-1], features[-1], kernel_size=3, padding=1)
        self.layer4_norm = LayerNorm(features[-1], eps=1e-6, data_format="channels_first")
        self.layer4_act = nn.GELU()
        self.up_4_to_3 = nn.ConvTranspose2d(features[-1], features[-1], kernel_size=2, stride=2)

        self.layer3_conv = nn.Conv2d(features[-1] + convnext_dims[-2], features[-2], kernel_size=3, padding=1)
        self.layer3_norm = LayerNorm(features[-2], eps=1e-6, data_format="channels_first")
        self.layer3_act = nn.GELU()
        self.up_3_to_2 = nn.ConvTranspose2d(features[-2], features[-2], kernel_size=2, stride=2)

        self.layer2_conv = nn.Conv2d(features[-2] + convnext_dims[-3], features[-3], kernel_size=3, padding=1)
        self.layer2_norm = LayerNorm(features[-3], eps=1e-6, data_format="channels_first")
        self.layer2_act = nn.GELU()
        self.up_2_to_1 = nn.ConvTranspose2d(features[-3], features[-3], kernel_size=2, stride=2)

        self.layer1_conv = nn.Conv2d(features[-3] + convnext_dims[-4], features[-4], kernel_size=3, padding=1)
        self.layer1_norm = LayerNorm(features[-4], eps=1e-6, data_format="channels_first")
        self.layer1_act = nn.GELU()
        self.up_1_to_0 = nn.ConvTranspose2d(features[-4], features[-4], kernel_size=2, stride=2)

        # 192, 192
        self.layer0_conv = nn.Conv2d(features[-4], 64, kernel_size=3, padding=1)
        self.layer0_norm = LayerNorm(64, eps=1e-6, data_format="channels_first")
        self.layer0_act = nn.GELU()
        self.up_to_384 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        # 384, 384
        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.toplayer_conv = nn.Conv2d(64, 64, kernel_size=3)
        self.toplayer_norm = LayerNorm(64, eps=1e-6, data_format="channels_first")
        self.toplayer_act = nn.GELU()

        # Predict
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        # x1 = (1, 256, 96, 96)
        # x2 = (1, 512, 48, 48)
        # x3 = (1, 1024, 24, 24)
        # x4 = (1, 2048, 12, 12)
        x1, x2, x3, x4 = self.backbone(input)

        x = self.layer4_conv(x4)
        x = self.layer4_norm(x)
        x = self.layer4_act(x)
        x = self.up_4_to_3(x)

        x = torch.cat((x, x3), dim=1)
        x = self.layer3_conv(x)
        x = self.layer3_norm(x)
        x = self.layer3_act(x)
        x = self.up_3_to_2(x)

        x = torch.cat((x, x2), dim=1)
        x = self.layer2_conv(x)
        x = self.layer2_norm(x)
        x = self.layer2_act(x)
        x = self.up_2_to_1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.layer1_conv(x)
        x = self.layer1_norm(x)
        x = self.layer1_act(x)
        x = self.up_1_to_0(x)

        x = self.layer0_conv(x)
        x = self.layer0_norm(x)
        x = self.layer0_act(x)
        x = self.up_to_384(x)

        # Top layer here also comes input again
        input = self.input_conv(input)
        x = torch.cat((x, input), dim=1)
        self.toplayer_conv(x)
        self.toplayer_norm(x)
        self.toplayer_act(x)

        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x


# ConvNext from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        
        our_results = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            our_results.append(x)
        return our_results

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def test():
    x = torch.randn((1, 3, 384, 384))
    out = torch.randn((1, 1, 384, 384))
    model = ConvNext_Unet(in_channels=3, out_channels=1)
    preds = model(x)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(sum([np.prod(p.size()) for p in model_parameters]))
    assert preds.shape == out.shape


if __name__ == "__main__":
    test()
