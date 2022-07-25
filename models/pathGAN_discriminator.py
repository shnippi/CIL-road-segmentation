import torch
import torch.nn as nn
import numpy as np

class PatchGAN_Descriminator(nn.Module):
    # Change in_channels to 6 for rgb
    def __init__(self, in_channels=4, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial_Block = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1  if feature == features[-1] else 2)
            )
            in_channels = feature
        
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        '''
        :param x: Real or fake image
        :param y: Real or fake image
        '''
        x = torch.cat([x,y], dim=1)
        x = self.initial_Block(x)
        x = self.model(x)
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.conv(x)

    
def test():
    x = torch.randn((1, 3, 512, 512))
    y = torch.randn((1, 1, 512, 512))
    out = torch.randn((1, 1, 42, 42))
    model = PatchGAN_Descriminator(in_channels=4)
    preds = model(x, y)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(sum([np.prod(p.size()) for p in model_parameters]))
    assert preds.shape == out.shape


if __name__ == "__main__":
    test()
