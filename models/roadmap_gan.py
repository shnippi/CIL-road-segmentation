import torch
import torch.nn as nn

class Pix2Pix_Generator(nn.Module):
    def __init__(self, in_channles=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channles, features, 4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        
        self.down1 = BlockDown(features*1, features*2, act="leaky", use_dropout=False)
        self.down2 = BlockDown(features*2, features*4, act="leaky", use_dropout=False)
        self.down3 = BlockDown(features*4, features*8, act="leaky", use_dropout=False)
        self.down4 = BlockDown(features*8, features*8, act="leaky", use_dropout=False)
        self.down5 = BlockDown(features*8, features*8, act="leaky", use_dropout=False)
        self.down6 = BlockDown(features*8, features*8, act="leaky", use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, stride=2, padding=1, padding_mode="reflect"),
            nn.ReLU(),
        )

        self.up1 = BlockUp(features*8*1, features*8, act="relu", use_dropout=True)
        self.up2 = BlockUp(features*8*2, features*8, act="relu", use_dropout=True)
        self.up3 = BlockUp(features*8*2, features*8, act="relu", use_dropout=True)
        self.up4 = BlockUp(features*8*2, features*8, act="relu", use_dropout=False)
        self.up5 = BlockUp(features*8*2, features*4, act="relu", use_dropout=False)
        self.up6 = BlockUp(features*4*2, features*2, act="relu", use_dropout=False)
        self.up7 = BlockUp(features*2*2, features*1, act="relu", use_dropout=False)

        self.finalup = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channles, 4, stride=2, padding=1),
            nn.Tanh(),    
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down6(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1,d7], 1))
        up3 = self.up3(torch.cat([up2,d6], 1))
        up4 = self.up4(torch.cat([up3,d5], 1))
        up5 = self.up5(torch.cat([up4,d4], 1))
        up6 = self.up6(torch.cat([up5,d3], 1))
        up7 = self.up7(torch.cat([up6,d2], 1))

        final = self.finalup(torch.cat([up7, d1], 1))

        return final


class Pix2Pix_Descriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial_Block = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
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


class BlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x) if self.use_dropout else x
        return x


class BlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x) if self.use_dropout else x
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


'''
def test():
    x = torch.randn(1,3,512,512)
    model = Pix2Pix_Generator()
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()
'''