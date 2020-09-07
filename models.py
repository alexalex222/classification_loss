import math
from torch import nn
import torchvision


def init_layer(layer):
    # Initialization using fan-in
    if isinstance(layer, nn.Conv2d):
        n = layer.kernel_size[0]*layer.kernel_size[1]*layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2.0/float(n)))
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.fill_(0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.C = nn.Conv2d(in_channel, out_channel, 3, padding=padding)
        self.BN = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, input_channel=3, flatten=True):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = input_channel if i == 0 else 64
            outdim = 64
            block = ConvBlock(indim, outdim, pool=(i < 4))   # only pooling for fist 4 layers
            trunk.append(block)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)

        self.final_feat_dim = 1600
        self.output_dim = 64

    def forward(self, x):
        out = self.trunk(x)
        return out

class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        basenet = torchvision.models.resnet18(pretrained=pretrained)
        self.extractor = nn.Sequential(*list(basenet.children())[:-1])
        self.output_dim = 512

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x