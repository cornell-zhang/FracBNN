'''
Reference:
    https://github.com/akamaster/pytorch_resnet_cifar10

Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
import utils.quantization as q


__all__ = ['ResNet', 'resnet20']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    '''
    Proposed ReActNet model variant.
    For details, please refer to our paper:
        https://arxiv.org/abs/2012.12206
    '''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.rprelu1 = q.RPReLU(in_channels=planes)
        self.rprelu2 = q.RPReLU(in_channels=planes)

        self.conv1 = q.PGBinaryConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = q.PGBinaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    LambdaLayer(lambda x: torch.cat((x, x), dim=1)),
            )

        self.binarize = q.QuantSign.apply
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn4 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.rprelu1(self.bn1(self.conv1(self.binarize(x)))) + self.shortcut(x)
        x = self.bn3(x)
        x = self.rprelu2(self.bn2(self.conv2(self.binarize(x)))) + x
        x = self.bn4(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, batch_size=128, num_gpus=1):
        super(ResNet, self).__init__()
        self.in_planes = 16

        ''' The input layer is binarized! '''
        self.conv1 = q.BinaryConv2d(96, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        """ N = batch_size / num_gpus """
        assert batch_size % num_gpus == 0, \
            "Given batch size cannot evenly distributed to available gpus."
        N = batch_size // num_gpus
        self.encoder = q.InputEncoder(input_size=(N,3,32,32), resolution=8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10, batch_size=128, num_gpus=1):
    print("Binary Input PG PreAct RPrelu ResNet-20 BNN")
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes,
                  batch_size=batch_size, num_gpus=num_gpus)


if __name__ == "__main__":
    from thop import profile
    from thop.vision import basic_hooks as hooks
    model = resnet20().cuda()
    input = torch.randn(1, 3, 32, 32)
    output = model(input)

