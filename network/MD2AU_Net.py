import torch
import torch.nn as nn
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': './Model_weights/resnet18-5c106cde.pth',
    'resnet34': './Model_weights/resnet34-333f7ec4.pth',
    'resnet50': './Model_weights/resnet50-19c8e357.pth',
    'resnet101': './Model_weights/resnet101-5d3b4d8f.pth',
    'resnet152': './Model_weights/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class DeformableSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.offset_conv = nn.Conv2d(
        in_channels,
        2 * kernel_size * kernel_size,  
        kernel_size=3,
        padding=1
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.deform_conv = nn.Conv2d(
           in_channels, in_channels, kernel_size=kernel_size,
            padding=kernel_size//2, bias=False
        )

        self.fusion_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, bias=False
        )

        nn.init.kaiming_normal_(self.deform_conv.weight)
        nn.init.kaiming_normal_(self.fusion_conv.weight)

    def forward(self, x):
        offset = self.offset_conv(x)  
        
        out = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.deform_conv.weight,
            padding=self.kernel_size//2
        )
        
        out = self.fusion_conv(out)

        return out
    
class MultiScaleChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)  
        self.pool2 = nn.AdaptiveAvgPool2d(2)  
        self.pool3 = nn.AdaptiveAvgPool2d(4)  

        self.fc = nn.Sequential(
            nn.Linear(in_channels*(1 + 4 + 16), in_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        B, C, H, W = x.shape
        y1 = self.pool1(x).view(B, C)     
        y2 = self.pool2(x).view(B, C*4)     
        y3 = self.pool3(x).view(B, C*16)             

        y = torch.cat([y1, y2, y3], dim=1)          
        scale = self.fc(y).view(B, C, 1, 1)         

        return x * scale
        
class MDDA(nn.Module):
    def __init__(self, in_channels):
            super().__init__()
            self.deform_att = DeformableSpatialAttention(in_channels)
            self.channel_att = MultiScaleChannelAttention(in_channels)

            # 改进特征融合
            self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
            self.conv2 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)

            self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
            spatial = self.deform_att(x)
            channel = self.channel_att(x)

            spatial_feat = self.conv1(spatial)
            channel_feat = self.conv2(channel)

            fused = torch.cat([spatial_feat, channel_feat], dim=1)
            fused = self.fusion(fused)
            return fused


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class _ResNetEncoder(nn.Module):

    def __init__(self, block, layers, pretrained=False):
        self.inplanes = 64
        super(_ResNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.attention1 = MS_DDAM(64)  
        self.attention2 = MS_DDAM(128) 
        self.attention3 = MS_DDAM(256) 
        self.attention4 = MS_DDAM(512)  

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self._initialize(pretrained=pretrained)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        if pretrained:
            checkpoint = torch.load(model_urls['resnet18'])
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x1 = self.layer1(x1)
        block1 = self.attention1(x1)

        x2 = self.layer2(block1)
        block2 = self.attention2(x2)

        x3 = self.layer3(block2)
        block3 = self.attention3(x3)

        x4 = self.layer4(block3)
        block4 = self.attention4(x4)

        return x, block1, block2, block3, block4


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):  
        super().__init__() 
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.InstanceNorm2d(out_channels),  
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class _Final_BNReluConv(nn.Module):
     def __init__(self, num_input_features, num_classes):
         super(_Final_BNReluConv, self).__init__()
         self.bn = nn.InstanceNorm2d(num_input_features)
         self.conv = nn.Conv2d(num_input_features, num_classes, kernel_size=1, bias=True)
         self._initialize()

     def _initialize(self):
         for m in self.modules():
             if isinstance(m, nn.Conv2d):
                 nn.init.kaiming_normal_(m.weight)
                 nn.init.constant_(m.bias, 0)

     def forward(self, x):
         x = self.bn(x)
         x = F.relu(x, inplace=True)
         x = self.conv(x)
         return x


class _UnetDecoder(nn.Module):
     def __init__(self, num_classes):
         super(_UnetDecoder, self).__init__()
         self.flat = DoubleConv(512, 256)

         self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
         self.up_conv1 = DoubleConv(256, 128)

         self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
         self.up_conv2 = DoubleConv(128, 64)

         self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
         self.up_conv3 = DoubleConv(64, 64)

         self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
         self.up_conv4 = DoubleConv(64, 64)

         self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
         self.up_conv5 = DoubleConv(64, 64)

         self.final_bnreluconv__ = _Final_BNReluConv(num_input_features=64, num_classes=num_classes)

     def forward(self, inputs, block1, block2, block3, block4):
         flat = self.flat(block4)
         up1 = self.up_conv1(self.up1(flat) + block3)
         up2 = self.up_conv2(self.up2(up1) + block2)
         up3 = self.up_conv3(self.up3(up2) + block1)
         up4 = self.up_conv4(self.up4(up3) + inputs)
         up5 = self.up_conv5(self.up5(up4))
         label_map = self.final_bnreluconv__(up5) 
         return label_map


class ResUnet(nn.Module):
     def __init__(self, num_classes, pretrained=True):
         super(ResUnet, self).__init__()
         self.num_classes = num_classes
         self.encoder = _ResNetEncoder(BasicBlock, [2, 2, 2, 2], pretrained=pretrained)
         self.decoder = _UnetDecoder(num_classes)
         self.sigmoid = nn.Sigmoid()

         self.conv =  nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        )
         self.linear = nn.Sequential(nn.Linear(128, 64),
            nn.Linear(64, 3))

     def forward(self, anchor):
         inputs, block1, block2, block3, block4 = self.encoder(anchor)

         label_map = self.decoder(inputs=inputs, block1=block1, block2=block2, block3=block3, block4=block4)

         return label_map


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
