import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np


def normal_wrapper(normal_method, in_ch, in_ch_div=2):
    if normal_method == "bn":
        return nn.BatchNorm3d(in_ch)
    elif normal_method == "bnt":
        # this should be used when batch_size=1
        return nn.BatchNorm3d(in_ch, affine=True, track_running_stats=False)
    elif normal_method == "bntna":
        # this should be used when batch_size=1
        return nn.BatchNorm3d(in_ch, affine=False, track_running_stats=False)
    elif normal_method == "ln":
        return nn.GroupNorm(1, in_ch)
    elif normal_method == "lnna":
        return nn.GroupNorm(1, in_ch, affine=False)
    elif normal_method == "in":
        return nn.GroupNorm(in_ch, in_ch)
    elif normal_method == "sbn":
        return nn.SyncBatchNorm(in_ch)
    else:
        raise NotImplementedError


def act_wrapper(act_method, num_parameters=1, init=0.25):
    if act_method == "relu":
        return nn.ReLU(inplace=True)
    elif act_method == "prelu":
        return nn.PReLU(num_parameters, init)
    else:
        raise NotImplementedError

def crop_concat_5d(t1, t2):
    """"Channel-wise cropping for 5-d tensors in NCDHW format,
    assuming t1 is smaller than t2 in all DHW dimension. """
    assert (t1.dim() == t2.dim() == 5)
    assert (t1.shape[-1] <= t2.shape[-1])
    slices = (slice(None, None), slice(None, None)) \
             + tuple(
        [slice(int(np.ceil((b - a) / 2)), a + int(np.ceil((b - a) / 2))) for a, b in zip(t1.shape[2:], t2.shape[2:])])
    x = torch.cat([t1, t2[slices]], dim=1)
    return x

class UpsampleConvBlock5d(nn.Module):

    def __init__(self, in_chs, base_chs, checkpoint_segments, scale_factor,
                 conv_ksize, conv_bias, conv_pad, dropout=0.1,
                 norm_method='bn', act_methpd='relu', **kwargs):
        super(UpsampleConvBlock5d, self).__init__()
        self.checkpoint_segments = checkpoint_segments
        self.scale_factor = scale_factor
        if not isinstance(conv_ksize, (tuple, list)):
            conv_ksize = [conv_ksize] * len(in_chs)

        if not isinstance(conv_pad, (tuple, list)):
            conv_pad = [conv_pad] * len(in_chs)

        if dropout > 0:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx], bias=conv_bias),
                    normal_wrapper(norm_method, base_ch),
                    act_wrapper(act_methpd),
                    nn.Dropout(dropout)
                ) for idx, (in_ch, base_ch) in enumerate(zip(in_chs, base_chs))
            ])
        else:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx], bias=conv_bias),
                    normal_wrapper(norm_method, base_ch),
                    act_wrapper(act_methpd),
                ) for idx, (in_ch, base_ch) in enumerate(zip(in_chs, base_chs))
            ])

        self.merge_func = kwargs.get('merge_func', crop_concat_5d)
        self.upsample = nn.Upsample(size=None, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)

    def forward(self, inputs, cats, args=None):
        up_inputs = self.upsample(inputs)
        x = crop_concat_5d(up_inputs, cats)
        x = self.conv_blocks(x)
        return x

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    zero_pads = zero_pads.to(x.device)

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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


class ResNetSegCls(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='A',
                 n_classes=[6, 3]):
        self.inplanes = 64
        super(ResNetSegCls, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.n_classes = n_classes
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.us1 = UpsampleConvBlock5d([512 * block.expansion + 64, 64],
                                       [64, 64],
                                       1, (2, 2, 2),
                                       (3, 3), True, (1, 1),
                                       norm_method="bn", act_method="relu", dropout=0.0)
        self.us2 = UpsampleConvBlock5d([64 + 64, 64],
                                       [64, 64],
                                       1, 2,
                                       (3, 3), True, (1, 1),
                                       norm_method="bn", act_method="relu", dropout=0.0)
        self.us3 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([nn.Conv3d(
            32,
            n_class, kernel_size=1, padding=0, stride=1,
            bias=True) for n_class in self.n_classes])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def get_target_layer(self):
        return self.us3

    def forward(self, x, lungs=None):
        B = x.shape[0]
        x = self.conv1(x)  # /2
        x = self.bn1(x)
        x = self.relu(x)
        xp = self.maxpool(x)  # /2
        x1 = self.layer1(xp)  # /1
        x2 = self.layer2(x1)  # /2
        x3 = self.layer3(x2)  # /1
        x4 = self.layer4(x3)  # /1
        xup1 = self.us1(x4, x1)
        xup2 = self.us2(xup1, x)
        xup3 = self.us3(xup2)
        dense_outs = [fc(xup3) for fc in self.fcs]
        cls_outs = [F.adaptive_avg_pool3d(dout, 1).view(B, -1) for dout in dense_outs]
        return dense_outs, cls_outs


class ResNetSegReg(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='A'):
        self.inplanes = 64
        super(ResNetSegReg, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.us1 = UpsampleConvBlock5d([512 * block.expansion + 64, 64],
                                       [64, 64],
                                       1, (2, 2, 2),
                                       (3, 3), True, (1, 1),
                                       norm_method="bn", act_method="relu", dropout=0.0)
        self.us2 = UpsampleConvBlock5d([64 + 64, 64],
                                       [64, 64],
                                       1, 2,
                                       (3, 3), True, (1, 1),
                                       norm_method="bn", act_method="relu", dropout=0.0)
        self.us3 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([nn.Conv3d(
            32,
            n_class, kernel_size=1, padding=0, stride=1,
            bias=True) for n_class in [1, 1]])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def get_target_layer(self):
        return self.us3

    def forward(self, x, lungs=None):
        B = x.shape[0]
        x = self.conv1(x)  # /2
        x = self.bn1(x)
        x = self.relu(x)
        xp = self.maxpool(x)  # /2
        x1 = self.layer1(xp)  # /1
        x2 = self.layer2(x1)  # /2
        x3 = self.layer3(x2)  # /1
        x4 = self.layer4(x3)  # /1
        xup1 = self.us1(x4, x1)
        xup2 = self.us2(xup1, x)
        xup3 = self.us3(xup2)
        dense_outs = [torch.sigmoid(fc(xup3)) for fc in self.fcs]
        lungs = F.interpolate(lungs, xup3.shape[-3:], mode='nearest')
        reg_outs = [(dout * lungs).view(B, -1).sum(dim=-1) / lungs.view(B, -1).sum(dim=-1) for dout in dense_outs]
        return dense_outs, reg_outs


def resnet34segcls(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNetSegCls(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet34segreg(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNetSegReg(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 n_classes,
                 shortcut_type='A'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.n_classes = n_classes
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.fc = nn.Conv3d(512, self.n_classes, kernel_size=1, padding=0, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        dense_predictions = self.fc(x)
        x = F.adaptive_avg_pool3d(dense_predictions, 1).view(B, -1)
        return x, dense_predictions


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
