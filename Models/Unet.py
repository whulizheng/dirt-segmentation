import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import nn


class Model(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Model, self).__init__()
        self.name = "U-Net"
        self.in_channel, self.height, self.width = input_shape
        self.out_channel = num_classes
        self.down1 = StackEncoder(self.in_channel, 12, kernel_size=(3, 3))
        self.down2 = StackEncoder(12, 24, kernel_size=(3, 3))
        self.down3 = StackEncoder(24, 46, kernel_size=(3, 3))
        self.down4 = StackEncoder(46, 64, kernel_size=(3, 3))
        self.down5 = StackEncoder(64, 128, kernel_size=(3, 3))

        self.center = ConvBlock(128, 128, kernel_size=(3, 3), padding=1)

        self.up5 = StackDecoder(128, 128, 64, kernel_size=(3, 3))
        self.up4 = StackDecoder(64, 64, 46, kernel_size=(3, 3))
        self.up3 = StackDecoder(46, 46, 24, kernel_size=(3, 3))
        self.up2 = StackDecoder(24, 24, 12, kernel_size=(3, 3))
        self.up1 = StackDecoder(12, 12, 12, kernel_size=(3, 3))
        self.conv = OutConv(12, self.out_channel)

    def forward(self, x):
        down1, out = self.down1(x)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)

        out = self.center(out)

        up5 = self.up5(out, down5)
        up4 = self.up4(up5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)

        out = self.conv(up1)

        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


class StackEncoder(nn.Module):
    def __init__(self, channel1, channel2, kernel_size=(3, 3), padding=1):
        super(StackEncoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ConvBlock(channel1, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
        )

    def forward(self, x):
        big_out = self.block(x)
        poolout = self.maxpool(big_out)
        return big_out, poolout


class StackDecoder(nn.Module):
    def __init__(self, big_channel, channel1, channel2, kernel_size=(3, 3), padding=1):
        super(StackDecoder, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channel1+big_channel, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.upsample(x, size=(height, width), mode='bilinear')
        # combining channels of  input from encoder and upsampling input
        x = torch.cat([x, down_tensor], 1)
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


def dice_coef_metric(pred, label):
    intersection = 2.0 * (pred * label).sum()
    union = pred.sum() + label.sum()
    if pred.sum() == 0 and label.sum() == 0:
        return 1.
    return intersection / union


def dice_coef_loss(pred, label):
    smooth = 1.0
    intersection = 2.0 * (pred * label).sum() + smooth
    union = pred.sum() + label.sum() + smooth
    return 1 - (intersection / union)


class Loss(nn.Module):
    def __init__(self, num_classes, activation=None, reduction='mean'):
        super(Loss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(1, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice

def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss =  (2 * intersection + eps) / (unionset + eps)

    return loss.sum() / N


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N
