from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyAlexNetCMC(nn.Module):
    def __init__(self, device, feat_dim=128):
        super(MyAlexNetCMC, self).__init__()
        self.encoder = alexnet(device, feat_dim=feat_dim)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, fast_weights, layer=8):
        return self.encoder(x, fast_weights, layer)


class alexnet(nn.Module):
    def __init__(self, device, feat_dim=128):
        super(alexnet, self).__init__()

        self.l_to_ab = alexnet_half(device, in_channel=1, feat_dim=feat_dim)
        self.ab_to_l = alexnet_half(device, in_channel=2, feat_dim=feat_dim)
        self.ori = alexnet_half(device, in_channel=3, feat_dim=feat_dim)
        a=1
        pass


    def forward(self, x, fast_weights, layer=8):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, fast_weights, layer, 'l_to_ab')
        feat_ab = self.ab_to_l(ab, fast_weights, layer, 'ab_to_l')
        feat_ori = self.ori(x, fast_weights, layer, 'ori')

        return feat_l, feat_ab, feat_ori


class alexnet_half(nn.Module):
    def __init__(self, device, in_channel=1, feat_dim=128):
        super(alexnet_half, self).__init__()
        self.device = device
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96 // 2, 11, 4, 2, bias=False),
            nn.BatchNorm2d(96 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(96 // 2, 256 // 2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(256 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256 // 2, 384 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384 // 2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(384 // 2, 384 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384 // 2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(384 // 2, 256 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096 // 2, feat_dim)
        )
        self.l2norm = Normalize(2)

    def forward(self, x, weights, layer, feat_type, index=0):
        if weights:
            if layer <= 0:
                return x
            # conv1
            index += 1
            x = F.conv2d(x, weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.0.weight'.format(index)],
                         stride=4, padding=2)
            x = F.batch_norm(x, torch.zeros(x.data.size()[1]).to(self.device),
                             torch.ones(x.data.size()[1]).to(self.device),
                             weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.1.weight'.format(index)],
                             weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.1.bias'.format(index)],
                             training=True)
            x = F.relu(x, inplace=True)
            x = F.max_pool2d(x, kernel_size=3, stride=2, )
            if layer == 1:
                return x
            # conv2
            index += 1
            x = F.conv2d(x, weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.0.weight'.format(index)],
                         stride=1, padding=2)
            x = F.batch_norm(x, torch.zeros(x.data.size()[1]).to(self.device),
                             torch.ones(x.data.size()[1]).to(self.device),
                             weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.1.weight'.format(index)],
                             weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.1.bias'.format(index)],
                             training=True)
            x = F.relu(x, inplace=True)
            x = F.max_pool2d(x, kernel_size=3, stride=2, )
            if layer == 2:
                return x
            # conv3
            index += 1
            x = F.conv2d(x, weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.0.weight'.format(index)],
                         stride=1, padding=1)
            x = F.batch_norm(x, torch.zeros(x.data.size()[1]).to(self.device),
                             torch.ones(x.data.size()[1]).to(self.device),
                             weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.1.weight'.format(index)],
                             weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.1.bias'.format(index)],
                             training=True)
            x = F.relu(x, inplace=True)
            if layer == 3:
                return x
            # conv4
            index += 1
            x = F.conv2d(x, weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.0.weight'.format(index)],
                         stride=1, padding=1)
            x = F.batch_norm(x, torch.zeros(x.data.size()[1]).to(self.device),
                             torch.ones(x.data.size()[1]).to(self.device),
                             weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.1.weight'.format(index)],
                             weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.1.bias'.format(index)],
                             training=True)
            x = F.relu(x, inplace=True)
            if layer == 4:
                return x
            # conv5
            index += 1
            x = F.conv2d(x, weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.0.weight'.format(index)],
                         stride=1, padding=1)
            x = F.batch_norm(x, torch.zeros(x.data.size()[1]).to(self.device),
                             torch.ones(x.data.size()[1]).to(self.device),
                             weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.1.weight'.format(index)],
                             weights['encoder.module.' + str(feat_type) + '.conv_block_{:d}.1.bias'.format(index)],
                             training=True)
            x = F.relu(x, inplace=True)
            x = F.max_pool2d(x, kernel_size=3, stride=2, )
            if layer == 5:
                return x
            # fc6
            x = x.view(x.shape[0], -1)
            index += 1
            x = F.linear(x, weights['encoder.module.' + str(feat_type) + '.fc{:d}.0.weight'.format(index)],
                         weights['encoder.module.' + str(feat_type) + '.fc{:d}.0.bias'.format(index)])
            x = F.batch_norm(x, torch.zeros(x.data.size()[1]).to(self.device),
                             torch.ones(x.data.size()[1]).to(self.device),
                             weights['encoder.module.' + str(feat_type) + '.fc{:d}.1.weight'.format(index)],
                             weights['encoder.module.' + str(feat_type) + '.fc{:d}.1.bias'.format(index)],
                             training=True)
            x = F.relu(x, inplace=True)
            if layer == 6:
                return x
            # fc7
            index += 1
            x = F.linear(x, weights['encoder.module.' + str(feat_type) + '.fc{:d}.0.weight'.format(index)],
                         weights['encoder.module.' + str(feat_type) + '.fc{:d}.0.bias'.format(index)])
            x = F.batch_norm(x, torch.zeros(x.data.size()[1]).to(self.device),
                             torch.ones(x.data.size()[1]).to(self.device),
                             weights['encoder.module.' + str(feat_type) + '.fc{:d}.1.weight'.format(index)],
                             weights['encoder.module.' + str(feat_type) + '.fc{:d}.1.bias'.format(index)],
                             training=True)
            x = F.relu(x, inplace=True)
            if layer == 7:
                return x
            # fc8
            index += 1
            x = F.linear(x, weights['encoder.module.' + str(feat_type) + '.fc{:d}.0.weight'.format(index)],
                         weights['encoder.module.' + str(feat_type) + '.fc{:d}.0.bias'.format(index)])
            x = F.normalize(x)
        else:
            if layer <= 0:
                return x
            x = self.conv_block_1(x)
            if layer == 1:
                return x
            x = self.conv_block_2(x)
            if layer == 2:
                return x
            x = self.conv_block_3(x)
            if layer == 3:
                return x
            x = self.conv_block_4(x)
            if layer == 4:
                return x
            x = self.conv_block_5(x)
            if layer == 5:
                return x
            x = x.view(x.shape[0], -1)
            x = self.fc6(x)
            if layer == 6:
                return x
            x = self.fc7(x)
            if layer == 7:
                return x
            x = self.fc8(x)
            x = self.l2norm(x)
        return x


class MyMetaGenNet(nn.Module):
    def __init__(self, device, feat_dim=128):
        super(MyMetaGenNet, self).__init__()
        self.encoder = metagen_net(device, feat_dim=feat_dim)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, fast_weights, layer=3):
        return self.encoder(x, fast_weights, layer)


class metagen_net(nn.Module):
    def __init__(self, device, feat_dim=128):
        super(metagen_net, self).__init__()
        self.device = device
        self.feat_dim = feat_dim
        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096 // 2, feat_dim)
        )
        self.l2norm = Normalize(2)

    def forward(self, x, weights, layer=3, index=0):
        if not x.shape[1] == self.feat_dim:
            feat_convert_fc = nn.Sequential(
                nn.Linear(x.shape[1], self.feat_dim)
            ).to(self.device)
            x = feat_convert_fc(x)
        if weights:
            # fc1
            x = x.view(x.shape[0], -1)
            index += 1
            x = F.linear(x, weights['encoder.module.fc{:d}.0.weight'.format(index)],
                         weights['encoder.module.fc{:d}.0.bias'.format(index)])
            x = F.batch_norm(x, torch.zeros(x.data.size()[1]).to(self.device),
                             torch.ones(x.data.size()[1]).to(self.device),
                             weights['encoder.module.fc{:d}.1.weight'.format(index)],
                             weights['encoder.module.fc{:d}.1.bias'.format(index)],
                             training=True)
            x = F.relu(x, inplace=True)
            if layer == 1:
                return x
            # fc2
            index += 1
            x = F.linear(x, weights['encoder.module.fc{:d}.0.weight'.format(index)],
                         weights['encoder.module.fc{:d}.0.bias'.format(index)])
            x = F.batch_norm(x, torch.zeros(x.data.size()[1]).to(self.device),
                             torch.ones(x.data.size()[1]).to(self.device),
                             weights['encoder.module.fc{:d}.1.weight'.format(index)],
                             weights['encoder.module.fc{:d}.1.bias'.format(index)],
                             training=True)
            x = F.relu(x, inplace=True)
            if layer == 2:
                return x
            # fc3
            index += 1
            x = F.linear(x, weights['encoder.module.fc{:d}.0.weight'.format(index)],
                         weights['encoder.module.fc{:d}.0.bias'.format(index)])
            x = F.normalize(x)
            return x
        else:
            x = self.fc1(x)
            if layer == 1:
                return x
            x = self.fc2(x)
            if layer == 2:
                return x
            x = self.fc3(x)
            x = self.l2norm(x)
            return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


if __name__ == '__main__':

    import torch

    model = alexnet().cuda()
    data = torch.rand(10, 3, 224, 224).cuda()
    out = model.compute_feat(data, 5)

    for i in range(10):
        out = model.compute_feat(data, i)
        print(i, out.shape)
