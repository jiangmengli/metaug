import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ori', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, ori, y, idx=None, updatemem=True):
        K = int(self.params[0].item())

        momentum = self.params[4].item()
        batchSize = l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab2l = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))

        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l2ab = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))
        # sample
        weight_ori = torch.index_select(self.memory_ori, 0, idx.view(-1)).detach()
        weight_ori = weight_ori.view(batchSize, K + 1, inputSize)
        out_l2ori = torch.bmm(weight_ori, l.view(batchSize, inputSize, 1))

        # other
        out_ori2l = torch.bmm(weight_l, ori.view(batchSize, inputSize, 1))
        out_ori2ab = torch.bmm(weight_ab, ori.view(batchSize, inputSize, 1))
        out_ab2ori = torch.bmm(weight_ori, ab.view(batchSize, inputSize, 1))
        out_ab2l = out_ab2l.contiguous()
        out_l2ab = out_l2ab.contiguous()
        out_ori2l = out_ori2l.contiguous()
        out_l2ori = out_l2ori.contiguous()
        out_ab2ori = out_ab2ori.contiguous()
        out_ori2ab = out_ori2ab.contiguous()

        # update memory
        if updatemem:
            with torch.no_grad():
                l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
                l_pos.mul_(momentum)
                l_pos.add_(torch.mul(l, 1 - momentum))
                l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_l = l_pos.div(l_norm)
                self.memory_l.index_copy_(0, y, updated_l)

                ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
                ab_pos.mul_(momentum)
                ab_pos.add_(torch.mul(ab, 1 - momentum))
                ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_ab = ab_pos.div(ab_norm)
                self.memory_ab.index_copy_(0, y, updated_ab)

                ori_pos = torch.index_select(self.memory_ori, 0, y.view(-1))
                ori_pos.mul_(momentum)
                ori_pos.add_(torch.mul(ori, 1 - momentum))
                ori_norm = ori_pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_ori = ori_pos.div(ori_norm)
                self.memory_ori.index_copy_(0, y, updated_ori)

        return out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab
