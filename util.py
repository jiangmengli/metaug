from __future__ import print_function

import torch
import numpy as np


margin_hyper = 1e-4

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def margin_injection_loss_calc(mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori,
                               mtab_out_ab2l, mtab_out_l2ab, mtab_out_ab2ori, mtab_out_ori2ab,
                               mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab,
                               out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab,
                               margin_type, margin_loss_type):
    # split pos neg
    mtl_pos_ab2l, mtl_neg_ab2l = torch.split(mtl_out_ab2l, [1, mtl_out_ab2l.shape[1]-1], dim=1)
    mtl_pos_l2ab, mtl_neg_l2ab = torch.split(mtl_out_l2ab, [1, mtl_out_l2ab.shape[1]-1], dim=1)
    mtl_pos_ori2l, mtl_neg_ori2l = torch.split(mtl_out_ori2l, [1, mtl_out_ori2l.shape[1]-1], dim=1)
    mtl_pos_l2ori, mtl_neg_l2ori = torch.split(mtl_out_l2ori, [1, mtl_out_l2ori.shape[1]-1], dim=1)

    mtab_pos_ab2l, mtab_neg_ab2l = torch.split(mtab_out_ab2l, [1, mtab_out_ab2l.shape[1]-1], dim=1)
    mtab_pos_l2ab, mtab_neg_l2ab = torch.split(mtab_out_l2ab, [1, mtab_out_l2ab.shape[1]-1], dim=1)
    mtab_pos_ab2ori, mtab_neg_ab2ori = torch.split(mtab_out_ab2ori, [1, mtab_out_ab2ori.shape[1]-1], dim=1)
    mtab_pos_ori2ab, mtab_neg_ori2ab = torch.split(mtab_out_ori2ab, [1, mtab_out_ori2ab.shape[1]-1], dim=1)

    mtori_pos_ori2l, mtori_neg_ori2l = torch.split(mtori_out_ori2l, [1, mtori_out_ori2l.shape[1]-1], dim=1)
    mtori_pos_l2ori, mtori_neg_l2ori = torch.split(mtori_out_l2ori, [1, mtori_out_l2ori.shape[1]-1], dim=1)
    mtori_pos_ab2ori, mtori_neg_ab2ori = torch.split(mtori_out_ab2ori, [1, mtori_out_ab2ori.shape[1]-1], dim=1)
    mtori_pos_ori2ab, mtori_neg_ori2ab = torch.split(mtori_out_ori2ab, [1, mtori_out_ori2ab.shape[1]-1], dim=1)

    pos_ab2l, neg_ab2l = torch.split(out_ab2l, [1, out_ab2l.shape[1]-1], dim=1)
    pos_l2ab, neg_l2ab = torch.split(out_l2ab, [1, out_l2ab.shape[1]-1], dim=1)
    pos_ori2l, neg_ori2l = torch.split(out_ori2l, [1, out_ori2l.shape[1]-1], dim=1)
    pos_l2ori, neg_l2ori = torch.split(out_l2ori, [1, out_l2ori.shape[1]-1], dim=1)
    pos_ab2ori, neg_ab2ori = torch.split(out_ab2ori, [1, out_ab2ori.shape[1]-1], dim=1)
    pos_ori2ab, neg_ori2ab = torch.split(out_ori2ab, [1, out_ori2ab.shape[1]-1], dim=1)

    # post-process
    mtl_pos_ab2l = mtl_pos_ab2l.squeeze(2).add(1).div(2)
    mtl_neg_ab2l = mtl_neg_ab2l.squeeze(2).add(1).div(2)
    mtl_pos_l2ab = mtl_pos_l2ab.squeeze(2).add(1).div(2)
    mtl_neg_l2ab = mtl_neg_l2ab.squeeze(2).add(1).div(2)
    mtl_pos_ori2l = mtl_pos_ori2l.squeeze(2).add(1).div(2)
    mtl_neg_ori2l = mtl_neg_ori2l.squeeze(2).add(1).div(2)
    mtl_pos_l2ori = mtl_pos_l2ori.squeeze(2).add(1).div(2)
    mtl_neg_l2ori = mtl_neg_l2ori.squeeze(2).add(1).div(2)

    mtab_pos_ab2l = mtab_pos_ab2l.squeeze(2).add(1).div(2)
    mtab_neg_ab2l = mtab_neg_ab2l.squeeze(2).add(1).div(2)
    mtab_pos_l2ab = mtab_pos_l2ab.squeeze(2).add(1).div(2)
    mtab_neg_l2ab = mtab_neg_l2ab.squeeze(2).add(1).div(2)
    mtab_pos_ab2ori = mtab_pos_ab2ori.squeeze(2).add(1).div(2)
    mtab_neg_ab2ori = mtab_neg_ab2ori.squeeze(2).add(1).div(2)
    mtab_pos_ori2ab = mtab_pos_ori2ab.squeeze(2).add(1).div(2)
    mtab_neg_ori2ab = mtab_neg_ori2ab.squeeze(2).add(1).div(2)

    mtori_pos_ori2l = mtori_pos_ori2l.squeeze(2).add(1).div(2)
    mtori_neg_ori2l = mtori_neg_ori2l.squeeze(2).add(1).div(2)
    mtori_pos_l2ori = mtori_pos_l2ori.squeeze(2).add(1).div(2)
    mtori_neg_l2ori = mtori_neg_l2ori.squeeze(2).add(1).div(2)
    mtori_pos_ab2ori = mtori_pos_ab2ori.squeeze(2).add(1).div(2)
    mtori_neg_ab2ori = mtori_neg_ab2ori.squeeze(2).add(1).div(2)
    mtori_pos_ori2ab = mtori_pos_ori2ab.squeeze(2).add(1).div(2)
    mtori_neg_ori2ab = mtori_neg_ori2ab.squeeze(2).add(1).div(2)
    mt_pos = torch.cat((mtl_pos_ab2l, mtl_pos_l2ab, mtl_pos_ori2l, mtl_pos_l2ori, mtab_pos_ab2l, mtab_pos_l2ab,
                        mtab_pos_ab2ori, mtab_pos_ori2ab, mtori_pos_ori2l, mtori_pos_l2ori, mtori_pos_ab2ori,
                        mtori_pos_ori2ab), dim=1)
    mt_neg = torch.cat((mtl_neg_ab2l, mtl_neg_l2ab, mtl_neg_ori2l, mtl_neg_l2ori, mtab_neg_ab2l, mtab_neg_l2ab,
                        mtab_neg_ab2ori, mtab_neg_ori2ab, mtori_neg_ori2l, mtori_neg_l2ori, mtori_neg_ab2ori,
                        mtori_neg_ori2ab), dim=1)

    pos_ab2l = pos_ab2l.squeeze(2).add(1).div(2)
    neg_ab2l = neg_ab2l.squeeze(2).add(1).div(2)
    pos_l2ab = pos_l2ab.squeeze(2).add(1).div(2)
    neg_l2ab = neg_l2ab.squeeze(2).add(1).div(2)
    pos_ori2l = pos_ori2l.squeeze(2).add(1).div(2)
    neg_ori2l = neg_ori2l.squeeze(2).add(1).div(2)
    pos_l2ori = pos_l2ori.squeeze(2).add(1).div(2)
    neg_l2ori = neg_l2ori.squeeze(2).add(1).div(2)
    pos_ab2ori = pos_ab2ori.squeeze(2).add(1).div(2)
    neg_ab2ori = neg_ab2ori.squeeze(2).add(1).div(2)
    pos_ori2ab = pos_ori2ab.squeeze(2).add(1).div(2)
    neg_ori2ab = neg_ori2ab.squeeze(2).add(1).div(2)
    pos = torch.cat((pos_ab2l, pos_l2ab, pos_ori2l, pos_l2ori, pos_ab2ori, pos_ori2ab), dim=1)
    neg = torch.cat((neg_ab2l, neg_l2ab, neg_ori2l, neg_l2ori, neg_ab2ori, neg_ori2ab), dim=1)

    # get max pos min neg
    min_positive_value, min_positive_pos = torch.min(pos, dim=-1)
    max_negative_value, max_negative_pos = torch.max(neg, dim=-1)

    # get large gamma (margin)
    if margin_type == 'small':
        lgamma_margin_pos, _ = torch.max(torch.cat((min_positive_value.unsqueeze(1), max_negative_value.unsqueeze(1)), dim=1), dim=-1)
        lgamma_margin_pos = lgamma_margin_pos.unsqueeze(1)
        lgamma_margin_neg, _ = torch.min(torch.cat((min_positive_value.unsqueeze(1), max_negative_value.unsqueeze(1)), dim=1), dim=-1)
        lgamma_margin_neg = lgamma_margin_neg.unsqueeze(1)
    elif margin_type == 'large':
        lgamma_margin_pos, _ = torch.min(torch.cat((min_positive_value.unsqueeze(1), max_negative_value.unsqueeze(1)), dim=1), dim=-1)
        lgamma_margin_pos = lgamma_margin_pos.unsqueeze(1)
        lgamma_margin_neg, _ = torch.max(torch.cat((min_positive_value.unsqueeze(1), max_negative_value.unsqueeze(1)), dim=1), dim=-1)
        lgamma_margin_neg = lgamma_margin_neg.unsqueeze(1)
    else:
        lgamma_margin_pos = torch.mean(torch.cat((min_positive_value.unsqueeze(1), max_negative_value.unsqueeze(1)), dim=1), dim=1, keepdim=True)
        lgamma_margin_neg = lgamma_margin_pos

    # get margin injection loss
    loss = torch.mean(torch.clamp(mt_pos - lgamma_margin_pos, min=0)) * margin_hyper
    loss += torch.mean(torch.clamp(lgamma_margin_neg - mt_neg, min=0))

    return loss


if __name__ == '__main__':
    meter = AverageMeter()
