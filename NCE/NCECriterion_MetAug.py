import torch
from torch import nn
import config

eps = 1e-7


class NCECriterion(nn.Module):

    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        '''
        bsz = x.shape[0]
        m = x.size(1) - 1
        # noise distribution
        Pn = 1 / float(self.n_data)
        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        a = P_neg.clone().fill_(m * Pn)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz
        '''

        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)

        return loss


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class CircleLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self,args):
        super(CircleLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.soft_plus = nn.Softplus()
        self.delta_m = args.delta_m_gh
        self.alpha_m = args.alpha_m_gh
        self.alpha_de = args.alpha_de_gh
        self.alpha_enable = args.alpha_enable_gh
        self.delta_enable = args.delta_enable_gh
        self.gamma = args.gama_gh
        self.counter = 0

    def circle_loss_ite(self, sp, sn):
        ap = torch.clamp(- sp.detach() + 1 + self.m, max=1000, min=0.)
        an = torch.clamp(sn.detach() + self.m, max=1000, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p + eps) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))

        return loss

    def circle_loss_ite_de_to_softmax(self, sp, sn):
        if self.alpha_enable == 'yes':
            ap = torch.clamp(1/self.alpha_de - sp.detach().div(self.alpha_de)+ self.alpha_m, max=1000, min=0.)
            an = torch.clamp(sn.detach().div(self.alpha_de) + self.alpha_m, max=1000, min=0.)
            ap = 1
            an = 1
        elif self.alpha_enable == 'no':
            ap = 1
            an = 1
        else:
            raise ValueError('alpha enable must be yes or no,but get :{}'.format(self.alpha_enable))

        if self.delta_enable == 'yes':
            delta_p = 1 - self.delta_m
            delta_n = self.delta_m
        elif self.delta_enable == 'no':
            delta_p = 0
            delta_n = 0
        else:
            raise ValueError('alpha enable must be yes or no,but get :{}'.format(self.alpha_enable))

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)).sum().div(sp.shape[0])

        loss = loss.div(self.gamma).mul(16)

        return loss

    def calculate_original_softmax(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss

    def forward(self, out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab):
        pos_ab2l,neg_ab2l = torch.split(out_ab2l,[1,out_ab2l.shape[1]-1],dim=1)
        pos_l2ab,neg_l2ab = torch.split(out_l2ab,[1,out_l2ab.shape[1]-1],dim=1)
        pos_ori2l,neg_ori2l = torch.split(out_ori2l,[1,out_ori2l.shape[1]-1],dim=1)
        pos_l2ori,neg_l2ori = torch.split(out_l2ori,[1,out_l2ori.shape[1]-1],dim=1)
        pos_ab2ori,neg_ab2ori = torch.split(out_ab2ori,[1,out_ab2ori.shape[1]-1],dim=1)
        pos_ori2ab,neg_ori2ab = torch.split(out_ori2ab,[1,out_ori2ab.shape[1]-1],dim=1)

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

        if config.SHOW_OUT_LAB:
            if self.counter > 30:
                out_ab2l_print_n = out_ab2l[:, torch.arange(out_ab2l.size(1)) != 0].detach()
                print("out_ab2l minus is ", out_ab2l[0][0].sum().detach() - out_ab2l_print_n[0].sum()/(out_ab2l.shape[1]-1))
                self.counter = 0
            else:
                self.counter = self.counter + 1

        loss = self.circle_loss_ite_de_to_softmax(pos, neg).sum()

        return loss
