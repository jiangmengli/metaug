"""
Train MetAug with AlexNet
"""
from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import config

import tensorboard_logger as tb_logger
from torchvision import transforms
from dataset import RGB2Lab, RGB2YCbCr
from collections import OrderedDict

from util import adjust_learning_rate, AverageMeter, margin_injection_loss_calc

from models.alexnet_MetAug import MyAlexNetCMC
from models.alexnet_MetAug import MyMetaGenNet
from NCE.NCEAverage_MetAug import NCEAverage
from NCE.NCECriterion_MetAug import CircleLoss

from dataset import ImageFolderInstance

try:
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--model_version', type=str, default='MetAug_1e-5_1e-11', choices=['cmvc', 'meta_aug', 'MetAug'])

    parser.add_argument('--print_freq', type=int, default=30, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE_DEFAULT, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # metagenaug optimization
    parser.add_argument('--mt_learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--mt_weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--mt_momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--mt_loss_hp', type=float, default=1e-5, help='hyper-param for metaaug losses')

    parser.add_argument('--bb_learning_rate', type=float, default=0.001, help='bb_learning_rate')

    # margin
    parser.add_argument('--margin_injection', type=int, default=1, help='the flag of margin injection'
                                                                        '1: inject margin when train main_model and metaauggen'
                                                                        '2: inject margin only when train metaauggen'
                                                                        '0: NOT inject margin')
    parser.add_argument('--margin_type', type=str, default='large', help='the selection type of the margin'
                                                                       'small: the max of possim max and negsim min'
                                                                       'large: the min of possim max and negsim min'
                                                                       'mean: the mean of possim max and negsim min')
    parser.add_argument('--mj_loss_hp', type=float, default=1e-11, help='hyper-param for margin injection losses')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3'])
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=config.DEFAULT_K)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar-10', choices=['cifar-10', 'cifar-100', 'stl-10',
                                                                       'tiny-imagenet', 'imagenet100', 'imagenet'])

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['Lab', 'YCbCr'])

    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    parser.add_argument('--mark_gh', type=str, default=None, help='mark')
    parser.add_argument('--alpha_enable_gh', type=str, default='no', help='')
    parser.add_argument('--delta_enable_gh', type=str, default='no', help='')
    parser.add_argument('--delta_m_gh', type=float, default=0.25)
    parser.add_argument('--alpha_m_gh', type=float, default=0.25)
    parser.add_argument('--alpha_de_gh', type=float, default=1)
    parser.add_argument('--gama_gh', type=float, default=16)

    # GPU setting
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop_low = 0.08

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'
    opt.model_name = 'memory_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}'.format(opt.model_version, opt.method, opt.nce_k,
                                                                       opt.model, opt.learning_rate, opt.weight_decay,
                                                                       opt.batch_size)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.data_folder = '../../datasets/' + opt.dataset + '/'

    return opt


def get_train_loader(args):
    """get the train loader"""
    data_folder = os.path.join(args.data_folder, 'train')

    if args.view == 'Lab':
        mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        color_transfer = RGB2Lab()
    elif args.view == 'YCbCr':
        mean = [116.151, 121.080, 132.342]
        std = [109.500, 111.855, 111.964]
        color_transfer = RGB2YCbCr()
    else:
        raise NotImplemented('view not implemented {}'.format(args.view))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        color_transfer,
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
    train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def set_model(args, n_data):
    try:
        _ = int(args.gpu)
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    except:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set the model
    if args.model == 'alexnet':
        model = MyAlexNetCMC(device, args.feat_dim).to(device)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax).to(device)
    criterion_gh = CircleLoss(args).to(device)

    l_mtgen = MyMetaGenNet(device).to(device)

    ab_mtgen = MyMetaGenNet(device).to(device)

    ori_mtgen = MyMetaGenNet(device).to(device)

    if torch.cuda.is_available():
        cudnn.benchmark = True

    return model, contrast, criterion_gh, l_mtgen, ab_mtgen, ori_mtgen


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, contrast, criterion_gh, optimizer, l_mtgen, l_mtgen_op, ab_mtgen, ab_mtgen_op, ori_mtgen, ori_mtgen_op, opt):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float()
        if torch.cuda.is_available():
            index = index.cuda()
            inputs = inputs.cuda()

        # Step 1: Fix metagen aug, and optimize main_model -------------------------------------------------------------
        for param in l_mtgen.parameters():
            param.requires_grad = False
        for param in ab_mtgen.parameters():
            param.requires_grad = False
        for param in ori_mtgen.parameters():
            param.requires_grad = False
        for param in model.parameters():
            param.requires_grad = True
        # ===================forward=====================
        optimizer.zero_grad()

        feat_l, feat_ab, feat_ori = model(inputs, False)

        mt_feat_l = l_mtgen(feat_l, False)
        mt_feat_ab = ab_mtgen(feat_ab, False)
        mt_feat_ori = ori_mtgen(feat_ori, False)

        # ori loss calc
        out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab = contrast(feat_l, feat_ab, feat_ori, index)

        loss = criterion_gh(out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab)

        # aug loss calc
        mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori, out_ab2ori, out_ori2ab = contrast(mt_feat_l, feat_ab, feat_ori, index,
                                                                                    updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori, out_ab2ori, out_ori2ab)
        mtab_out_ab2l, mtab_out_l2ab, out_ori2l, out_l2ori, mtab_out_ab2ori, mtab_out_ori2ab = contrast(feat_l, mt_feat_ab, feat_ori, index,
                                                                                    updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mtab_out_ab2l, mtab_out_l2ab, out_ori2l, out_l2ori, mtab_out_ab2ori, mtab_out_ori2ab)
        out_ab2l, out_l2ab, mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab = contrast(feat_l, feat_ab, mt_feat_ori, index,
                                                                                    updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(out_ab2l, out_l2ab, mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab)
        mt_out_ab2l, mt_out_l2ab, mt_out_ori2l, mt_out_l2ori, mt_out_ab2ori, mt_out_ori2ab = contrast(mt_feat_l, mt_feat_ab, mt_feat_ori, index,
                                                                                    updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mt_out_ab2l, mt_out_l2ab, mt_out_ori2l, mt_out_l2ori, mt_out_ab2ori, mt_out_ori2ab)

        # margin-injection
        if opt.margin_injection == 1:
            loss += opt.mj_loss_hp * margin_injection_loss_calc(mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori,
                                                                mtab_out_ab2l, mtab_out_l2ab, mtab_out_ab2ori, mtab_out_ori2ab,
                                                                mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab,
                                                                out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab,
                                                                opt.margin_type)

        # backward
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        losses.update(loss.item(), bsz)

        # Step 2: Fix main_model, and optimize metagen aug -------------------------------------------------------------
        for param in l_mtgen.parameters():
            param.requires_grad = True
        for param in ab_mtgen.parameters():
            param.requires_grad = True
        for param in ori_mtgen.parameters():
            param.requires_grad = True
        # =================meta forward==================
        l_mtgen_op.zero_grad()
        ab_mtgen_op.zero_grad()
        ori_mtgen_op.zero_grad()

        feat_l, feat_ab, feat_ori = model(inputs, False)

        mt_feat_l = l_mtgen(feat_l, False)
        mt_feat_ab = ab_mtgen(feat_ab, False)
        mt_feat_ori = ori_mtgen(feat_ori, False)

        # ori loss calc
        out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab = contrast(feat_l, feat_ab, feat_ori, index)

        loss = criterion_gh(out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab)

        # aug loss calc
        mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori, out_ab2ori, out_ori2ab = contrast(mt_feat_l, feat_ab, feat_ori, index,
                                                                                                    updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori, out_ab2ori, out_ori2ab)
        mtab_out_ab2l, mtab_out_l2ab, out_ori2l, out_l2ori, mtab_out_ab2ori, mtab_out_ori2ab = contrast(feat_l, mt_feat_ab, feat_ori, index,
                                                                                                        updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mtab_out_ab2l, mtab_out_l2ab, out_ori2l, out_l2ori, mtab_out_ab2ori, mtab_out_ori2ab)
        out_ab2l, out_l2ab, mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab = contrast(feat_l, feat_ab, mt_feat_ori, index,
                                                                                                            updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(out_ab2l, out_l2ab, mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab)
        mt_out_ab2l, mt_out_l2ab, mt_out_ori2l, mt_out_l2ori, mt_out_ab2ori, mt_out_ori2ab = contrast(mt_feat_l, mt_feat_ab, mt_feat_ori, index,
                                                                                                      updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mt_out_ab2l, mt_out_l2ab, mt_out_ori2l, mt_out_l2ori, mt_out_ab2ori, mt_out_ori2ab)

        # margin-injection
        if opt.margin_injection >= 1:
            loss += opt.mj_loss_hp * margin_injection_loss_calc(mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori,
                                                                mtab_out_ab2l, mtab_out_l2ab, mtab_out_ab2ori, mtab_out_ori2ab,
                                                                mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab,
                                                                out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab,
                                                                opt.margin_type)

        # meta feat generation
        feat_l, feat_ab, feat_ori = model(inputs, False)
        # get meta weights
        fast_weights = OrderedDict((name, param) for (name, param) in l_mtgen.named_parameters())
        # create_graph flag for computing second-derivative
        grads = torch.autograd.grad(loss, l_mtgen.parameters(), create_graph=True)
        data = [p.data for p in list(l_mtgen.parameters())]
        # compute theta_1^+ by applying sgd on multi-task loss
        fast_weights = OrderedDict((name, param - opt.bb_learning_rate * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))
        mt_feat_l = l_mtgen(feat_l, fast_weights)

        # get meta weights
        fast_weights = OrderedDict((name, param) for (name, param) in ab_mtgen.named_parameters())
        # create_graph flag for computing second-derivative
        grads = torch.autograd.grad(loss, ab_mtgen.parameters(), create_graph=True)
        data = [p.data for p in list(ab_mtgen.parameters())]
        # compute theta_1^+ by applying sgd on multi-task loss
        fast_weights = OrderedDict((name, param - opt.bb_learning_rate * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))
        mt_feat_ab = ab_mtgen(feat_ab, fast_weights)

        # get meta weights
        fast_weights = OrderedDict((name, param) for (name, param) in ori_mtgen.named_parameters())
        # create_graph flag for computing second-derivative
        grads = torch.autograd.grad(loss, ori_mtgen.parameters(), create_graph=True)
        data = [p.data for p in list(ori_mtgen.parameters())]
        # compute theta_1^+ by applying sgd on multi-task loss
        fast_weights = OrderedDict((name, param - opt.bb_learning_rate * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))
        mt_feat_ori = ori_mtgen(feat_ori, fast_weights)

        # ori loss calc
        out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab = contrast(feat_l, feat_ab, feat_ori, index)

        loss = criterion_gh(out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab)

        # aug loss calc
        mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori, out_ab2ori, out_ori2ab = contrast(mt_feat_l, feat_ab, feat_ori, index,
                                                                                                    updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori, out_ab2ori, out_ori2ab)
        mtab_out_ab2l, mtab_out_l2ab, out_ori2l, out_l2ori, mtab_out_ab2ori, mtab_out_ori2ab = contrast(feat_l, mt_feat_ab, feat_ori, index,
                                                                                                        updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mtab_out_ab2l, mtab_out_l2ab, out_ori2l, out_l2ori, mtab_out_ab2ori, mtab_out_ori2ab)
        out_ab2l, out_l2ab, mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab = contrast(feat_l, feat_ab, mt_feat_ori, index,
                                                                                                            updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(out_ab2l, out_l2ab, mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab)
        mt_out_ab2l, mt_out_l2ab, mt_out_ori2l, mt_out_l2ori, mt_out_ab2ori, mt_out_ori2ab = contrast(mt_feat_l, mt_feat_ab, mt_feat_ori, index,
                                                                                                      updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mt_out_ab2l, mt_out_l2ab, mt_out_ori2l, mt_out_l2ori, mt_out_ab2ori, mt_out_ori2ab)

        # margin-injection
        if opt.margin_injection >= 1:
            loss += opt.mj_loss_hp * margin_injection_loss_calc(mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori,
                                                                mtab_out_ab2l, mtab_out_l2ab, mtab_out_ab2ori, mtab_out_ori2ab,
                                                                mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab,
                                                                out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab,
                                                                opt.margin_type)

        # backward
        if opt.amp:
            with amp.scale_loss(loss, l_mtgen_op) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        l_mtgen_op.step()
        ab_mtgen_op.step()
        ori_mtgen_op.step()

        # Step 3: Fix metagen aug, and optimize main_model by using the updated meta feature aug -----------------------
        for param in l_mtgen.parameters():
            param.requires_grad = False
        for param in ab_mtgen.parameters():
            param.requires_grad = False
        for param in ori_mtgen.parameters():
            param.requires_grad = False
        # ===================forward=====================
        optimizer.zero_grad()

        feat_l, feat_ab, feat_ori = model(inputs, False)

        mt_feat_l = l_mtgen(feat_l, False)
        mt_feat_ab = ab_mtgen(feat_ab, False)
        mt_feat_ori = ori_mtgen(feat_ori, False)

        # ori loss calc
        out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab = contrast(feat_l, feat_ab, feat_ori, index)

        loss = criterion_gh(out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab)

        # aug loss calc
        mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori, out_ab2ori, out_ori2ab = contrast(mt_feat_l, feat_ab, feat_ori, index,
                                                                                                    updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori, out_ab2ori, out_ori2ab)
        mtab_out_ab2l, mtab_out_l2ab, out_ori2l, out_l2ori, mtab_out_ab2ori, mtab_out_ori2ab = contrast(feat_l, mt_feat_ab, feat_ori, index,
                                                                                                        updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mtab_out_ab2l, mtab_out_l2ab, out_ori2l, out_l2ori, mtab_out_ab2ori, mtab_out_ori2ab)
        out_ab2l, out_l2ab, mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab = contrast(feat_l, feat_ab, mt_feat_ori, index,
                                                                                                            updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(out_ab2l, out_l2ab, mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab)
        mt_out_ab2l, mt_out_l2ab, mt_out_ori2l, mt_out_l2ori, mt_out_ab2ori, mt_out_ori2ab = contrast(mt_feat_l, mt_feat_ab, mt_feat_ori, index,
                                                                                                      updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mt_out_ab2l, mt_out_l2ab, mt_out_ori2l, mt_out_l2ori, mt_out_ab2ori, mt_out_ori2ab)

        # margin-injection
        if opt.margin_injection == 1:
            loss += opt.mj_loss_hp * margin_injection_loss_calc(mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori,
                                                                mtab_out_ab2l, mtab_out_l2ab, mtab_out_ab2ori, mtab_out_ori2ab,
                                                                mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab,
                                                                out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab,
                                                                opt.margin_type)

        # get meta weights
        fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())
        # create_graph flag for computing second-derivative
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        data = [p.data for p in list(model.parameters())]
        # compute theta_1^+ by applying sgd on multi-task loss
        fast_weights = OrderedDict((name, param - opt.bb_learning_rate * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))

        # meta feat generation
        feat_l, feat_ab, feat_ori = model(inputs, fast_weights)

        mt_feat_l = l_mtgen(feat_l, False)
        mt_feat_ab = ab_mtgen(feat_ab, False)
        mt_feat_ori = ori_mtgen(feat_ori, False)

        # ori loss calc
        out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab = contrast(feat_l, feat_ab, feat_ori, index)

        loss = criterion_gh(out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab)

        # aug loss calc
        mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori, out_ab2ori, out_ori2ab = contrast(mt_feat_l, feat_ab, feat_ori, index,
                                                                                                    updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori, out_ab2ori, out_ori2ab)
        mtab_out_ab2l, mtab_out_l2ab, out_ori2l, out_l2ori, mtab_out_ab2ori, mtab_out_ori2ab = contrast(feat_l, mt_feat_ab, feat_ori, index,
                                                                                                        updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mtab_out_ab2l, mtab_out_l2ab, out_ori2l, out_l2ori, mtab_out_ab2ori, mtab_out_ori2ab)
        out_ab2l, out_l2ab, mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab = contrast(feat_l, feat_ab, mt_feat_ori, index,
                                                                                                            updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(out_ab2l, out_l2ab, mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab)
        mt_out_ab2l, mt_out_l2ab, mt_out_ori2l, mt_out_l2ori, mt_out_ab2ori, mt_out_ori2ab = contrast(mt_feat_l, mt_feat_ab, mt_feat_ori, index,
                                                                                                      updatemem=False)
        loss += opt.mt_loss_hp * criterion_gh(mt_out_ab2l, mt_out_l2ab, mt_out_ori2l, mt_out_l2ori, mt_out_ab2ori, mt_out_ori2ab)

        # margin-injection
        if opt.margin_injection == 1:
            loss += opt.mj_loss_hp * margin_injection_loss_calc(mtl_out_ab2l, mtl_out_l2ab, mtl_out_ori2l, mtl_out_l2ori,
                                                                mtab_out_ab2l, mtab_out_l2ab, mtab_out_ab2ori, mtab_out_ori2ab,
                                                                mtori_out_ori2l, mtori_out_l2ori, mtori_out_ab2ori, mtori_out_ori2ab,
                                                                out_ab2l, out_l2ab, out_ori2l, out_l2ori, out_ab2ori, out_ori2ab,
                                                                opt.margin_type)

        # backward
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

def main():
    # parse the args
    args = parse_option()

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_gh, l_mtgen, ab_mtgen, ori_mtgen = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    l_mtgen_op = torch.optim.SGD(l_mtgen.parameters(),
                                 lr=args.mt_learning_rate,
                                 momentum=args.mt_momentum,
                                 weight_decay=args.mt_weight_decay)

    ab_mtgen_op = torch.optim.SGD(ab_mtgen.parameters(),
                                  lr=args.mt_learning_rate,
                                  momentum=args.mt_momentum,
                                  weight_decay=args.mt_weight_decay)

    ori_mtgen_op = torch.optim.SGD(ori_mtgen.parameters(),
                                   lr=args.mt_learning_rate,
                                   momentum=args.mt_momentum,
                                   weight_decay=args.mt_weight_decay)

    # set mixed precision
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        losses = train(epoch, train_loader, model, contrast, criterion_gh, optimizer, l_mtgen, l_mtgen_op, ab_mtgen,
                       ab_mtgen_op, ori_mtgen, ori_mtgen_op, args)


        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('losses', losses, epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'l_mtgen': l_mtgen.state_dict(),
                'l_mtgen_op': l_mtgen_op.state_dict(),
                'ab_mtgen': ab_mtgen.state_dict(),
                'ab_mtgen_op': ab_mtgen_op.state_dict(),
                'ori_mtgen': ori_mtgen.state_dict(),
                'ori_mtgen_op': ori_mtgen_op.state_dict(),
                'epoch': epoch,
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
