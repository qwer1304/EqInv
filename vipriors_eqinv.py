import argparse
import os
import glob
#os.environ["CUDA_VISIBLE_DEVICES"] ='0'
import random
import shutil
import time
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import autograd

import torch.optim

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import defaultdict, OrderedDict

import utils
import utils_cluster
from torchvision.models.resnet import resnet50
#from randaugment import RandAugment

import hashlib

from itertools import chain

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--image_class', choices=['ImageNet', 'STL', 'CIFAR'], default='ImageNet', help='Image class, default=ImageNet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[30, 40], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--test-freq', default=None, type=int,
                    metavar='N', help='test frequency (default: None, test only at end)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action="store_true", default=False, help='evaluate?')
parser.add_argument('--extract_features', action="store_true", help="extract features for post processiin during evaluate")
parser.add_argument('--name', default=None, type=str,
                    help='exp name')
parser.add_argument('--clip-a', default=None, type=str,
                    help='clip architecture')
parser.add_argument('--adam', action="store_true", default=False, help='use adam optimizer?')
parser.add_argument('--class_num', default=1000, type=int, help='num of classes')
parser.add_argument('--temperature', default=0.1, type=float, help='temperature for contrastive loss')
parser.add_argument('--cont_weight', default=1.0, type=float, help='weight of contrastive loss')
parser.add_argument('--save_root', type=str, default='save', help='root dir for saving')

parser.add_argument('--stage1_model', type=str, default='ipirm', help='the stage 1 model')
parser.add_argument('--num_shot', type=str, default='50', help='the number of shot')
parser.add_argument('--random_aug', action="store_true", default=False, help='random_aug?')

#### add mask
parser.add_argument('--activat_type', type=str, default='sigmoid', choices=['sigmoid', 'ident', 'gumbel'], help='type of activation in mask')
parser.add_argument('--opt_mask', action="store_true", default=False, help='optimize the mask')

parser.add_argument('--pretrain_model', action="store_true", default=False, help='use pretrain model?')
parser.add_argument('--pretrain_path', type=str, default=None, help='the path of pretrain model')

# invariance
parser.add_argument('--inv', type=str, default='rex', choices=['rex', 'rvp', 'irmv1', 'sand'], help='type of invariant loss')
parser.add_argument('--inv_start', type=int, default=0, help='start epoch of inv loss')
parser.add_argument('--inv_weight', default=1., type=float, help='the weight of invariance')
parser.add_argument('--inv_ema', default=0., type=float, help='the weight of invariance EMA penalty. 0 - no EMA')
parser.add_argument('--mlp', action="store_true", default=False, help='use mlp before the loss and feature?')
parser.add_argument('--backbone_propagate', action="store_true", default=False, help='whether to propagate inv loss to backbone')
parser.add_argument('--nonancenvirm', action="store_true", default=False, help='use non-anchor environment IRM for penalty')
parser.add_argument('--pos_samples_fraction_in_inv', default=0., type=float, help='[0..1], fraction of positive samples to use in inv loss calculation')

# image
parser.add_argument('--image_size', type=int, default=224, help='image size')

# color in label
parser.add_argument('--target_transform', type=str, default=None, help='a function definition to apply to target')

# space between columns
parser.add_argument('--spaces', type=int, default=4, help='spaces between entries in progress print (instead of tab)')

# clustering
parser.add_argument('--cluster_path', type=str, default=None, 
    help='path to cluster file. None means automatic creation ./misc/<name>/env_ref_set_<resumed|pretrained|default>')
parser.add_argument('--only_cluster', action="store_true", help='only do clustering')
parser.add_argument('--cluster_temp', type=float, default=0.1, help='temperature for clusteing') 
parser.add_argument('--cluster_save_dist', action="store_true", help='save cluster distances in ./misc/<name>/env_ref_dist')
parser.add_argument('--num_clusters', type=int, default=2, help='number of custer K') 
parser.add_argument('--clusters_to_use', type=int, nargs='+', default=None, help='clusters to use out of K clusters') 


# shuffle validation and test datasets
parser.add_argument('--val_shuffle', action="store_true", default=False, help='shuffle validation daatase')
parser.add_argument('--test_shuffle', action="store_true", default=False, help='shuffle test daatase')

# balancing classes across environments
parser.add_argument('--inv_weight_to_balance_classes', action="store_true", default=False, help='balance environment classes imbalance by inverse weighting')
parser.add_argument('--drop_samples_to_balance_classes', action="store_true", default=False, help='balance environment classes imbalance by dropping extra samples')

# loss
parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')

args = parser.parse_args()

assert not (args.inv_weight_to_balance_classes and args.drop_samples_to_balance_classes), "Don't use both class balancing methods together"
best_acc1 = 0


'''
supervised contrastive loss
https://arxiv.org/abs/2004.11362
https://github.com/HobbitLong/SupContrast
'''
def info_nce_loss_supervised(features, batch_size, temperature=0.07, base_temperature=0.07, labels=None, choose_pos=None, weights=None):
    ### features (bs, views, dim)
    # Here, views = 1, but there're 2N samples ("multiviewed batch")
    labels = labels.contiguous().view(-1, 1)
    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    mask = torch.eq(labels, labels.T).float().cuda() # (bs,bs). Matrix of 1 for inputs that have the SAME label

    contrast_count = features.shape[1] # number of views, here - 1
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # here - does nothing

    anchor_feature = contrast_feature
    anchor_count = contrast_count

    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count) # here - does nothing
    # mask-out self-contrast cases
    """
    Writes all values from the tensor 'src' into 't' at the indices specified in the 'index' tensor. 
    For each value in 'src', its output index is specified by its index in 'src' for dimension != 'dim' 
    and by the corresponding value in 'index' for dimension = 'dim'.    
    """
    """
    Here: Writes 0 into 'all ones' tensor (bs,bs) at the indices specified in the tensor
    [0,batch_size * anchor_count) = [0, batch_size).
    So, since 't' is 2D and 'dim'==1, it writes 0 into the diagonal.
    """
    logits_mask = torch.scatter(
        torch.ones_like(mask),                                       # t - tensor to apply scatter to
        1,                                                           # dim
        torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),  # index
        0                                                            # src
    )
    mask = mask * logits_mask # remove self contrast from mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask # exponentiated logits w/o selfs
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos

    # apply weights
    if weights is not None:
        weights = weights.repeat_interleave(anchor_count).to(loss.device)  # shape: [batch_size * anchor_count]
        if choose_pos is None:
            loss = (weights * loss).sum() / weights.sum()
        else:
            weights = weights.view(anchor_count, batch_size)[:, choose_pos]
            loss = (loss.view(anchor_count, batch_size)[:, choose_pos] * weights).sum() / weights.sum()
    else:
        if choose_pos is None:
            loss = loss.view(anchor_count, batch_size).mean()
        else:
            loss = loss.view(anchor_count, batch_size)[:, choose_pos].sum() / choose_pos.sum()

    return loss


class Model_Imagenet(nn.Module):
    def __init__(self, feature_dim=128, image_class='ImageNet'):
        super(Model_Imagenet, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if image_class != 'ImageNet':  # STL, CIFAR
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            else:
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)



class Net(nn.Module):
    def __init__(self, num_class, pretrained_path, image_class='ImageNet'):
        super(Net, self).__init__()
        # encoder
        model = Model_Imagenet(image_class=image_class)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        msg = []
        if pretrained_path is not None and os.path.isfile(pretrained_path):
            print("=> loading pretrained checkpoint '{}'".format(pretrained_path))
            checkpoint = torch.load(pretrained_path, map_location=device)
            if 'state_dict' in checkpoint.keys():
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # Remove "module.model." prefix
                name = k.replace("module.model.", "")
                name = name.replace("module.", "")                  
                new_state_dict[name] = v
            state_dict = new_state_dict
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
        else:
            print("=> no pretrained checkpoint found at '{}'".format(pretrained_path))
        self.f = model.f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        if msg and msg.unexpected_keys:
            # Create your fc layer if not yet created
            if state_dict['fc.weight'].shape == self.fc.weight.shape and \
               state_dict['fc.bias'].shape == self.fc.bias.shape:
                # Copy weights
                self.fc.weight.data.copy_(state_dict['fc.weight'])
                self.fc.bias.data.copy_(state_dict['fc.bias'])
                print('Recovering fc layer from checkpoint')

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out



def main():
    if not os.path.exists('{}/{}'.format(args.save_root, args.name)):
        os.makedirs('{}/{}'.format(args.save_root, args.name))
    args.log_file = '{}/{}/eval_log.txt'.format(args.save_root, args.name)


    #################### Model ######################
    model_base = Net(num_class=args.class_num, pretrained_path=args.pretrain_path, image_class=args.image_class)
    import copy
    ft_fc = copy.deepcopy(model_base.fc)
    model_base.fc = nn.Identity()

    if args.opt_mask:
        mask_layer = torch.rand(ft_fc.weight.size(1),device="cuda")
    else:
        mask_layer = torch.ones(ft_fc.weight.size(1),device="cuda")   
    model = utils.ResNet_ft_eqinv(model_base, ft_fc, mask_layer=mask_layer, args=args)
    model = torch.nn.DataParallel(model).cuda()


    ######## define loss function (criterion) and optimizer
    init_lr = args.lr * args.batch_size / 256
    print('lr scale to %.2f' %(init_lr))
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_label_smoothed = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()
    if args.adam:
        init_lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.)
    else:
        optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    best_acc1 = 0
    best_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_epoch = checkpoint['best_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    ### prepare few shot dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.random_aug:
        train_transform_hard = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            normalize, ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize, ])


    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize, ])

    target_transform = eval(args.target_transform) if args.target_transform is not None else None
    
    images = utils.Imagenet_idx(root=args.data+'/val', transform=val_transform, target_transform=target_transform)
    val_loader = torch.utils.data.DataLoader(images, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.val_shuffle)
    test_images = utils.Imagenet_idx(root=args.data+'/testgt', transform=val_transform, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.test_shuffle)

    if args.random_aug:
        train_images = utils.Imagenet_idx_pair_transformone(root=args.data + '/train', transform_simple=train_transform, 
            transform_hard=train_transform_hard, target_transform=target_transform)
    else:
        train_images = utils.Imagenet_idx_pair(root=args.data+'/train', transform=train_transform, target_transform=target_transform)
    memory_images = utils.Imagenet_idx(root=args.data + '/train', transform=val_transform, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True)
    memory_loader = torch.utils.data.DataLoader(memory_images, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    activation_map = utils.activation_map(args.activat_type)


    if args.evaluate:
        print(f"Staring evaluation name: {args.name}")
        if args.resume:
            epoch = args.start_epoch
        else:
            epoch = -1
        print('eval on val data')
        validate(val_loader, model, criterion, args, epoch=epoch, prefix='Val: ')
        print('eval on test data')
        validate(test_loader, model, criterion, args, epoch=epoch, prefix='Test: ')
        return


    # Model loading order:
    # 1. Deafult model
    # 2. Pretrained model
    # 3. Checkpointed model
    
    # Cluster creation / loading order:
    # 0. Load given cluster (see loading order above)
    # 1. Load most recent in cluster save directory, if exists
    # 2. Create cluster from loaded model.
    #    Suffix reflects the model used: 'resumed', 'pretrained', 'default'
    
    #### Process Cluster
    assert args.stage1_model == 'ipirm'
    # number of images per class when training ip-irm
    assert args.num_shot in ['10', '20', '50']

    if args.cluster_path is None:
        directory = f'misc/{args.name}'
        pattern = 'env_ref_set_*' 

        # Find matching files
        files = glob.glob(os.path.join(directory, pattern))

        # Sort by modification time
        files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
        fp_exist = None
        if files_sorted:
            fp_exist = files_sorted[0]
        
        if args.resume:
            hash_object = hashlib.sha256(args.resume.encode())
            hex_dig = hash_object.hexdigest()
            suffix = hex_dig + '_resumed'
        elif args.pretrain_path is not None and os.path.isfile(args.pretrain_path):
            hash_object = hashlib.sha256(args.pretrain_path.encode())
            hex_dig = hash_object.hexdigest()
            suffix = hex_dig + '_pretrained'
        else:
            suffix = 'default'
        fp_new = os.path.join(directory, 'env_ref_set_' + suffix)
        
        if args.only_cluster or not fp_exist:
            fp = fp_new
        else:
            fp = fp_exist
            
    else:
        directory = f'misc/{args.name}'
        fp = args.cluster_path
        
    fp_dist = os.path.join(directory, 'env_ref_dist')
    
    if args.only_cluster or not os.path.exists(fp):
        # Cannot use end="" b/c cal_cosine_distance prints progress bar and overwrites its
        if args.only_cluster:
            print('Recalculation of cluster file requested... ')
        else:
            print('No cluster file, creating... ')
        if args.cluster_save_dist:
            env_ref_set, dist = utils_cluster.cal_cosine_distance(model, memory_loader, args.class_num, temperature=args.cluster_temp, 
                anchor_class=None, class_debias_logits=True, return_dist=True, K=args.num_clusters)
            os.makedirs(os.path.dirname(fp_dist), exist_ok=True)
            # dist is a dictionary with anchor classes as keys of similarity scores
            torch.save(dist, fp_dist)
            print(f"Cluster distances saved in {fp_dist}")
        else:
            env_ref_set = utils_cluster.cal_cosine_distance(model, memory_loader, args.class_num, temperature=args.cluster_temp, 
                anchor_class=None, class_debias_logits=True, K=args.num_clusters)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        torch.save(env_ref_set, fp)
        print(f'cluster {fp} ready!') 
        if args.only_cluster:
            return
    else:
        env_ref_set = torch.load(fp)
        print(f'Cluster {fp} loaded.')
        assert len(env_ref_set[0]) == args.num_clusters, "Num clusters in cluster file {} != num_clusters {}".format(len(env_ref_set[0]), args.num_clusters)
        assert args.clusters_to_use is None or \
            max(args.clusters_to_use) <= args.num_clusters-1, "Largest cluster to use {} must be < {}".format(max(args.clusters_to_use), args.num_clusters)

    print(f"Staring training name: {args.name}")
    for epoch in range(args.start_epoch, args.epochs):

        if  args.adam and epoch == args.inv_start and args.inv_weight > 0 and args.nonancenvirm:
            optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.)

        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        if args.nonancenvirm:
            criterion_tuple = (criterion, criterion, criterion_label_smoothed)
            train_env_nonanchirm(train_loader, model, activation_map, env_ref_set, criterion_tuple, optimizer, epoch, args)
        else:
            train_env(train_loader, model, activation_map, env_ref_set, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, epoch, prefix='Val: ')

        # evaluate on test set
        if args.test_freq is not None and epoch % args.test_freq == 0:
            acc1_test = validate(test_loader, model, criterion, args, epoch, prefix='Test: ')

        # remember best acc@1 & best epoch and save checkpoint
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_epoch = epoch + 1
            
        save_checkpoint({
            'epoch':        epoch + 1,
            'arch':         args.arch,
            'state_dict':   model.state_dict(),
            'best_acc1':    best_acc1,
            'best_epoch':   best_epoch,
            'optimizer':    optimizer.state_dict(),
        }, is_best, args, filename='{}/{}/checkpoint.pth.tar'.format(args.save_root, args.name))

    if args.opt_mask:
        torch.save(model.module.mask_layer, '{}/{}/mask_layer_opt'.format(args.save_root, args.name))

    utils.write_log('\nThe best Val accuracy: {}'.format(best_acc1), args.log_file, print_=True)
    utils.write_log('\nStart to test on Test Set', args.log_file, print_=True)
    acc1_test = validate(test_loader, model, criterion, args, epoch, prefix='Test: ')

def _irm_penalty(logits, y):
    device = "cuda" if logits.is_cuda else "cpu"
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
    loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
    grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
    grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
    result = torch.sum(grad_1 * grad_2)
    return result

def _sand_penalty(env_loss_list, model):
    real_model = model.module if isinstance(model, torch.nn.DataParallel) else model # real_model is a reference, so can update
    params = list(chain(real_model.model.parameters(), real_model.fc.parameters()))
    param_gradients = [[] for _ in params]
    for env_loss in env_loss_list:
        env_grads = autograd.grad(env_loss, params, retain_graph=True)
        for grads, env_grad in zip(param_gradients, env_grads):
            grads.append(env_grad)
    return param_gradients

def _mask_grads(gradients, model, k=10., tau=0.7):
    """
    Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
    parameter based on the agreement of gradients coming from different environments.
    """
    real_model = model.module if isinstance(model, torch.nn.DataParallel) else model # real_model is a reference, so can update
    device = gradients[0][0].device
    params = list(chain(real_model.model.parameters(), real_model.fc.parameters()))
    for param, grads in zip(params, gradients):
        grads = torch.stack(grads, dim=0)
        avg_grad = torch.mean(grads, dim=0)
        grad_signs = torch.sign(grads)
        gamma = torch.tensor(1.0).to(device)
        grads_var = grads.var(dim=0)
        grads_var[torch.isnan(grads_var)] = 1e-17
        lam = (gamma * grads_var).pow(-1)
        mask = torch.tanh(k * lam * (torch.abs(grad_signs.mean(dim=0)) - tau))
        mask = torch.max(mask, torch.zeros_like(mask))
        mask[torch.isnan(mask)] = 1e-17
        mask_t = (mask.sum() / mask.numel())
        param.grad = mask * avg_grad
        param.grad *= (1. / (1e-10 + mask_t))

def train_env_nonanchirm(train_loader, model, activation_map, env_ref_set, criterion_tuple, optimizer, epoch, args):
    criterion_ERM, criterion_cont, criterion_inv = criterion_tuple
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_cont = AverageMeter('Loss_Cont', ':.4e')
    losses_inv = AverageMeter('Loss_Inv', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    LR = AverageMeter('LR', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_cont, losses_inv, top1, top5, LR],
        prefix="Epoch: [{}]".format(epoch),
        log_file=args.log_file)

    # switch to train mode
    model.train()

    all_sample_num = len(train_loader.dataset)

    end = time.time()
    for i, training_items in enumerate(train_loader):
        # training_items is a batch
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.random_aug:
            images1, images2, images1_hard, images2_hard, target, images_idx = training_items
            images1_hard, images2_hard = images1_hard.cuda(non_blocking=True), images2_hard.cuda(non_blocking=True)
        else:
            images1, images2, target, images_idx = training_items
        # images1 and images2 are two views (transformations) of the SAME image from the dataset
        images1, images2 = images1.cuda(non_blocking=True), images2.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        if args.inv_weight_to_balance_classes:
            # target: shape [batch_size]
            classes, counts = torch.unique(target, return_counts=True)
            freq = counts.float() / counts.sum()  # class frequencies
            inv_freq = 1.0 / freq                 # inverse frequencies
            class_weights = inv_freq / inv_freq.sum()  # normalize if needed (optional)

            # Assign weight to each sample
            weights = class_weights[target]  # shape: [batch_size]    
            weights = weights.to(target.device, non_blocking=True)
        else:
            weights = torch.ones_like(target)

        # compute output
        # def forward(self, image, return_feature=False, return_masked_feature=False)
        # output = self.fc(masked_feature_erm)
        # return self.mlp(masked_feature_erm), masked_feature_inv, output
        # masked_feature_inv is detached from backbone (unless backbone_propagate flag is ON)
        masked_feature1, masked_feature_inv1, output1 = model(images1, return_masked_feature=True)
        masked_feature2, masked_feature_inv2, output2 = model(images2, return_masked_feature=True)
        if args.random_aug:
            output_hard = model(torch.cat([images1_hard, images2_hard]))

        # both augmented images combined 
        # They collapse the two views into a SINGLE one! WHY??????????!!!!!!!!!!!
        # This is due to "Supervised Contrastive Loss" as in the original paper (SupCon).
        masked_feature_for_globalcont, masked_feature, output = torch.cat([masked_feature1, masked_feature2], dim=0), torch.cat([masked_feature_inv1, masked_feature_inv2], dim=0), torch.cat([output1, output2], dim=0)
        target, images_idx = torch.cat([target, target], dim=0), torch.cat([images_idx, images_idx], dim=0) # (2B,...)
        weights = torch.cat([weights, weights], dim=0)
        images_idx = images_idx.to(target.device)

        if args.inv_weight > 0:
            # compute envs for different classes
            env_nll, env_pen, temp_pen = [], [], []
            for class_idx in range(args.class_num):

                mask_pos = target==class_idx # choose the specifc positive samples
                if mask_pos.sum() == 0: # batch has no images from current class
                    continue

                # note that these are positive & negatives samples in the batch NOT split into environments
                # positiveness/negativeness is determined by sample's label
                # Note that here positives include ALL positives - the anchor sample hasn't been determined yet.
                output_pos, target_num_pos, masked_feature_pos, weights_pos =  \
                    output[mask_pos], target[mask_pos], masked_feature[mask_pos], weights[mask_pos] # get positive and negative samples
                output_neg, images_idx_neg, target_num_neg, masked_feature_neg, weights_neg = \
                    output[~mask_pos], images_idx[~mask_pos], target[~mask_pos], masked_feature[~mask_pos], weights[~mask_pos]

                # generate the env lookup table
                """
                    env_ref_set is a dictionary over class labels.
                    each entry is a tuple over class-environments (K = 2) of sample indices in loader that are assigned to that environment (equal number)
                    the environments have been precomputed by running the images through the default (pre-trained) model, calculated the (corrected) cosine
                    distance between the "other" samples and anchor samples, sorting in descending order the distances and splitting the result 50/50 into
                    two environments.
                    all_samples_env_table is a table (num_samples in loader, num_samples in class-environment)
                """
                env_ref_set_class = env_ref_set[class_idx]
                # all_sample_num is the number of samples in the dataset before doubling
                if args.clusters_to_use is not None:
                    env_ref_set_class = [env_ref_set_class[j] for j in args.clusters_to_use]
                all_samples_env_table = torch.zeros(all_sample_num, len(env_ref_set_class))
                for env_idx in range(len(env_ref_set_class)):
                    all_samples_env_table[env_ref_set_class[env_idx], env_idx] = 1  # set "other" samples of current env to 1

                all_samples_env_table = all_samples_env_table.to(output.device)
                # traverse different envs
                for env_idx in range(len(env_ref_set_class)): # split the negative samples
                    # assign_samples selects the negative samples in this environment
                    output_neg_env, target_num_neg_env, masked_feature_neg_env, weights_neg_env = \
                        utils_cluster.assign_samples([output_neg, target_num_neg, masked_feature_neg, weights_neg], \
                                                     images_idx_neg, all_samples_env_table, env_idx)
                    
                    # when the number of samples per label is balanced, because the "other" samples are split equally between two environments, we get
                    # imbalance of positive and negative samples.
                    # output_env are all samples (positive and negative) of that environment
                    if args.drop_samples_to_balance_classes:
                        rand_idx = torch.randperm(output_pos.size(0))[:min(output_pos.size(0), output_neg_env.size(0))]
                        output_pos_sub = output_pos[rand_idx]
                        target_num_pos_sub = target_num_pos[rand_idx]
                        masked_feature_pos_sub = masked_feature_pos[rand_idx]
                        weights_pos_sub = weights_pos[rand_idx]
                    else:
                        output_pos_sub = output_pos
                        target_num_pos_sub = target_num_pos
                        masked_feature_pos_sub = masked_feature_pos
                        weights_pos_sub = weights_pos
                        
                    # retain only args.pos_samples_to_use fraction of positive samples
                    rand_idx = torch.randperm(output_pos_sub.size(0))[:int(output_pos_sub.size(0)*args.pos_samples_fraction_in_inv)]
                    output_pos_sub = output_pos_sub[rand_idx]
                    target_num_pos_sub = target_num_pos_sub[rand_idx]
                    masked_feature_pos_sub = masked_feature_pos_sub[rand_idx]
                    weights_pos_sub = weights_pos_sub[rand_idx]
                    
                    output_env, target_num_env, masked_feature_env, weights_env = \
                        torch.cat([output_pos_sub, output_neg_env], dim=0), \
                        torch.cat([target_num_pos_sub, target_num_neg_env], dim=0), \
                        torch.cat([masked_feature_pos_sub, masked_feature_neg_env], dim=0), \
                        torch.cat([weights_pos_sub, weights_neg_env], dim=0)
                    masked_feature_env_norm = F.normalize(masked_feature_env, dim=-1)
                    # cont_loss_env is the contrastive loss of this environment
                    # masked_feature_env = mask(model(x)) <-------- NOTE: mlp is NOT applied!!!!!!
                    """
                    cont_loss_env = args.cont_weight * \
                        info_nce_loss_supervised(masked_feature_env_norm.unsqueeze(1),   # stack of positive and negative samples in this env
                                                 masked_feature_env_norm.size(0),        # their number (batch size)
                                                 temperature=args.temperature, 
                                                 labels=target_num_env,                  # labels of these samples
                                                 choose_pos=target_num_env==class_idx,   # position of positive samples
                                                 weights=weights_env)
                    """

                    # nll of "other" in this environment appended to list
                    # output = fc(mask(model(x))
                    env_nll.append(criterion_inv(output_env, target_num_env)) 
                    if args.inv == "irmv1":
                        temp_pen.append(_irm_penalty(output_env, target_num_env))
                    else:
                        temp_pen.append(env_nll[-1]) # loss of this environment appended to list
                # end for env_idx in range(len(env_ref_set_class))
                
                if args.inv == 'rex':
                    env_pen.append(torch.var(torch.stack(temp_pen))) # varaince of losses of the environments appended to list of losses of all classes
                elif args.inv == 'rvp':
                    epsilon = 1e-8
                    env_pen.append(torch.std(torch.stack(temp_pen)) + epsilon) # std of losses of the environments appended to list of losses of all classes
                elif args.inv == "irmv1":
                    env_pen.append(torch.stack(temp_pen).mean())
                elif args.inv == "sand":
                    params_grads = _sand_penalty(temp_pen, model)
                    if not env_pen: # first pass
                        env_pen = params_grads
                    else:
                        for agg_grads, new_grads in zip(env_pen, params_grads):
                            agg_grads.extend(new_grads)                   
                else:
                    raise ValueError(f'invalid inv method {args.inv}')
                temp_pen = []
            # end for class_idx in range(args.class_num):
            
            # Invariance Term: mean of variances of contrastive losses of class-environments
            if epoch >= args.inv_start and args.inv != "sand":
                inv_weight = args.inv_weight
                penalty = sum(env_pen) / len(env_pen) # average loss over classes
                real_model = model.module if isinstance(model, torch.nn.DataParallel) else model # real_model is a reference, so can update
                inv_running_penalty = real_model.inv_running_penalty.to(penalty.device).detach()
                inv_running_penalty = args.inv_ema * inv_running_penalty + (1 - args.inv_ema) * penalty
                real_model.inv_running_penalty = inv_running_penalty
                loss_inv = inv_weight * inv_running_penalty
            else:
                loss_inv = torch.Tensor([0.]).cuda()
            # end for class_idx in range(args.class_num):
        else: # args.inv_weight == 0:
            loss_inv = torch.Tensor([0.]).cuda()


        # ERM loss
        # output = fc(mask(model(x))
        if args.random_aug:
            loss_erm = criterion_ERM(output_hard, target)
        else:
            loss_erm = criterion_ERM(output, target)
        # masked_feature_for_globalcont is mlp(mask * model(x)) 
        masked_feature_for_globalcont_norm = F.normalize(masked_feature_for_globalcont, dim=-1)
        #                            stack of masked_feature1, masked_feature2. each is: mlp(masked_feature_erm).
        #                            1 and 2 are two copies of augmented image.
        loss_cont = args.cont_weight * \
            info_nce_loss_supervised(masked_feature_for_globalcont_norm.unsqueeze(1),
                                     masked_feature_for_globalcont_norm.size(0), 
                                     temperature=args.temperature, 
                                     labels=target)


        loss_all = loss_erm + loss_cont + loss_inv
        assert torch.isfinite(loss_inv).item(), 'loss_inv not finite' 
        assert torch.isfinite(loss_cont).item(), 'loss_cont not finite' 
        assert torch.isfinite(loss_erm).item(), 'loss_erm not finite' 

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss_erm.item(), images1.size(0)+images2.size(0))
        losses_cont.update(loss_cont.item(), images1.size(0)+images2.size(0))
        losses_inv.update(loss_inv.item(), images1.size(0)+images2.size(0))
        top1.update(acc1.item(), images1.size(0)+images2.size(0))
        top5.update(acc5.item(), images1.size(0)+images2.size(0))
        LR.update(optimizer.param_groups[0]['lr'])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.inv == "sand" and epoch >= args.inv_start:
            # gradient masking applied here
            _mask_grads(env_pen, model, args.inv_weight)
        else:
            loss_all.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args.spaces)

    progress.prefix = "Train: "  
    progress.display_summary(epoch)

def train_env(train_loader, model, activation_map, env_ref_set, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_cont = AverageMeter('Loss_Cont', ':.4e')
    losses_inv = AverageMeter('Loss_Inv', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    LR = AverageMeter('LR', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_cont, losses_inv, top1, top5, LR],
        prefix="Epoch: [{}]".format(epoch),
        log_file=args.log_file)

    # switch to train mode
    model.train()

    all_sample_num = len(train_loader.dataset)

    end = time.time()
    for i, training_items in enumerate(train_loader):
        # training_items is a batch
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.random_aug:
            images1, images2, images1_hard, images2_hard, target, images_idx = training_items
            images1_hard, images2_hard = images1_hard.cuda(non_blocking=True), images2_hard.cuda(non_blocking=True)
        else:
            images1, images2, target, images_idx = training_items
        # images1 and images2 are two views (transformations) of the SAME image from the dataset
        images1, images2 = images1.cuda(non_blocking=True), images2.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        if args.inv_weight_to_balance_classes:
            # target: shape [batch_size]
            classes, counts = torch.unique(target, return_counts=True)
            freq = counts.float() / counts.sum()  # class frequencies
            inv_freq = 1.0 / freq                 # inverse frequencies
            class_weights = inv_freq / inv_freq.sum()  # normalize if needed (optional)

            # Assign weight to each sample
            weights = class_weights[target]  # shape: [batch_size]    
            weights = weights.to(target.device, non_blocking=True)
        else:
            weights = torch.ones_like(target)

        # compute output
        # def forward(self, image, return_feature=False, return_masked_feature=False)
        # output = self.fc(masked_feature_erm)
        # return self.mlp(masked_feature_erm), masked_feature_inv, output
        # masked_feature_inv is detached from backbone
        masked_feature1, masked_feature_inv1, output1 = model(images1, return_masked_feature=True)
        masked_feature2, masked_feature_inv2, output2 = model(images2, return_masked_feature=True)
        if args.random_aug:
            output_hard = model(torch.cat([images1_hard, images2_hard]))

        # both augmented images combined 
        # They collapse the two views into a SINGLE one! WHY??????????!!!!!!!!!!!
        # This is due to "Supervised Contrastive Loss" as in the original paper (SupCon).
        masked_feature_for_globalcont, masked_feature, output = torch.cat([masked_feature1, masked_feature2], dim=0), torch.cat([masked_feature_inv1, masked_feature_inv2], dim=0), torch.cat([output1, output2], dim=0)
        target, images_idx = torch.cat([target, target], dim=0), torch.cat([images_idx, images_idx], dim=0) # (2B,...)
        weights = torch.cat([weights, weights], dim=0)
        images_idx = images_idx.to(target.device)

        if args.inv_weight > 0:
            # compute envs for different classes
            env_nll, env_cont_nll, env_pen, temp_pen = [], [], [], []
            for class_idx in range(args.class_num):

                mask_pos = target==class_idx # choose the specifc positive samples
                if mask_pos.sum() == 0: # batch has no images from current class
                    continue

                # note that these are positive & negatives samples in the batch NOT split into environments
                # positiveness/negativeness is determined by sample's label
                # Note that here positives include ALL positives - the anchor sample hasn't been determined yet.
                output_pos, target_num_pos, masked_feature_pos, weights_pos =  \
                    output[mask_pos], target[mask_pos], masked_feature[mask_pos], weights[mask_pos] # get positive and negative samples
                output_neg, images_idx_neg, target_num_neg, masked_feature_neg, weights_neg = \
                    output[~mask_pos], images_idx[~mask_pos], target[~mask_pos], masked_feature[~mask_pos], weights[~mask_pos]

                # generate the env lookup table
                """
                    env_ref_set is a dictionary over class labels.
                    each entry is a tuple over class-environments (K = 2) of sample indices in loader that are assigned to that environment (equal number)
                    the environments have been precomputed by running the images through the default (pre-trained) model, calculated the (corrected) cosine
                    distance between the "other" samples and anchor samples, sorting in descending order the distances and splitting the result 50/50 into
                    two environments.
                    all_samples_env_table is a table (num_samples in loader, num_samples in class-environment)
                """
                env_ref_set_class = env_ref_set[class_idx]
                # all_sample_num is the number of samples in the dataset before doubling
                all_samples_env_table = torch.zeros(all_sample_num, len(env_ref_set_class))
                for env_idx in range(len(env_ref_set_class)):
                    all_samples_env_table[env_ref_set_class[env_idx], env_idx] = 1  # set "other" samples of current env to 1

                all_samples_env_table = all_samples_env_table.to(output.device)
                # traverse different envs
                for env_idx in range(len(env_ref_set_class)): # split the negative samples
                    # assign_samples selects the negative samples in this environment
                    output_neg_env, target_num_neg_env, masked_feature_neg_env, weights_neg_env = \
                        utils_cluster.assign_samples([output_neg, target_num_neg, masked_feature_neg, weights_neg], \
                                                     images_idx_neg, all_samples_env_table, env_idx)
                    
                    # when the number of samples per label is balanced, because the "other" samples are split equally between two environments, we get
                    # imbalance of positive and negative samples.
                    # output_env are all samples (positive and negative) of that environment
                    if args.drop_samples_to_balance_classes:
                        rand_idx = torch.randperm(output_pos.size(0))[:min(output_pos.size(0), output_neg_env.size(0))]
                        output_pos_sub = output_pos[rand_idx]
                        target_num_pos_sub = target_num_pos[rand_idx]
                        masked_feature_pos_sub = masked_feature_pos[rand_idx]
                        weights_pos_sub = weights_pos[rand_idx]
                    else:
                        output_pos_sub = output_pos
                        target_num_pos_sub = target_num_pos
                        masked_feature_pos_sub = masked_feature_pos
                        weights_pos_sub = weights_pos
                        
                    output_env, target_num_env, masked_feature_env, weights_env = \
                        torch.cat([output_pos_sub, output_neg_env], dim=0), \
                        torch.cat([target_num_pos_sub, target_num_neg_env], dim=0), \
                        torch.cat([masked_feature_pos_sub, masked_feature_neg_env], dim=0), \
                        torch.cat([weights_pos_sub, weights_neg_env], dim=0)
                    masked_feature_env_norm = F.normalize(masked_feature_env, dim=-1)
                    # cont_loss_env is the contrastive loss of this environment
                    # masked_feature_env = mask(model(x)) <-------- NOTE: mlp is NOT applied!!!!!!
                    cont_loss_env = args.cont_weight * \
                        info_nce_loss_supervised(masked_feature_env_norm.unsqueeze(1),   # stack of positive and negative samples in this env
                                                 masked_feature_env_norm.size(0),        # their number (batch size)
                                                 temperature=args.temperature, 
                                                 labels=target_num_env,                  # labels of these samples
                                                 choose_pos=target_num_env==class_idx,   # position of positive samples
                                                 weights=weights_env)

                    env_nll.append(criterion(output_env, target_num_env)) # nll of this environment appended to list
                    temp_pen.append(cont_loss_env) # contrastive loss of this environment appended to list

                if args.inv == 'rex':
                    env_pen.append(torch.var(torch.stack(temp_pen))) # varaince of losses of the environments appended to list of losses of all classes
                elif args.inv == 'rvp':
                    epsilon = 1e-8
                    env_pen.append(torch.std(torch.stack(temp_pen)) + epsilon) # std of losses of the environments appended to list of losses of all classes
                else:
                    raise ValueError(f'invalid inv method {args.inv}')
                temp_pen = []


            # Invariance Term: mean of variances of contrastive losses of class-environments
            if epoch >= args.inv_start:
                inv_weight = args.inv_weight
                penalty = sum(env_pen) / len(env_pen) # average loss over classes
                real_model = model.module if isinstance(model, torch.nn.DataParallel) else model # real_model is a reference, so can update
                inv_running_penalty = real_model.inv_running_penalty.to(penalty.device).detach()
                inv_running_penalty = args.inv_ema * inv_running_penalty + (1 - args.inv_ema) * penalty
                real_model.inv_running_penalty = inv_running_penalty
                loss_inv = inv_weight * inv_running_penalty
            else:
                loss_inv = torch.Tensor([0.]).cuda()

        else:
            loss_inv = torch.Tensor([0.]).cuda()


        # ERM loss
        # output = fc(mask(model(x))
        if args.random_aug:
            loss_erm = criterion(output_hard, target)
        else:
            loss_erm = criterion(output, target)
        # masked_feature_for_globalcont is mlp(mask * model(x)) 
        masked_feature_for_globalcont_norm = F.normalize(masked_feature_for_globalcont, dim=-1)
        #                            stack of masked_feature1, masked_feature2. each is: mlp(masked_feature_erm).
        #                            1 and 2 are two copies of augmented image.
        loss_cont = args.cont_weight * \
            info_nce_loss_supervised(masked_feature_for_globalcont_norm.unsqueeze(1),
                                     masked_feature_for_globalcont_norm.size(0), 
                                     temperature=args.temperature, 
                                     labels=target)


        loss_all = loss_erm + loss_cont + loss_inv
        assert torch.isfinite(loss_inv).item(), 'loss_inv not finite' 
        assert torch.isfinite(loss_cont).item(), 'loss_cont not finite' 
        assert torch.isfinite(loss_erm).item(), 'loss_erm not finite' 


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss_erm.item(), images1.size(0)+images2.size(0))
        losses_cont.update(loss_cont.item(), images1.size(0)+images2.size(0))
        losses_inv.update(loss_inv.item(), images1.size(0)+images2.size(0))
        top1.update(acc1.item(), images1.size(0)+images2.size(0))
        top5.update(acc5.item(), images1.size(0)+images2.size(0))
        LR.update(optimizer.param_groups[0]['lr'])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args.spaces)

    progress.prefix = "Train: "  
    progress.display_summary(epoch)


def validate(val_loader, model, criterion, args, epoch, prefix='Test: '):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix=prefix,
        log_file=args.log_file)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        masked_feature_erm_list = []
        target_list = []
        target_raw_list = []
        output_list = []
        target_transform = val_loader.dataset.target_transform
        if args.extract_features:
            val_loader.dataset.target_transform = None
            
        for i, (images, target, images_idx) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_raw = target
            if args.extract_features and target_transform is not None:
                target = target_transform(target_raw).cuda(non_blocking=True)

            # compute output
            if args.extract_features:
                    mlp = model.module.args.mlp
                    model.module.args.mlp = False
                    masked_feature_erm, _, output = model(images, return_masked_feature=True)
                    model.module.args.mlp = mlp
                    masked_feature_erm_list.append(masked_feature_erm)
                    target_list.append(target)
                    target_raw_list.append(target_raw)
                    output_list.append(output)
            else:
                output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            """
            i is the minibatch index. print_freq > 1 (usually 20) is the number of minibatches between print-outs.
            Each AverageMeter is updated EVERY minibatch, which updates the running average.
            But, the average is printed out ONLY every print_freq minibatches.
            So, between prin-outs i and j, there were print_freq-1 updates to the average that
            weren't printed. This explains the apparent inconsistency of printed averages between
            successive print-outs.
            """
            if i % args.print_freq == 0:
                progress.display(i, args.spaces)

        if masked_feature_erm_list:
            masked_feature_erm = torch.cat(masked_feature_erm_list, dim=0)
            target = torch.cat(target_list, dim=0)
            target_raw = torch.cat(target_raw_list, dim=0)
            output = torch.cat(output_list, dim=0)
            # Save to file
            prefix = "test" if "Test" in prefix else "val"
            directory = f'misc/{args.name}'
            fp = os.path.join(directory, f"{prefix}_features_dump.pt")       
            os.makedirs(os.path.dirname(fp), exist_ok=True)

            torch.save({
                'features':     masked_feature_erm,
                'labels':       target,
                'labels_raw':   target_raw,
                'logits':       output,
                'model_epoch':  epoch,
                'head_weights': model.module.fc.weight,  # shape: (num_classes, embed_dim)
                'head_bias':    model.module.fc.bias,    # shape: (num_classes,)
                'n_classes':    args.class_num,
            }, fp)
            print(f"Dumped features into {fp}")
            
        progress.display_summary(epoch)
        val_loader.dataset.target_transform = target_transform

    return top1.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}/model_best.pth.tar'.format(args.save_root, args.name))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):           
        self.val = val
        self.sum = self.sum + val * n # don't use +=
        self.count = self.count + n # don't use +=
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_file='eval_log.txt'):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_file = log_file

    def display(self, batch, spaces):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print((' ' * spaces).join(entries))


    def display_summary(self, epoch):
        entries = [" * {} Epoch: [{}]".format(self.prefix, epoch)]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

        utils.write_log('{} Epoch {} '.format(self.prefix, epoch) + ' '.join(entries), self.log_file, print_=False)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        num_classes = output.size(1)
        maxk = min(max(topk), num_classes)
        
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




if __name__ == '__main__':
    main()

