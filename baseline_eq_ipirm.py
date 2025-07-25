import argparse
import os
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

import torch.optim

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import defaultdict
#from randaugment import RandAugment

import utils

from torchvision.models.resnet import resnet50

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ViPriors Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
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

parser.add_argument('--name', default=None, type=str,
                    help='exp name')
parser.add_argument('--adam', action="store_true", default=False, help='use adam optimizer?')
parser.add_argument('--class_num', default=1000, type=int, help='num of classes')
parser.add_argument('--temperature', default=0.1, type=float, help='temperature for contrastive loss')
parser.add_argument('--cont_weight', default=1.0, type=float, help='weight of contrastive loss')
parser.add_argument('--save_root', type=str, default='save', help='root dir for saving')
parser.add_argument('--ft_stage', type=str, default='all', help='realse all the backbone')

parser.add_argument('--random_aug', action="store_true", default=False, help='random_aug?')
parser.add_argument('--pretrain_path', type=str, default=None, help='the path of pretrain model')

# image
parser.add_argument('--image_size', type=int, default=224, help='image size')

# color in label
parser.add_argument('--target_transform', type=str, default=None, help='a function definition to apply to target')

# space between columns
parser.add_argument('--spaces', type=int, default=4, help='spaces between entries in progress print (instead of tab)')

# shuffle validation and test datasets
parser.add_argument('--val_shuffle', action="store_true", default=False, help='shuffle validation daatase')
parser.add_argument('--test_shuffle', action="store_true", default=False, help='shuffle test daatase')

args = parser.parse_args()

best_acc1 = 0



class Model_Imagenet(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model_Imagenet, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
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
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()
        # encoder
        model = Model_Imagenet()
        if os.path.isfile(pretrained_path):
            print("=> loading checkpoint '{}'".format(pretrained_path))
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            _, ext = os.path.splitext(pretrained_path)
            if ext == '.tar':
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
        else:
            print("=> no checkpoint found at '{}'".format(pretrained_path))
        self.f = model.f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

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
    model_base = Net(num_class=args.class_num, pretrained_path=args.pretrain_path)
    import copy
    ft_fc = copy.deepcopy(model_base.fc)
    model_base.fc = nn.Identity()

    model = utils.ResNet_ft(model_base, ft_fc, args=args)
    model = torch.nn.DataParallel(model).cuda()

    ######## define loss function (criterion) and optimizer
    init_lr = args.lr * args.batch_size / 256
    print('lr scale to %.2f' %(init_lr))
    criterion = nn.CrossEntropyLoss().cuda()
    if args.adam:
        init_lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.)
    else:
        optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
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
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            normalize, ])
    else:
        train_tranform = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,])


    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,])

    target_transform = eval(args.target_transform) if args.target_transform is not None else None

    images = utils.Imagenet_idx(root=args.data+'/val', transform=val_transform, target_transform=target_transform)
    val_loader = torch.utils.data.DataLoader(images, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.val_shuffle)


    # train_images = torchvision.datasets.ImageNet(data_path, split='train', transform=train_tranform)
    train_images = utils.Imagenet_idx(root=args.data+'/train', transform=train_tranform, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    test_images = utils.Imagenet_idx(root=args.data+'/testgt', transform=val_transform, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.test_shuffle)

    if args.evaluate:
        validate(val_loader, model, criterion, args, epoch=-1, prefix='Val: ')
        return


    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, epoch, prefix='Val: ')

        # evaluate on test set
        if args.test_freq is not None and epoch % args.test_freq == 0:
            acc1_test = validate(test_loader, model, criterion, args, epoch, prefix='Test: ')

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args, filename='{}/{}/checkpoint.pth.tar'.format(args.save_root, args.name))


    utils.write_log('\nStart to test on Test Set', args.log_file, print_=True)
    acc1_test = validate(test_loader, model, criterion, args, epoch, prefix='Test: ')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    LR = AverageMeter('LR', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, LR],
        prefix="Epoch: [{}]".format(epoch),
        log_file=args.log_file)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target, images_idx) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss_all = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss_all.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
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
        for i, (images, target, images_idx) in enumerate(val_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
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

            if i % args.print_freq == 0:
                progress.display(i, args.spaces)

        progress.display_summary(epoch)

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
        self.sum = self.sum + val * n
        self.count = self.count + n
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
        entries = [" *", "Epoch: [{}]".format(epoch)]
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

