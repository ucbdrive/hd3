import os
import time
import logging
import shutil
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

import hd3model as models
from utils.utils import *
import data.hd3data as datasets

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch Implementation of HD^3')
    parser.add_argument('--dataset_name', type=str, help='dataset name')
    parser.add_argument('--train_root', type=str, help='training data root')
    parser.add_argument('--val_root', type=str, help='validation data root')
    parser.add_argument('--train_list', type=str, help='train list')
    parser.add_argument('--val_list', type=str, help='val list')

    parser.add_argument('--task', type=str, help='stereo or flow')
    parser.add_argument('--encoder', type=str, help='vgg or dlaup')
    parser.add_argument('--decoder', type=str, help='resnet or hda')
    parser.add_argument('--context',
                        action='store_true',
                        default=False,
                        help='context module')

    parser.add_argument('--base_lr',
                        type=float,
                        default=1e-4,
                        help='learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='batch size')
    parser.add_argument('--workers',
                        type=int,
                        default=16,
                        help='data loader workers')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=4e-4,
                        help='weight decay')

    parser.add_argument('--pretrain',
                        type=str,
                        default='',
                        help='path to pretrained model')
    parser.add_argument('--pretrain_base',
                        type=str,
                        default='',
                        help='path to pretrained base network')
    parser.add_argument('--evaluate',
                        action='store_true',
                        default=False,
                        help='evaluate on validation set')
    parser.add_argument('--batch_size_val',
                        type=int,
                        default=1,
                        help='batch size for validation during training')
    parser.add_argument('--save_step',
                        type=int,
                        default=50,
                        help='model save step')
    parser.add_argument('--save_path',
                        type=str,
                        default='model',
                        help='model and summary save path')
    parser.add_argument('--print_freq',
                        type=int,
                        default=10,
                        help='print frequency')
    parser.add_argument('--visual_freq',
                        type=int,
                        default=20,
                        help='visualization frequency')
    return parser


# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger, writer
    args = get_parser().parse_args()
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info(args)
    logger.info("=> creating model ...")

    ### model ###
    corr_range = [4, 4, 4, 4, 4, 4]
    if args.task == 'flow':
        corr_range = corr_range[:5]
    model = models.HD3Model(args.task, args.encoder, args.decoder, corr_range,
                            args.context).cuda()

    logger.info(model)
    optimizer = torch.optim.Adam(model.optim_parameters(),
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    model = nn.DataParallel(model).cuda()

    cudnn.enabled = True
    cudnn.benchmark = True
    best_epe_all = 1e9

    if args.pretrain:
        ckpt_name = args.pretrain
        if os.path.isfile(ckpt_name):
            logger.info("=> loading checkpoint '{}'".format(ckpt_name))
            checkpoint = torch.load(ckpt_name)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}'".format(ckpt_name))
        else:
            logger.info("=> no checkpoint found at '{}'".format(ckpt_name))
    elif args.pretrain_base:
        logger.info("=> loading pretrained base model '{}'".format(
            args.pretrain_base))
        base_prefix = "module.hd3net.encoder." if args.encoder!='dlaup' \
                      else "module.hd3net.encoder.base."
        load_module_state_dict(model,
                               torch.load(args.pretrain_base),
                               add=base_prefix)
        logger.info("=> loaded pretrained base model '{}'".format(
            args.pretrain_base))

    ### data loader ###
    train_transform, val_transform = datasets.get_transform(
        args.dataset_name, args.task, args.evaluate)
    train_data = datasets.HD3Data(mode=args.task,
                                  data_root=args.train_root,
                                  data_list=args.train_list,
                                  label_num=1,
                                  transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    if args.evaluate:
        val_data = datasets.HD3Data(mode=args.task,
                                    data_root=args.val_root,
                                    data_list=args.val_list,
                                    label_num=1,
                                    transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size_val,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)

    ### Go! ###
    scheduler = get_lr_scheduler(optimizer, args.dataset_name)
    for epoch in range(1, args.epochs + 1):
        if scheduler is not None:
            scheduler.step()
        loss_train = train(train_loader, model, optimizer, epoch,
                           args.batch_size)
        writer.add_scalar('loss_train', loss_train, epoch)

        is_best = False
        if args.evaluate:
            torch.cuda.empty_cache()
            loss_val, epe_val = validate(val_loader, model)
            writer.add_scalar('loss_val', loss_val, epoch)
            writer.add_scalar('epe_val', epe_val, epoch)
            is_best = epe_val < best_epe_all
            best_epe_all = min(epe_val, best_epe_all)

        filename = os.path.join(args.save_path, 'model_latest.pth')
        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.cpu().state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_epe_all': best_epe_all
            }, filename)
        model.cuda()

        if is_best:
            shutil.copyfile(filename,
                            os.path.join(args.save_path, 'model_best.pth'))

        if epoch % args.save_step == 0:
            shutil.copyfile(
                filename,
                args.save_path + '/train_epoch_' + str(epoch) + '.pth')


def train(train_loader, model, optimizer, epoch, batch_size):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = None

    model.train()
    end = time.time()
    for i, (img_list, label_list) in enumerate(train_loader):
        if img_list[0].shape[0] < batch_size:
            continue
        data_time.update(time.time() - end)
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = args.epochs * len(train_loader)

        img_list = [img.to(torch.device("cuda")) for img in img_list]
        label_list = [label.to(torch.device("cuda")) for label in label_list]

        output = model(img_list=img_list, label_list=label_list, get_loss=True)
        total_loss = output['loss']['total'].sum()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if loss_meter is None:
            loss_meter = {}
            for loss_type, _ in output['loss'].items():
                loss_meter[loss_type] = AverageMeter()
        for loss_type, loss_value in output['loss'].items():
            loss_meter[loss_type].update(loss_value.mean().data,
                                         img_list[0].size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m),
                                                    int(t_s))

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time}.'.format(
                            epoch,
                            args.epochs,
                            i + 1,
                            len(train_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            remain_time=remain_time))
            for loss_type, loss_value in loss_meter.items():
                logger.info('Loss {} {loss_meter.val:.4f} '.format(
                    loss_type, loss_meter=loss_value))

        writer.add_scalar('total_loss_train_batch',
                          loss_meter['total'].val.cpu().numpy(), current_iter)

    return loss_meter['total'].avg.cpu().numpy()


def validate(val_loader, model):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = None
    epe_meter = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (img_list, label_list) in enumerate(val_loader):
            data_time.update(time.time() - end)

            img_list = [img.to(torch.device("cuda")) for img in img_list]
            label_list = [
                label.to(torch.device("cuda")) for label in label_list
            ]

            output = model(img_list=img_list,
                           label_list=label_list,
                           get_loss=True,
                           get_epe=True,
                           get_vis=i % args.visual_freq == 0)

            epe_meter.update(output['epe'].mean().data, img_list[0].size(0))
            if loss_meter is None:
                loss_meter = {}
                for loss_type, _ in output['loss'].items():
                    loss_meter[loss_type] = AverageMeter()
            for loss_type, loss_value in output['loss'].items():
                loss_meter[loss_type].update(loss_value.mean().data,
                                             img_list[0].size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % args.print_freq == 0:
                logger.info(
                    'Test: [{}/{}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'EPE {epe_meter.val:.4f} ({epe_meter.avg:.4f}).'.format(
                        i + 1,
                        len(val_loader),
                        data_time=data_time,
                        batch_time=batch_time,
                        epe_meter=epe_meter))
                for loss_type, loss_value in loss_meter.items():
                    logger.info('Loss {} {loss_meter.val:.4f} '.format(
                        loss_type, loss_meter=loss_value))

            if 'vis' in output.keys():
                for b in range(output['vis'].size(0)):
                    writer.add_image('Visualization_%d' % b, output['vis'][b],
                                     i)

    logger.info(
        'Val result: EPE {epe_meter.avg:.3f}.'.format(epe_meter=epe_meter))

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter['total'].avg.cpu().numpy(), epe_meter.avg


def get_lr_scheduler(optimizer, dataset_name):
    if dataset_name in ['FlyingChairs', 'FlyingThings3D']:
        milestones = [70, 100, 130, 160]
    elif dataset_name == 'KITTI':
        milestones = [1000, 1500]
    elif dataset_name == 'MPISintel':
        milestones = [600, 900]
    else:
        raise ValueError('Unknown dataset name {}'.format(dataset_name))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestones,
                                                     gamma=0.5)
    return scheduler


if __name__ == '__main__':
    main()
