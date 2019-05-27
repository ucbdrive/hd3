import os
from os.path import join
import cv2
import time
import math
import logging
from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import data.hd3data as datasets
import data.flowtransforms as transforms
import hd3model as models
from utils.utils import *
from models.hd3_ops import *
import utils.flowlib as fl


# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch HD^3 Evaluation')
    parser.add_argument('--task', type=str, help='stereo or flow')
    parser.add_argument('--encoder', type=str, help='vgg or dlaup')
    parser.add_argument('--decoder', type=str, help='resnet, or hda')
    parser.add_argument('--context', action='store_true', default=False)
    parser.add_argument('--data_root', type=str, help='data root')
    parser.add_argument('--data_list', type=str, help='data list')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help=
        'batch size larger than 1 may have issues when input images have different sizes'
    )
    parser.add_argument(
        '--workers', type=int, default=8, help='data loader workers')
    parser.add_argument('--model_path', type=str, help='evaluation model path')
    parser.add_argument('--save_folder', type=str, help='results save folder')
    parser.add_argument(
        '--flow_format',
        type=str,
        default='png',
        help='saved flow format, png or flo')
    parser.add_argument('--evaluate', action='store_true', default=False)
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


def get_target_size(H, W):
    h = 64 * np.array([[math.floor(H / 64), math.floor(H / 64) + 1]])
    w = 64 * np.array([[math.floor(W / 64), math.floor(W / 64) + 1]])
    ratio = np.abs(np.matmul(np.transpose(h), 1 / w) - H / W)
    index = np.argmin(ratio)
    return h[0, index // 2], w[0, index % 2]


def main():
    global args, logger
    args = get_parser().parse_args()
    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")

    # get input image size and save name list
    # each line of data_list should contain image_0, image_1, (optional gt)
    with open(args.data_list, 'r') as f:
        fnames = f.readlines()
        assert len(fnames[0].strip().split(' ')) == 2 + args.evaluate
        names = [l.strip().split(' ')[0].split('/')[-1] for l in fnames]
        sub_folders = [
            l.strip().split(' ')[0][:-len(names[i])]
            for i, l in enumerate(fnames)
        ]
        names = [l.split('.')[0] for l in names]
        input_size = cv2.imread(join(args.data_root,
                                     fnames[0].split(' ')[0])).shape

    # transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    th, tw = get_target_size(input_size[0], input_size[1])
    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)])
    val_data = datasets.HD3Data(
        mode=args.task,
        data_root=args.data_root,
        data_list=args.data_list,
        label_num=args.evaluate,
        transform=val_transform,
        out_size=True)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    corr_range = [4, 4, 4, 4, 4, 4]
    if args.task == 'flow':
        corr_range = corr_range[:5]
    model = models.HD3Model(args.task, args.encoder, args.decoder, corr_range,
                            args.context).cuda()
    logger.info(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.enabled = True
    cudnn.benchmark = True

    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(
            args.model_path))

    vis_folder = os.path.join(args.save_folder, 'vis')
    vec_folder = os.path.join(args.save_folder, 'vec')
    check_makedirs(vis_folder)
    check_makedirs(vec_folder)

    # start testing
    logger.info('>>>>>>>>>>>>>>>> Start Test >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    avg_epe = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (img_list, label_list, img_size) in enumerate(val_loader):
            data_time.update(time.time() - end)

            img_size = img_size.cpu().numpy()
            img_list = [img.to(torch.device("cuda")) for img in img_list]
            label_list = [
                label.to(torch.device("cuda")) for label in label_list
            ]

            # resize test
            resized_img_list = [
                F.interpolate(
                    img, (th, tw), mode='bilinear', align_corners=True)
                for img in img_list
            ]
            output = model(
                img_list=resized_img_list,
                label_list=label_list,
                get_vect=True,
                get_epe=args.evaluate)
            scale_factor = 1 / 2**(7 - len(corr_range))
            output['vect'] = resize_dense_vector(output['vect'] * scale_factor,
                                                 img_size[0, 1],
                                                 img_size[0, 0])

            if args.evaluate:
                avg_epe.update(output['epe'].mean().data, img_list[0].size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 10 == 0:
                logger.info(
                    'Test: [{}/{}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.
                    format(
                        i + 1,
                        len(val_loader),
                        data_time=data_time,
                        batch_time=batch_time))

            pred_vect = output['vect'].data.cpu().numpy()
            pred_vect = np.transpose(pred_vect, (0, 2, 3, 1))
            curr_bs = pred_vect.shape[0]

            for idx in range(curr_bs):
                curr_idx = i * args.batch_size + idx
                curr_vect = pred_vect[idx]

                # make folders
                vis_sub_folder = join(vis_folder, sub_folders[curr_idx])
                vec_sub_folder = join(vec_folder, sub_folders[curr_idx])
                check_makedirs(vis_sub_folder)
                check_makedirs(vec_sub_folder)

                # save visualzation (disparity transformed to flow here)
                vis_fn = join(vis_sub_folder, names[curr_idx] + '.png')
                if args.task == 'flow':
                    vis_flo = fl.flow_to_image(curr_vect)
                else:
                    vis_flo = fl.flow_to_image(fl.disp2flow(curr_vect))
                vis_flo = cv2.cvtColor(vis_flo, cv2.COLOR_RGB2BGR)
                cv2.imwrite(vis_fn, vis_flo)

                # save point estimates
                fn_suffix = 'png'
                if args.task == 'flow':
                    fn_suffix = args.flow_format
                vect_fn = join(vec_sub_folder,
                               names[curr_idx] + '.' + fn_suffix)
                if args.task == 'flow':
                    if fn_suffix == 'png':
                        # save png format flow
                        mask_blob = np.ones(
                            (img_size[idx][1], img_size[idx][0]),
                            dtype=np.uint16)
                        fl.write_kitti_png_file(vect_fn, curr_vect, mask_blob)
                    else:
                        # save flo format flow
                        fl.write_flow(curr_vect, vect_fn)
                else:
                    # save disparity map
                    cv2.imwrite(vect_fn,
                                np.uint16(-curr_vect[:, :, 0] * 256.0))

    if args.evaluate:
        logger.info('Average End Point Error {avg_epe.avg:.2f}'.format(
            avg_epe=avg_epe))

    logger.info('<<<<<<<<<<<<<<<<< End Test <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
