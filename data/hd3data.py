from os.path import exists, join, splitext
import numpy as np
from PIL import Image
import utils.flowlib as fl
from torch.utils.data import Dataset
from . import flowtransforms as transforms


class HD3Data(Dataset):
    # Disparity annotations are transformed into flow format;
    # Sparse annotations possess an extra dimension as the valid mask;
    def __init__(self,
                 mode,
                 data_root,
                 data_list,
                 label_num=0,
                 transform=None,
                 out_size=False):
        assert mode in ["flow", "stereo"]
        self.mode = mode
        self.data_root = data_root
        self.data_list = self.read_lists(data_list)
        self.label_num = label_num
        self.transform = transform
        self.out_size = out_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_num = len(self.data_list[index]) - self.label_num
        img_list = []
        label_list = []
        for i, name in enumerate(self.data_list[index]):
            if i < img_num:
                img_list.append(read_gen(join(self.data_root, name), "image"))
            else:
                label = read_gen(join(self.data_root, name), self.mode)
                if self.mode == "stereo":
                    label = fl.disp2flow(label)
                label_list.append(label)
        data = [img_list, label_list]
        data = list(self.transform(*data))

        if self.out_size:
            data.append(np.array(img_list[0].size, dtype=int))

        return tuple(data)

    def read_lists(self, data_list):
        assert exists(data_list)
        samples = [line.strip().split(' ') for line in open(data_list, 'r')]
        return samples


def read_gen(file_name, mode):
    ext = splitext(file_name)[-1]
    if mode == 'image':
        assert ext in ['.png', '.jpeg', '.ppm', '.jpg']
        return Image.open(file_name)
    elif mode == 'flow':
        assert ext in ['.flo', '.png', '.pfm']
        return fl.read_flow(file_name)
    elif mode == 'stereo':
        assert ext in ['.png', '.pfm']
        return fl.read_disp(file_name)
    else:
        raise ValueError('Unknown mode {}'.format(mode))


def get_transform(dataset_name, task, evaluate=True):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    pad_mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]

    val_transform = None
    if dataset_name == 'FlyingChairs':
        train_transform = transforms.Compose([
            transforms.RandomScale([1, 2]),
            transforms.Crop([384, 512], 'rand', pad_mean),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        if evaluate:
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    elif dataset_name == 'FlyingThings3D':
        if task == 'flow':
            train_transform = transforms.Compose([
                transforms.Crop([384, 832], 'rand', pad_mean),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

            if evaluate:
                val_transform = transforms.Compose([
                    transforms.Crop([384, 832], 'center', pad_mean),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
        else:
            train_transform = transforms.Compose([
                transforms.Crop([320, 896], 'rand', pad_mean),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

            if evaluate:
                val_transform = transforms.Compose([
                    transforms.Crop([320, 896], 'center', pad_mean),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

    elif dataset_name == 'KITTI':
        if task == 'flow':
            train_transform = transforms.Compose([
                transforms.MultiScaleRandomCrop([0.5, 1.15], [320, 896],
                                                'nearest'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomPhotometric(
                    noise_stddev=0.0,
                    min_contrast=-0.3,
                    max_contrast=0.3,
                    brightness_stddev=0.02,
                    min_color=0.9,
                    max_color=1.1,
                    min_gamma=0.7,
                    max_gamma=1.5),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.MultiScaleRandomCrop([0.5, 1.15], [320, 896],
                                                'nearest'),
                transforms.ToTensor(),
                transforms.RandomPhotometric(
                    noise_stddev=0.0,
                    min_contrast=-0.3,
                    max_contrast=0.3,
                    brightness_stddev=0.02,
                    min_color=0.9,
                    max_color=1.1,
                    min_gamma=0.7,
                    max_gamma=1.5),
                transforms.Normalize(mean=mean, std=std)
            ])

        if evaluate:
            val_transform = transforms.Compose([
                transforms.Resize([1280, 384], 'nearest'),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    elif dataset_name == 'MPISintel':
        train_transform = transforms.Compose([
            transforms.MultiScaleRandomCrop([0.5, 1.13], [384, 768],
                                            'bilinear'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomPhotometric(
                noise_stddev=0.0,
                min_contrast=-0.3,
                max_contrast=0.3,
                brightness_stddev=0.02,
                min_color=0.9,
                max_color=1.1,
                min_gamma=0.7,
                max_gamma=1.5),
            transforms.Normalize(mean=mean, std=std)
        ])

        if evaluate:
            val_transform = transforms.Compose([
                transforms.Resize([1024, 448], 'bilinear'),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    else:
        raise ValueError('Unknown dataset name {}'.format(dataset_name))

    return train_transform, val_transform
