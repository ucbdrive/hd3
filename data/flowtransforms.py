import math
import numbers
import random
import collections
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import scipy.ndimage as ndimage
import torch
import utils.flowlib as fl


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range
    [0.0, 1.0]."""

    def imgToTensor(self, img):
        if isinstance(img, Image.Image):
            img = np.asarray(img)
        elif not isinstance(img, np.ndarray):
            raise (RuntimeError(
                "flowtransforms.ToTensor() only handle PIL Image and np.ndarray"
                "[eg: data readed by PIL.Image.open()].\n"))
        if len(img.shape) > 3 or len(img.shape) < 2:
            raise (RuntimeError(
                "flowtransforms.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"
            ))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        return img.div_(255)

    def labelToTensor(self, label):
        if not isinstance(label, np.ndarray) or len(label.shape) != 3:
            raise (RuntimeError(
                "flowtransforms.ToTensor() only handle np.ndarray with 3 dims label.\n"
            ))
        return torch.from_numpy(label.transpose(2, 0, 1)).float()

    def __call__(self, img_list, label_list):
        img_list = [self.imgToTensor(img) for img in img_list]
        label_list = [self.labelToTensor(label) for label in label_list]
        return img_list, label_list


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, img_list, label_list):
        for img in img_list:
            for t, m, s in zip(img, self.mean, self.std):
                t.sub_(m).div_(s)
        return img_list, label_list


class Resize(object):
    """
    Resize the input PIL Image to the given size.
    'size' is a 2-element tuple or list in the order of (w, h)
    """

    def __init__(self, size, method='bilinear'):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.method = method

    def __call__(self, img_list, label_list):
        img_list = [img.resize(self.size, Image.BILINEAR) for img in img_list]
        label_list = [
            fl.resize_flow(label, self.size[0], self.size[1], self.method)
            for label in label_list
        ]
        return img_list, label_list


class RandomScale(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    *** WARNING *** Scale operation is dangerous for sparse optical flow groundtruth
    """

    def __init__(self, scale, aspect_ratio=None, method='bilinear'):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("transforms.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError(
                "transforms.RandScale() aspect_ratio param error.\n"))
        self.method = method

    def __call__(self, img_list, label_list):
        temp_scale = self.scale[0] + (self.scale[1] -
                                      self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (
                self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale / temp_aspect_ratio
        w, h = img_list[0].size
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)
        img_list = [
            img.resize((new_w, new_h), Image.BILINEAR) for img in img_list
        ]
        label_list = [
            fl.resize_flow(label, new_w, new_h, self.method)
            for label in label_list
        ]
        return img_list, label_list


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img_list, label_list):
        if random.random() < 0.5:
            img_list = [
                img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_list
            ]
            label_list = [
                fl.horizontal_flip_flow(label) for label in label_list
            ]
        return img_list, label_list


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img_list, label_list):
        if random.random() < 0.5:
            img_list = [
                img.transpose(Image.FLIP_TOP_BOTTOM) for img in img_list
            ]
            label_list = [fl.vertical_flip_flow(label) for label in label_list]
        return img_list, label_list


class RandomGaussianBlur(object):

    def __call__(self, img_list, label_list, radius=2):
        if random.random() < 0.5:
            img_list = [
                img.filter(ImageFilter.GaussianBlur(radius))
                for img in img_list
            ]
        return img_list, label_list


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, img_list, label_list):
        img_list = [Image.merge('RGB', img.split()[::-1]) for img in img_list]
        return img_list, label_list


class MultiScaleRandomCrop(object):

    def __init__(self, scale, size, method='nearest'):
        self.crop_h = size[0]
        self.crop_w = size[1]
        self.scale = scale
        self.method = method

    def __call__(self, img_list, label_list):
        temp_scale = self.scale[0] + (self.scale[1] -
                                      self.scale[0]) * random.random()
        new_crop_h = int(self.crop_h * temp_scale)
        new_crop_w = int(self.crop_w * temp_scale)

        w, h = img_list[0].size
        h_off = random.randint(0, h - new_crop_h)
        w_off = random.randint(0, w - new_crop_w)

        img_list = [
            img.crop((w_off, h_off, w_off + new_crop_w, h_off + new_crop_h))
            for img in img_list
        ]
        label_list = [
            label[h_off:h_off + new_crop_h, w_off:w_off + new_crop_w, :]
            for label in label_list
        ]

        img_list = [
            img.resize((self.crop_w, self.crop_h), Image.BILINEAR)
            for img in img_list
        ]
        label_list = [
            fl.resize_flow(label, self.crop_w, self.crop_h, self.method)
            for label in label_list
        ]

        return img_list, label_list


class Crop(object):
    """Crops the given PIL Image.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """

    def __init__(self, size, crop_type='center', padding=None):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError(
                    "padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))

    def __call__(self, img_list, label_list):
        w, h = img_list[0].size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError(
                    "flowtransforms.Crop() need padding while padding argument is None\n"
                ))
            border = (pad_w_half, pad_h_half, pad_w - pad_w_half,
                      pad_h - pad_h_half)
            img_list = [
                ImageOps.expand(
                    img,
                    border=border,
                    fill=tuple([int(item) for item in self.padding]))
                for img in img_list
            ]
            if len(label_list) > 0:
                if label_list[0].shape[2] == 3:
                    label_list = [
                        np.pad(label,
                               ((pad_h_half, pad_h - pad_h_half),
                                (pad_w_half, pad_w - pad_w_half), (0, 0)),
                               'constant') for label in label_list
                    ]
                else:
                    raise (RuntimeError(
                        "Cropping to larger size not supported for optical flow without mask.\n"
                    ))
        w, h = img_list[0].size
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = (h - self.crop_h) // 2
            w_off = (w - self.crop_w) // 2
        img_list = [
            img.crop((w_off, h_off, w_off + self.crop_w, h_off + self.crop_h))
            for img in img_list
        ]
        label_list = [
            label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w, :]
            for label in label_list
        ]
        return img_list, label_list


class PadToSize(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""

    def __init__(self, side=0, crop_w=0, crop_h=0, fill=-1):
        assert isinstance(side, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
            isinstance(fill, tuple)
        assert isinstance(crop_w, numbers.Number)
        assert isinstance(crop_h, numbers.Number)
        assert not (side > 0 and (crop_w > 0 or crop_h > 0))

        if side != 0:
            self.side_h = side
            self.side_w = side
        else:
            self.side_h = crop_h
            self.side_w = crop_w
        self.fill = fill

    def __call__(self, img_list, label_list):
        w, h = img_list[0].size
        sw = self.side_w
        sh = self.side_h
        assert (sw >= w and sh >= h)

        top, left = (sh - h) // 2, (sw - w) // 2
        bottom = sh - h - top
        right = sw - w - left

        if len(label_list) > 0:
            if label_list[0].shape[2] == 3:
                label_list = [
                    np.pad(label, ((top, bottom), (left, right), (0, 0)),
                           'constant') for label in label_list
                ]
            else:
                raise (RuntimeError(
                    "Cropping to larger size not supported for optical flow without mask.\n"
                ))
        if self.fill == -1:
            img_list = [
                pad_image('reflection', img, top, bottom, left, right)
                for img in img_list
            ]
        else:
            img_list = [
                pad_image(
                    'constant', img, top, bottom, left, right, value=self.fill)
                for img in img_list
            ]

        return img_list, label_list


def pad_reflection(image, top, bottom, left, right):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    next_top = next_bottom = next_left = next_right = 0
    if top > h - 1:
        next_top = top - h + 1
        top = h - 1
    if bottom > h - 1:
        next_bottom = bottom - h + 1
        bottom = h - 1
    if left > w - 1:
        next_left = left - w + 1
        left = w - 1
    if right > w - 1:
        next_right = right - w + 1
        right = w - 1
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image[top:top + h, left:left + w] = image
    new_image[:top, left:left + w] = image[top:0:-1, :]
    new_image[top + h:, left:left + w] = image[-1:-bottom - 1:-1, :]
    new_image[:, :left] = new_image[:, left * 2:left:-1]
    new_image[:, left + w:] = new_image[:, -right - 1:-right * 2 - 1:-1]
    return pad_reflection(new_image, next_top, next_bottom, next_left,
                          next_right)


def pad_constant(image, top, bottom, left, right, value):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image.fill(value)
    new_image[top:top + h, left:left + w] = image
    return new_image


def pad_image(mode, image, top, bottom, left, right, value=0):
    if mode == 'reflection':
        return Image.fromarray(
            pad_reflection(np.asarray(image), top, bottom, left, right))
    elif mode == 'constant':
        return Image.fromarray(
            pad_constant(np.asarray(image), top, bottom, left, right, value))
    else:
        raise ValueError('Unknown mode {}'.format(mode))


class RandomRotate(object):
    """
        Random rotation of the image from -angle to angle (in degrees)
        This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
        angle: max angle of the rotation
        interpolation order: Default: 2 (bilinear)
        reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
        diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, img_list, label_list):
        assert len(img_list) == 2 and len(label_list) <= 1
        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)
        angle1 = applied_angle - diff / 2
        angle2 = applied_angle + diff / 2
        angle1_rad = angle1 * np.pi / 180

        img_list[0] = ndimage.interpolation.rotate(
            np.asarray(img_list[0]),
            angle1,
            reshape=self.reshape,
            order=self.order)
        img_list[1] = ndimage.interpolation.rotate(
            np.asarray(img_list[1]),
            angle2,
            reshape=self.reshape,
            order=self.order)
        img_list[0] = Image.fromarray(img_list[0])
        img_list[1] = Image.fromarray(img_list[1])

        if len(label_list) == 1:
            target = label_list[0]
            h, w, c = target.shape
            assert c == 2

            def rotate_flow(i, j, k):
                return -k * (j - w / 2) * (diff * np.pi /
                                           180) + (1 - k) * (i - h / 2) * (
                                               diff * np.pi / 180)

            rotate_flow_map = np.fromfunction(rotate_flow, target.shape)
            target += rotate_flow_map
            target = ndimage.interpolation.rotate(
                target, angle1, reshape=self.reshape, order=self.order)
            # flow vectors must be rotated too! careful about Y flow which is upside down
            target_ = np.copy(target)
            target[:, :, 0] = np.cos(angle1_rad) * target_[:, :, 0] + np.sin(
                angle1_rad) * target_[:, :, 1]
            target[:, :, 1] = -np.sin(angle1_rad) * target_[:, :, 0] + np.cos(
                angle1_rad) * target_[:, :, 1]
            label_list = [target]
        return img_list, label_list


class RandomTranslate(object):

    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, img_list, label_list):
        assert len(img_list) == 2 and len(label_list) <= 1
        w, h = img_list[0].size
        th, tw = self.translation
        tw = int(random.uniform(-tw, tw) * w / 100)
        th = int(random.uniform(-th, th) * h / 100)
        if tw == 0 and th == 0:
            return img_list, label_list
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1, x2, x3, x4 = max(0, tw), min(w + tw, w), max(0,
                                                         -tw), min(w - tw, w)
        y1, y2, y3, y4 = max(0, th), min(h + th, h), max(0,
                                                         -th), min(h - th, h)

        img_list[0] = img_list[0].crop((x1, y1, x2, y2))
        img_list[1] = img_list[1].crop((x3, y3, x4, y4))

        if len(label_list) == 1:
            label_list[0] = label_list[0][y1:y2, x1:x2]
            label_list[0][:, :, 0] += tw
            label_list[0][:, :, 1] += th

        return img_list, label_list


class RandomPhotometric(object):
    """Applies photometric augmentations to a list of image tensors.
    Each image in the list is augmented in the same way.

    Args:
        ims: list of 3-channel images normalized to [0, 1].

    Returns:
        normalized images with photometric augmentations. Has the same
        shape as the input.
    """

    def __init__(self,
                 noise_stddev=0.0,
                 min_contrast=0.0,
                 max_contrast=0.0,
                 brightness_stddev=0.0,
                 min_color=1.0,
                 max_color=1.0,
                 min_gamma=1.0,
                 max_gamma=1.0):
        self.noise_stddev = noise_stddev
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.brightness_stddev = brightness_stddev
        self.min_color = min_color
        self.max_color = max_color
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, ims, label_list=None):
        contrast = np.random.uniform(self.min_contrast, self.max_contrast)
        gamma = np.random.uniform(self.min_gamma, self.max_gamma)
        gamma_inv = 1.0 / gamma
        color = torch.from_numpy(
            np.random.uniform(self.min_color, self.max_color, (3))).float()
        if self.noise_stddev > 0.0:
            noise = np.random.normal(scale=self.noise_stddev)
        else:
            noise = 0
        if self.brightness_stddev > 0.0:
            brightness = np.random.normal(scale=self.brightness_stddev)
        else:
            brightness = 0

        out = []
        for im in ims:
            im_re = im.permute(1, 2, 0)
            im_re = (im_re * (contrast + 1.0) + brightness) * color
            im_re = torch.clamp(im_re, min=0.0, max=1.0)
            im_re = torch.pow(im_re, gamma_inv)
            im_re += noise

            im = im_re.permute(2, 0, 1)
            out.append(im)

        return out, label_list
