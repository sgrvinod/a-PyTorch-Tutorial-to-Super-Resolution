import json
import os
import random
from glob import glob
from typing import List

import numpy as np
import scipy.stats as stats
import torch
import torchvision.transforms.functional as FT
from PIL import Image
from torchvision import transforms

from processing.add_noise.degrade_funcs import jpeg_blur

IMG = Image.Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some constants
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def create_data_lists(train_folders, test_folders, min_size, output_folder):
    """
    Create lists for images in the training set and each of the test sets.

    :param train_folders: folders containing the training images; these will be merged
    :param test_folders: folders containing the test images; each test folder will form its own test set
    :param min_size: minimum width and height of images to be considered
    :param output_folder: save data lists here
    """
    print("\nCreating data lists... this may take some time.\n")
    train_images = list()
    for d in train_folders:
        for i in os.listdir(d):
            if i.endswith('.png') or i.endswith('.jpg') or i.endswith('jpeg'):
                img_path = os.path.join(d, i)
                img = Image.open(img_path, mode='r')
                if img.width >= min_size and img.height >= min_size:
                    train_images.append(img_path)
    print("There are %d images in the training data.\n" % len(train_images))
    with open(os.path.join(output_folder, 'train_images.json'), 'w') as j:
        json.dump(train_images, j)

    for d in test_folders:
        test_images = list()
        test_name = d.split("/")[-1]
        for i in os.listdir(d):
            if i.endswith('.png') or i.endswith('.jpg') or i.endswith('jpeg'):
                img_path = os.path.join(d, i)
                img = Image.open(img_path, mode='r')
                if img.width >= min_size and img.height >= min_size:
                    test_images.append(img_path)
        print("There are %d images in the %s test data.\n" % (len(test_images), test_name))
        with open(os.path.join(output_folder, test_name + '_test_images.json'), 'w') as j:
            json.dump(test_images, j)

    print("JSONS containing lists of Train and Test images have been saved to %s\n" % output_folder)


def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img


class ImageTransforms(object):
    """
    Image transformation pipeline.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be downsampled from the HR images by this factor
        :param lr_img_type: the target format for the LR image; see convert_image() above for available formats
        :param hr_img_type: the target format for the HR image; see convert_image() above for available formats
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.downsample_methods = [Image.NEAREST, Image.BOX,
                                   Image.BILINEAR, Image.HAMMING,
                                   Image.BICUBIC, Image.LANCZOS]
        self.downsample_proba = [0.1, 0.1, 0.2, 0.1, 0.3, 0.2]
        jpeg_mean, jpeg_std = 50, 25

        self.jpeg_quality_dist = self._jpeg_quality_dist(mean=50, std=25, lower=1, upper=100)
        # weights = scipy.io.loadmat(path.join('./processing/jpeg_artifacts/weights/q{}.mat'.format(40)))
        # self.denoiser = ARCNN(weights).to("cpu").eval()
        assert self.split in {'train', 'test'}
        self.augumentor = transforms.RandomOrder(
            [
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
                transforms.RandomGrayscale(p=0.15),
                transforms.RandomHorizontalFlip(0.25)
            ]

        )

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """

        # Crop
        if self.split == 'train':
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            super_crop = False
            if random.random() > 0.67:
                super_crop = True
                real_crop_size = int((random.random() + 2) * self.crop_size)
            else:
                real_crop_size = self.crop_size
            left = random.randint(1, img.width - real_crop_size)
            top = random.randint(1, img.height - real_crop_size)
            right = left + real_crop_size
            bottom = top + real_crop_size
            hr_img = img.crop((left, top, right, bottom))
            if super_crop:
                hr_img = hr_img.resize((self.crop_size, self.crop_size))
            hr_img = self.augumentor(hr_img)
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        resize_method = np.random.choice(self.downsample_methods, p=self.downsample_proba)
        new_w, new_h = int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)
        lr_img = hr_img.resize((new_w, new_h), resize_method)
        # Add Jpeg artifacts to lr_img
        if random.random() > 0.25:
            quality = self.jpeg_quality_dist.rvs()
            lr_img = jpeg_blur(img=lr_img, q=quality)

        # Sanity check
        assert hr_img.width == lr_img.width * self.scaling_factor
        assert hr_img.height == lr_img.height * self.scaling_factor

        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)
        return lr_img, hr_img

    def _jpeg_quality_dist(self, mean: float, std: float, lower: float, upper: float):
        return stats.truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
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


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    """
    Save model checkpoint.

    :param state: checkpoint contents
    """

    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def tensor_to_img(t: torch.Tensor) -> IMG:
    if len(t.shape) == 4:
        array = (t[0].permute(1, 2, 0).numpy())
    else:
        assert len(t.shape) == 3
        array = (t.permute(1, 2, 0).numpy())
    img = Image.fromarray(np.uint8((array + 1) / 2 * 255))
    return img


def combine_image_horizontally(imgs: List[IMG]) -> IMG:
    max_h = max([img.height for img in imgs])
    arrays = []
    for i, img in enumerate(imgs):
        if i % 2 == 0:
            img = img.resize(imgs[i + 1].size)
        new_img = Image.new("RGB", (img.width, max_h))
        new_img.paste(img)
        arrays.append(np.asarray(new_img))
    imgs_comb = np.hstack(arrays)
    return Image.fromarray(imgs_comb)


def save_img(img: IMG, file_path: str):
    img.save(file_path)


def tensor_to_np(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def get_all_img_files(img_dir: str):
    img_type = {'png', 'jpg', 'jpeg'}
    return sum([glob('{}/*{}'.format(img_dir, t)) for t in img_type], [])
