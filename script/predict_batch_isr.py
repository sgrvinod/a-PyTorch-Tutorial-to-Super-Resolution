import sys

sys.path.append('N:\\code\\super_resolution\\a-PyTorch-Tutorial-to-Super-Resolution')

import os
import cv2 as cv
from glob import glob
from typing import List, Tuple
from ISR.models import RDN
from ISR.models import RRDN
import torch
from utils import ImageTransforms

device = torch.device("cpu")
rdn = RDN(weights='psnr-large')
rrdn = RRDN(weights='gans')
srgan_checkpoint = "../checkpoint_srgan.pth.tar"
srresnet_checkpoint = "../checkpoint_srresnet.pth.tar"

# Load model, either the SRResNet or the SRGAN
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
srgan_generator.eval()
import numpy as np

imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)

from PIL import Image

IMG = Image.Image

MAX_PX = 250 * 250
transform = ImageTransforms(split='test',
                            crop_size=0,
                            scaling_factor=1,
                            lr_img_type='imagenet-norm',
                            hr_img_type='imagenet-norm')


def resize_lr_img(img: IMG) -> IMG:
    w, h = img.size
    ratio = (w * h / MAX_PX) ** 0.5
    if ratio > 1:
        img = img.resize((int(w / ratio), int(h / ratio)))
    return img


def get_all_img_paths(img_dir: str) -> List[str]:
    types = ('/*.jpg', '/*.jpeg', '/*.png')  # the tuple of file types
    files = []
    for t in types:
        files.extend(glob(img_dir + t))
    return files

def denoise(img: IMG) -> IMG:
    open_cv_image = np.array(img)
    img = cv.cvtColor(open_cv_image, cv.COLOR_BGR2RGB)
    dst = cv.fastNlMeansDenoisingColored(open_cv_image, None, 4, 4, 6, 8)
    return Image.fromarray(dst)

def load_img(img_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    img = Image.open(img_path, mode='r')
    img = img.convert('RGB')
    img = resize_lr_img(img)
    img = denoise(img)
    lr_img, hr_img = transform(img)
    org_img = (hr_img * imagenet_std + imagenet_mean) * 2 - 1
    return org_img, hr_img


def save_tensor(t: torch.Tensor, file_path: str):
    img = tensor_to_img(t)
    save_img(img=img, file_path=file_path)


def tensor_to_img(t: torch.Tensor) -> IMG:
    if len(t.shape) == 4:
        array = (t[0].permute(1, 2, 0).numpy())
    else:
        assert len(t.shape) == 3
        array = (t.permute(1, 2, 0).numpy())
    img = Image.fromarray(np.uint8((array + 1) / 2 * 255))
    return img


def combine_image_horizontally(imgs: List[IMG]) -> IMG:
    max_size = imgs[-1].size
    imgs_comb = np.hstack((np.asarray(i.resize(max_size)) for i in imgs))
    return Image.fromarray(imgs_comb)


def save_img(img: IMG, file_path: str):
    img.save(file_path)


all_img_paths = get_all_img_paths('../image_2')
outdir = 'output_isr/'
os.makedirs(outdir, exist_ok=True)
for i, img_path in enumerate(all_img_paths):
    try:
        org_file_name = os.path.basename(img_path)[::-1].replace('.', '__', 1)[::-1]
        print('processing {}[{}/{}]'.format(org_file_name, i + 1, len(all_img_paths)))
        img = Image.open(img_path, mode='r')
        img = img.convert('RGB')
        img = resize_lr_img(img)
        img = denoise(img)

        save_img(img, outdir + "{}_{}.png".format(org_file_name, 'org'))
        print(img.size)
        lr_img_array = np.array(img)
        print('predict with srresnet')
        sr_img_rdn = Image.fromarray(rdn.predict(lr_img_array))
        save_img((sr_img_rdn), outdir + "{}_{}.png".format(org_file_name, 'srr'))
        print('predict with srgam')
        sr_img_rrdn = Image.fromarray(rrdn.predict(lr_img_array))
        save_img(sr_img_rrdn, outdir + "{}_{}.png".format(org_file_name, 'srg'))
        save_img(combine_image_horizontally([img, sr_img_rdn, sr_img_rrdn]),
                 outdir + '{}_com.png'.format(org_file_name))
    except Exception as e:
        print(e)