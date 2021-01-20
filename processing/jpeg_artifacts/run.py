import os
import time
from argparse import ArgumentParser
from os import path

import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader

import model
from processing.jpeg_artifacts.loader import YcbCrLoader, ycbcr2bgr


def tensor_to_np(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)


parser = ArgumentParser()
parser.add_argument('--dir', help='Directory containing input images', default=path.join('.', './test_imgs'))
parser.add_argument('--output', help='Output directory', default=path.join('.', './results'))
parser.add_argument('--batch_size', help='Batch size to be used.', default=16, type=int)
parser.add_argument('--quality', help='Use the pretrained model trained with (10/20/30/40) quality', default=40,
                    type=int)

args = parser.parse_args()

# Load .mat weights
weights = scipy.io.loadmat(path.join('weights', 'q%d.mat' % args.quality))
for k, v in weights.items():
    if '__' not in k:  # Unwanted mega attributes start with __
        print(k, v.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = YcbCrLoader(args.dir)
loader = DataLoader(dataset, batch_size=args.batch_size)
os.makedirs(args.output, exist_ok=True)

net = model.ARCNN(weights).to(device).eval()

start_time = time.time()
with torch.no_grad():
    for name, im in loader:
        im = im.to(device)
        result = net(im[:, 0:1, :, :])  # We only take the Y channel
        comb_result = torch.cat((result, im[:, 1:3, :, :]), 1)

        for i in range(result.shape[0]):
            cv2.imwrite(path.join(args.output, name[i].replace('.jpg', '.png')),
                        (ycbcr2bgr(
                            tensor_to_np(comb_result[i])
                        ) * 255 + 0.5).astype(np.int32))

end_time = time.time()
print('Time taken: %f, per image: %f' % (end_time - start_time, (end_time - start_time) / len(dataset)))
