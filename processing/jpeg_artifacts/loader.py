import os
from os import path

from torch.utils.data.dataset import Dataset
import numpy as np
import cv2

# Taken from StackOverflow: https://stackoverflow.com/questions/26480125/how-to-get-the-same-output-of-rgb2ycbcr-matlab-function-in-python-opencv
def bgr2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def ycbcr2bgr(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*255.0-16)/(235-16) #to [0, 1]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*255.0-16)/(240-16) #to [0, 1]
    im_ycrcb = im_ycbcr[:,:,(0,2,1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2BGR)
    return im_rgb

class YcbCrLoader(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = os.listdir(root)

    def __getitem__(self, idx):
        im = cv2.imread(path.join(self.root, self.imgs[idx]))

        ycbcr = bgr2ycbcr(im.astype(np.float32)/255).transpose(2, 0, 1)

        return self.imgs[idx], ycbcr

    def __len__(self):
        return len(self.imgs)