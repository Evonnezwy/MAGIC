import sys
import os
import errno
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.metrics import f1_score

def dice_mean(pred, batch_y, smooth=1.):
    pred = pred.data.cpu().numpy()
    pred_f = np.argmax(pred, axis=1).astype('float32')
    pred_f = pred_f.flatten()
    true_f = batch_y.view([-1, ]).data.cpu().numpy()
    C = pred.shape[1] 
    mean_dice_list = []   
    mean_dice = 0
    for i in range(1, C):
        pred_i = np.zeros(pred_f.shape)
        pred_i[pred_f == i] = 1.
        true_i = np.zeros(true_f.shape)
        true_i[true_f == i] = 1.

        intersection = np.sum(true_i * pred_i)
        dice = (2. * intersection + smooth) / (np.sum(true_i) + np.sum(pred_i) + smooth)
        mean_dice += dice
        mean_dice_list.append(dice)
    mean_dice = mean_dice / (C-1)
    return mean_dice, mean_dice_list 


def cal_dice(pred, batch_y, smooth=1., eachcase_dice=False):
    mean_dice, mean_dice_list = dice_mean(pred, batch_y)
    dice =mean_dice

    if eachcase_dice:
        pred = pred.data.cpu().numpy()
        every_dice = [] 
        B, d1_, d2_ = batch_y.shape
        C = pred.shape[1]
        pred = pred.reshape([B, d1_, d2_, C])
        for i in range(B):
            p = np.argmax(pred[i], axis=-1).astype('float32')
            p_f = p.flatten()
            gt = batch_y[i].view([-1, ]).data.cpu().numpy()
            d = 0
            for i in range(1, C):
                pred_i = np.zeros(p_f.shape)
                pred_i[p_f == i] = 1.
                true_i = np.zeros(gt.shape)
                true_i[gt == i] = 1.
                intersection = np.sum(true_i * pred_i)
                d_ = (2. * intersection + smooth) / (np.sum(true_i) + np.sum(pred_i) + smooth)
                d += d_
            every_dice.append(d/(C-1))
        return dice, every_dice 
    return dice, mean_dice_list


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        count = val.size
        v = val.sum()

        self.count += count
        self.sum += v

        self.avg = self.sum / self.count




