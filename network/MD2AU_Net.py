import os
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
import torch.nn as nn
from sympy import im
import random
import pandas as pd
import numpy as np
import utils
import argparse
import torch
import time
import datetime
import glob
from sklearn.model_selection import train_test_split

from torch.autograd import Variable 
from MD2AU_Net import ResUnet
from torch.utils.data import DataLoader
from dataloader_final import Generator

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--numcls', default=4, type=int, help="num-class")
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('--max-epoch', default=200, type=int, metavar='N')
parser.add_argument('--loss', default='combined', help='loss type')
parser.add_argument('--batch', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoch', type=int, default=10, help='epoches')
args = parser.parse_args()
args.checkpoint_dir = r'./checkpoint_track'

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)


def main():
    np.random.seed(0)
    os.makedirs(f'save_models/',exist_ok=True)
            
    train_files = list()
    df = pd.read_csv(r'./train_list.csv')
    for ele in range(len(df['train'])):
        gtpaths = glob.glob(os.path.join(df['train'][ele], 'lab', '*lab*'))
        for gtp in gtpaths:
            train_files.append(gtp)

    val_files = list()
    df = pd.read_csv(r'./valid_list.csv')
    for ele in range(len(df['valid'])):
        gtpaths = glob.glob(os.path.join(df['valid'][ele], 'lab', '*lab*'))
        for gtp in gtpaths:
            val_files.append(gtp)


    train_dataset = Generator(train_files, 512)
    val_dataset = Generator(val_files, 512)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch // 2, drop_last=True)

    model = ResUnet(num_classes=args.numcls, pretrained=False)
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

    start_time = time.time()
    train_time = 0
    start_epoch = args.start_epoch
    best_dice = 0

    
    print("==> Start training")
    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, train_loader)
        train_time += round(time.time() - start_train_time)
        loss_val, dice_val = val(epoch, model, criterion, val_loader) 

        if dice_val > best_dice:
            best_dice = dice_val
            best_epoch = epoch
            file_object = open(os.path.join(args.checkpoint_dir, 'logger_bestdice.txt'), 'w')
            file_object.write("Best: epoch-{}, dice-{}".format(best_epoch, best_dice))
            file_object.close()
            
            state = {'state_dict': model.state_dict(), 'epoch': epoch, 'dice': dice_val}
            checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_ep'+str(epoch)+'.pth')
            torch.save(state, checkpoint_path)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Best: epoch-{}, dice-{}".format(best_epoch, best_dice))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion, optimizer, train_loader):
    start_time = time.time()
    model.train()
    loss_averager = utils.AverageMeter()
    dice_averager = utils.AverageMeter()

    for batch_idx, (img, gt) in enumerate(train_loader):
        img = Variable(img).cuda()
        gt = Variable(gt).long().cuda()
        pred = model(img)
        loss = criterion(pred, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_averager.update(loss.data.cpu().numpy())

        dice, dice_list = utils.cal_dice(pred=pred, batch_y=gt, eachcase_dice=False)

        dice_averager.update(dice)
        if batch_idx % 10 == 0: 
            print('epoch{}:\tbatch-{}/{}\tloss-{:.6f}\tdice-{:.6f}'.format(
                epoch, batch_idx, len(train_loader), loss_averager.avg, dice_averager.avg))

    end_time = time.time()
    print('Train:\t\tEpoch {}/{}\tloss-{:.6f}\tdice-{:.6f}\ttime-{:3.2f}'.format(
        epoch, args.max_epoch, loss_averager.avg, dice_averager.avg, end_time-start_time))
    file_object = open(os.path.join(args.checkpoint_dir, 'logger_TrainVal.txt'), 'a')
    file_object.write('Train:\t\tEpoch {}/{}\tloss-{:.6f}\tdice-{:.6f}\ttime-{:3.2f}\n'.format(
        epoch, args.max_epoch, loss_averager.avg, dice_averager.avg, end_time-start_time))
    file_object.close()


def val(epoch, model, criterion, val_loader):
    start_time = time.time()
    model.eval()
    loss_averager = utils.AverageMeter()
    dice_averager = utils.AverageMeter()
    dice_list_total = []
    dice_list_average = [0, 0, 0] 
 
    for batch_idx, (img, gt) in enumerate(val_loader):
        img = Variable(img).cuda()
        gt = Variable(gt).long().cuda()
        pred = model(img)
        loss = criterion(pred, gt)
        loss_averager.update(loss.data.cpu().numpy())
        dice, dice_list = utils.cal_dice(pred=pred, batch_y=gt, eachcase_dice=False)
        dice_averager.update(dice)
        # 这里是计算每个类别的dice
        dice_list_average[0] += dice_list[0]
        dice_list_average[1] += dice_list[1]
        dice_list_average[2] += dice_list[2]
        dice_list_total.extend(dice_list)

    end_time = time.time()
    print('Val:\t\tEpoch {}/{}\tloss-{:.6f}\tdice-{:.6f}\tdices-{}\ttime-{:3.2f}'.format(
        epoch, args.max_epoch, loss_averager.avg, dice_averager.avg, np.asarray(dice_list_average, dtype=np.float32) / len(val_loader), end_time-start_time))
    file_object = open(os.path.join(args.checkpoint_dir, 'logger_TrainVal.txt'), 'a')
    file_object.write('Val:\t\tEpoch {}/{}\tloss-{:.6f}\tdice-{:.6f}\ttime-{:3.2f}\n'.format(
        epoch, args.max_epoch, loss_averager.avg, dice_averager.avg, end_time-start_time))
    file_object.close()

    return loss_averager.avg, dice_averager.avg 

if __name__ == '__main__':
    main()



