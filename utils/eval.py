import os
import pandas as pd
import numpy as np
from regex import F
import utils
import argparse
import torch
torch.cuda.set_device(0)
import time
import glob
import datetime
import torch.nn as nn
from torch.autograd import Variable

from MD2AU_Net import ResUnet
from torch.utils.data import DataLoader
from dataloader_final import Generator
from PIL import Image, ImageFile, ImageEnhance
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--numcls', default=4, type=int, help="num-class")
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('--max-epoch', default=200, type=int, metavar='N')
parser.add_argument('--loss', default='combined', help='loss type')
parser.add_argument('--batch', type=int, default=2, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoch', type=int, default=10, help='epoches')
args = parser.parse_args()
args.checkpoint_dir = './checkpoint_track'
args.resume = r'  '
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

def main():
    train_df = pd.read_csv(r'./train_list.csv')
    val_df = pd.read_csv(r'./valid_list.csv')

    train_files_task1 = train_df[train_df['train'].str.contains('task1')]['train'].tolist()
    train_files_task1 = [os.path.join(path, 'lab', '*lab*') for path in train_files_task1]
    train_files_task1 = [item for sublist in train_files_task1 for item in glob.glob(sublist)]

    val_files_task1 = val_df[val_df['valid'].str.contains('task1')]['valid'].tolist()
    val_files_task1 = [os.path.join(path, 'lab', '*lab*') for path in val_files_task1]
    val_files_task1 = [item for sublist in val_files_task1 for item in glob.glob(sublist)]

    train_files_task2 = train_df[train_df['train'].str.contains('task2')]['train'].tolist()
    train_files_task2 = [os.path.join(path, 'lab', '*lab*') for path in train_files_task2]
    train_files_task2 = [item for sublist in train_files_task2 for item in glob.glob(sublist)]

    val_files_task2 = val_df[val_df['valid'].str.contains('task2')]['valid'].tolist()
    val_files_task2 = [os.path.join(path, 'lab', '*lab*') for path in val_files_task2]
    val_files_task2 = [item for sublist in val_files_task2 for item in glob.glob(sublist)]

    # train_files = train_files_task1
    # val_files = val_files_task1
 
    train_files = train_files_task2
    val_files = val_files_task2

    train_dataset = Generator(train_files, 512)
    val_dataset = Generator(val_files, 512)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch // 2, drop_last=True)

    model = ResUnet(num_classes=args.numcls, pretrained=False)
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()


    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

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
        train_time += round(time.time() - start_train_time)
        loss_val, dice_val = val(epoch, model, criterion, val_files) 

        state = {'state_dict': model.state_dict(), 'epoch': epoch, 'dice': dice_val}
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_ep'+str(epoch)+'.pth')
        torch.save(state, checkpoint_path)

        if dice_val > best_dice:
            best_dice = dice_val
            best_epoch = epoch
            file_object = open(os.path.join(args.checkpoint_dir, 'logger_bestdice.txt'), 'w')
            file_object.write("Best: epoch-{}, dice-{}".format(best_epoch, best_dice))
            file_object.close()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Best: epoch-{}, dice-{}".format(best_epoch, best_dice))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def val(epoch, model, criterion, val_files):
    start_time = time.time()
    model.eval()
    loss_averager = utils.AverageMeter()
    dice_averager = utils.AverageMeter()
    dice_list_total = []
    dice_list_average = [0, 0, 0]

    for idx in range(len(val_files)):
        gt = Image.open(val_files[idx])
        gt = gt.resize((512, 512), Image.NEAREST)
        gt = np.asarray(gt, dtype=np.float32)
        gt = torch.from_numpy(gt / 255. * 3).long()

        img = Image.open(val_files[idx].replace('lab', 'img'))
        transform = transforms.Compose([transforms.Resize((512, 512)),
                                        transforms.ToTensor()])
        
        img = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
            gt = gt.unsqueeze(0).cuda()

        pred = model(img)
        pred_ = pred.data.cpu().numpy()
        np_pred = np.argmax(pred_, axis=1) 
        
        loss = criterion(pred, gt)

        loss_averager.update(loss.data.cpu().numpy())

        dice, dice_list = utils.cal_dice(pred=pred, batch_y=gt, eachcase_dice=False)
        dice_averager.update(dice)
        dice_list_average[0] += dice_list[0]
        dice_list_average[1] += dice_list[1]
        dice_list_average[2] += dice_list[2]
        dice_list_total.extend(dice_list)

    end_time = time.time()
    print('Val:\t\tEpoch {}/{}\tloss-{:.6f}\tdice-{:.6f}\tdices-{}\ttime-{:3.2f}'.format(
        epoch, args.max_epoch, loss_averager.avg, dice_averager.avg, np.asarray(dice_list_average, dtype=np.float32) / len(val_files), end_time-start_time))
    file_object = open(os.path.join(args.checkpoint_dir, 'logger_TrainVal.txt'), 'a')
    file_object.write('Val:\t\tEpoch {}/{}\tloss-{:.6f}\tdice-{:.6f}\ttime-{:3.2f}\n'.format(
        epoch, args.max_epoch, loss_averager.avg, dice_averager.avg, end_time-start_time))
    file_object.close()

    return loss_averager.avg, dice_averager.avg



if __name__ == '__main__':
    main()
