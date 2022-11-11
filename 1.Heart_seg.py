import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np
#import matplotlib.pyplot as plt

import copy
import torchvision
from torchvision import transforms
import os
from torch.optim import lr_scheduler
import glob
from PIL import Image

def load_data(images, annotations):
    images = sorted(images)
    annotations = sorted(annotations)
    np.random.seed(2022)
    index = np.random.permutation(len(images))
    images = np.array(images)[index]
    anno = np.array(annotations)[index]
    test_count = int(len(images) * 0.2)  # 测试数据
    train_count = len(images) - test_count  # 训练数据
    test_images = images[:test_count]
    test_anno = anno[:test_count]
    images = images[test_count:]
    anno = anno[test_count:]
    return images, anno, test_images, test_anno


class Portrait_dataset(data.Dataset):
    def __init__(self, img_paths, anno_paths):
        self.imgs = img_paths
        self.annos = anno_paths

    def __getitem__(self, index):
        img = self.imgs[index]
        anno = self.annos[index]

        pil_img = Image.open(img)
        img_tensor = transform(pil_img)

        pil_anno = Image.open(anno)
        anno_tensor = transform(pil_anno)
        anno_tensor[anno_tensor > 0] = 1
        anno_tensor = torch.squeeze(anno_tensor).type(torch.long)
        anno_tensor = anno_tensor.reshape(1, 512, 512)
        # anno_tensor = anno_tensor.transpose(0,2)
        return img_tensor, anno_tensor

    def __len__(self):
        return len(self.imgs)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv_relu = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels,
                                      kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels,
                                      kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)
            )
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x, is_pool=True):
        if is_pool:
            x = self.pool(x)
        x = self.conv_relu(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(2 * channels, channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(channels,
                               channels // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_relu(x)
        x = self.upconv_relu(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.down1 = Downsample(3, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 1024)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(1024,
                               512,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = Upsample(512)
        self.up2 = Upsample(256)
        self.up3 = Upsample(128)

        self.conv_2 = Downsample(128, 64)
        self.last = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x, is_pool=False)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x5 = self.up(x5)

        x5 = torch.cat([x4, x5], dim=1)  # 32*32*1024
        x5 = self.up1(x5)  # 64*64*256)
        x5 = torch.cat([x3, x5], dim=1)  # 64*64*512
        x5 = self.up2(x5)  # 128*128*128
        x5 = torch.cat([x2, x5], dim=1)  # 128*128*256
        x5 = self.up3(x5)  # 256*256*64
        x5 = torch.cat([x1, x5], dim=1)  # 256*256*128

        x5 = self.conv_2(x5, is_pool=False)  # 256*256*64

        x5 = self.last(x5)  # 256*256*3
        return x5

# Dice Loss
def dice_loss(pred, target, smooth = 1.):
	pred = pred.contiguous()
	target = target.contiguous()

	intersection = (pred * target).sum(dim=2).sum(dim=2)

	loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

	return loss.mean()


def calculate_loss(pred, target):
	pred = torch.sigmoid(pred)
	loss = dice_loss(pred, target)

	return loss


def fit1(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = calculate_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
        print('.', end=' ')
    exp_lr_scheduler.step()
    epoch_loss = running_loss / (len(trainloader.dataset) / 16)

    epoch_acc = correct / (total * 512 * 512)

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = calculate_loss(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
            print('.', end=' ')
    epoch_test_loss = test_running_loss / (len(testloader.dataset) / 16)
    epoch_test_acc = test_correct / (test_total * 512 * 512)

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc, model.state_dict()

def fed_test(epoch, model, testloader1, testloader2):
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()
    with torch.no_grad():
        print('Fed_test')
        for x, y in testloader1:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = calculate_loss(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
            print('.', end=' ')
        for x, y in testloader2:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = calculate_loss(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
            print('.', end=' ')
    epoch_test_loss = test_running_loss / ((len(testloader1.dataset) + len(testloader2.dataset)) / 16)
    epoch_test_acc = test_correct / (test_total * 512 * 512)

    print('epoch: ', epoch,
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
          )

    return epoch_test_loss, epoch_test_acc


if __name__ == '__main__':
    BATCH_SIZE = 16
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    images = glob.glob(r'./data/training/*')
    annotations = glob.glob(r'./data/label/*')
    print(len(images))
    print(len(annotations))
    images, anno, test_images, test_anno = load_data(images, annotations)
    shtrain_dataset = Portrait_dataset(images, anno)
    shtest_dataset = Portrait_dataset(test_images, test_anno)
    shtrain_dl = data.DataLoader(
        shtrain_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    shtest_dl = data.DataLoader(
        shtest_dataset,
        batch_size=BATCH_SIZE,
    )

    model = Net()


    if torch.cuda.is_available():
        model.to('cuda')
        #bxmodel.to('cuda')


    epochs = 300
    PATH = './unetmodel.pth'



    train_loss = []
    train_dice = []
    test_loss = []
    test_dice = []
    max_dice = 0
    for epoch in range(epochs):



        epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc, w  = fit1(epoch,
                                                                        model,
                                                                        shtrain_dl,
                                                                        shtest_dl)


        if (1 - epoch_test_loss) > max_dice:
            max_dice = (1 - epoch_test_loss)
            torch.save(model.state_dict(), PATH,  _use_new_zipfile_serialization=False)
            print('max_dice', 1 - epoch_test_loss)
