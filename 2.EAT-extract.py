import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt


import torchvision
from torchvision import transforms
import os

import glob
from PIL import Image
import cv2
from numpy import *
import SimpleITK as sitk
def setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols):
    img_temp = img_data
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in np.arange(rows):
        for j in np.arange(cols):
            img_temp[i, j] = int((img_temp[i, j]-min)*dFactor)

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255

    return img_temp

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

def midop_1(volarray, label):


    volarray[volarray <= -190] = 0
    volarray[volarray >= -30] = 0
    volarray[volarray <= -30] = 1
    # 生成阈值图像
    threshold = sitk.GetImageFromArray(volarray)
    #threshold.SetSpacing(spacing)
    sitk_median = sitk.MedianImageFilter()
    sitk_median.SetRadius(2)
    sitk_median = sitk_median.Execute(threshold)
    a = sitk.GetArrayFromImage(sitk_median)
    #plt.imshow(a)
    #ex4 = a.reshape((512, 512, -1))

    masked = cv2.bitwise_and(a, a, mask=np.uint8(label.squeeze()))
    return masked

if __name__ == '__main__':
    BATCH_SIZE = 1
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    model = Net()
    #fedmodel.load_state_dict(torch.load(r'D:\pythoncode\jptcode\chest-ct-heart\shtorchmodel\fed\fed_unetbnmodel.pth'))
    model.load_state_dict(torch.load(r'./unetmodel.pth'))
    #path = glob.glob(r'F:\data\378dnpy\*_x.npy')
    path = glob.glob(r'./pred/*_x.npy')
    for i in range(len(path)):
        print(path[i])
        print('start' + str(i))
        pat = np.load(path[i])
        print(pat.shape)
        pat1 = np.load(path[i])
        size = pat.shape[2]
        res = []
        for j in range(size):
            #调整窗宽窗位
            img = setDicomWinWidthWinCenter(pat[:, :, j], 400, 40, 512, 512)
            #改为torch适配格式
            img = np.array(img)
            img = np.expand_dims(img, axis=-1)
            img = np.concatenate((img, img, img), axis=-1)
            img_tensor = Image.fromarray(np.uint8(img))
            img_tensor = transform(img_tensor)
            img_tensor = torch.unsqueeze(img_tensor, dim=0)
            #预测
            model.eval()
            with torch.no_grad():
                pred_mask = model(img_tensor)
                pred = torch.sigmoid(pred_mask)
                pred = pred[0].permute(1, 2, 0).detach().cpu().numpy()
                pred[pred < 0.5] = 0
                pred[pred > 0.5] = 1
            #心脏分割结果pred[512, 512, 1]
            #EAT提取
            masked = midop_1(pat1[:, :, j], pred)
            res.append(masked)
            #cv2.imwrite('./kk.png', masked)
        res = np.array(res).transpose((1, 2, 0))
        save_path = path[i][:-6] + '_y.npy'
        np.save(save_path, res)
        print(res.shape)
        print('end' + str(i))