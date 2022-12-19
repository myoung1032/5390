# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import time
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_FOLDER_PATH = "/Users/dennice/Desktop/ese539_project/"

import torch


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torch
import torch.nn as nn

#定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        #这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
    #这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        #在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out





def unpickle(file):
    import pickle
    with open(PROJECT_FOLDER_PATH + file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class my_Data_Set(nn.Module):
    def __init__(self, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()

        data_batch = unpickle("test_batch")
        self.labels=data_batch[b'labels']
        self.datas=data_batch[b'data']

        self.labels = np.array(self.labels)

        self.datas = np.array(self.datas)

    def __getitem__(self, item):

        # img=cv2.imread

        img = self.datas[item]

        image = self.datas[item]
        # 分离出r,g,b：3*1024
        image = image.reshape(-1, 1024)
        r = image[0, :].reshape(32, 32)  # 红色分量
        g = image[1, :].reshape(32, 32)  # 绿色分量
        b = image[2, :].reshape(32, 32)  # 蓝色分量
        img = np.zeros((32, 32, 3))
        img[:, :, 0] = r / 255
        img[:, :, 1] = g / 255
        img[:, :, 2] = b / 255
        return img.reshape((3,32,32)),self.labels[item]
        # return self.images[item].reshape(1,96,96), self.labels[item],self.info2[item]

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.labels)

def benchmark_inference(model, dataloader, n=None, trials=5, warmup=1):
    """ n = Number of inferences, trials = number of trials, warmup = number of warmup trials """
    # TODO Implement
    #warm up our cpu
    validate_model(model, dataloader, n_epoch = warmup)
    elapses = []
    for i in range(trials):
      start = time.time()

      validate_model(model, dataloader, n_epoch = n)

      end = time.time()
      elapses.append(end - start)
    return np.mean(elapses) # TODO Return inference time in seconds


def validate_model(model, dataloader, n_epoch=None):
        net = model
        for epoch in range(n_epoch):
            correct = 0
            total=0


            for i, data in enumerate(dataloader, 0):
                total = total + len(data[0])
                inputs, labels = data
                # print(len(inputs))
                inputs = inputs.to(torch.float32)

                output=net(inputs)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == labels).sum().item()
                #print("acc is "+str(correct/total))
            print("sample size = ", i)

# Press the green button in the gutter to run the script.

def ResNet18():
    return ResNet(ResBlock)
if __name__ == '__main__':
    net = ResNet18()

    print(net)
    test_dataset = my_Data_Set()
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    n = 50
    trials = 5;
    warmup = 5;
    runtime = benchmark_inference(net, test_loader, n=n, trials=trials, warmup=warmup)

    print("Task: inference {n} epochs\n average run time for {n_trial} times = {time}s".format(n = 50, n_trials=trials, time=runtime))
    print("end")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
