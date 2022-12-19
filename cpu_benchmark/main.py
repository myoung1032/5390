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

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96,3), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




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
                print(len(inputs))
                inputs = inputs.to(torch.float32)

                output=net(inputs)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == labels).sum().item()
                #print("acc is "+str(correct/total))
            print("sample size = ", i)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    net = AlexNet()
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
