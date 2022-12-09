# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch
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
    with open("/Users/mao/Desktop/upenn/diff_blas/cifar-10-batches-py/"+file, 'rb') as fo:
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    net = Net()
    print(net)
    test_dataset = my_Data_Set()
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    for epoch in range(1):
        correct = 0
        total=0

        for i, data in enumerate(test_loader, 0):
            total = total + len(data[0])
            inputs, labels = data
            inputs = inputs.to(torch.float32)

            output=net(inputs)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()
            print("acc is "+str(correct/total))

    print("end")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
