import numpy as np
import struct
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 训练集文件
train_images_idx3_ubyte_file = './pymnist/MNIST/raw/train-images-idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = './pymnist/MNIST/raw/train-labels-idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = './pymnist/MNIST/raw/t10k-images-idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = './pymnist/MNIST/raw/t10k-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):

    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):

    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):

    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):

    return decode_idx1_ubyte(idx_ubyte_file)


class my_trainData_Set(nn.Module):
    def __init__(self, transform=None, target_transform=None, loader=None):
        super(my_trainData_Set, self).__init__()

        # data_batch = unpickle("test_batch")
        # self.labels=data_batch[b'labels']
        # self.datas=data_batch[b'data']
        test_images = load_train_images()
        test_labels = load_train_labels()
        self.labels = test_labels

        self.datas = test_images

    def __getitem__(self, item):


        return self.datas[item].reshape(1,28,28),self.labels[item]
        # return self.images[item].reshape(1,96,96), self.labels[item],self.info2[item]

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.labels)
class my_testData_Set(nn.Module):
    def __init__(self, transform=None, target_transform=None, loader=None):
        super(my_testData_Set, self).__init__()

        # data_batch = unpickle("test_batch")
        # self.labels=data_batch[b'labels']
        # self.datas=data_batch[b'data']
        test_images = load_test_images()
        test_labels = load_test_labels()
        self.labels = test_labels

        self.datas = test_images

    def __getitem__(self, item):


        return self.datas[item].reshape(1,28,28),self.labels[item]
        # return self.images[item].reshape(1,96,96), self.labels[item],self.info2[item]

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.labels)

def run():
    # train_images = load_train_images()
    # train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    # for i in range(10):
    #     print(train_labels[i])
    #     print(test_images[i].shape[0:2])
    #     plt.imshow(train_images[i], cmap='gray')
    #     plt.show()
    print('done')

# def test():
#     total = 0
#     correct = 0
#     for images, labels in test_loader:
#         # images = (images.view(-1, 28*28))
#         labels = (labels)
#         outputs = net(images)
#
#         _, predicts = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicts == labels).sum()
#     print("Accuracy = %.2f" % (100 * correct / total))


def train():
    batch_size=200
    train_dataset=my_trainData_Set()
    test_dataset=my_testData_Set()
    # train_dataset = train_dataset.type(torch.LongTensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    net = Net()
    print(net)




    learning_rate = 1e-1
    num_epoches = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    for epoch in range(num_epoches):
        print("current epoch = {}".format(epoch))
        for i, (images, labels) in enumerate(train_loader):
            # print(images.shape)
            # images = (images.view(-1, 28*28))

            images = images.to(torch.float32)
            # labels = labels.to(torch.float32)

            labels = (labels.long())
            # print(labels)

            outputs = net(images)
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)  # calculate loss
            optimizer.zero_grad()  # clear net state before backward
            loss.backward()
            optimizer.step()  # update parameters

            if i % 5000 == 0:
                print("current loss = %.5f" % loss.item())
    print("finished training")
    torch.save(net.state_dict(), "./savedModel.pth")

    #test
    total = 0
    correct = 0
    for images, labels in test_loader:
        # images = (images.view(-1, 28*28))
        images = images.to(torch.float32)
        # labels = labels.to(torch.float32)

        labels = (labels.long())
        outputs = net(images)



        _, predicts = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum()

        print("Accuracy = %.2f" % (100 * correct / total))


if __name__ == '__main__':
    train()
