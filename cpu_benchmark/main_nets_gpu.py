import torch.nn as nn
import torch
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo


PROJECT_FOLDER_PATH = "/Users/dennice/Desktop/ese539_project/"

torch.manual_seed(0)
np.random.seed(0)

""" Define vgg net """

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 提取特征网络结构
def make_features(cfg: list):
    # 定义一个空列表，用来存放定义的每一层结构
    layers = []
    in_channels = 3  # RGB彩色通道
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)  # 将列表通过非关键字参数的形式进行传入

# 定义字典文件  key(vgg11为配置文件)   value为一个列表，其中数字代表卷积层卷积核的个数，M代表池化层结构
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 进行实例化
# **kwargs是将一个可变的关键字参数的字典传给函数实参，同样参数列表长度可以为0或为其他值。
def vgg(model_name="vgg16", **kwargs):  # **kwargs关键字参数
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model


""" define resnet """

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

def ResNet18():
    return ResNet(ResBlock)


""" define alexnet """

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

""" Define self-defined net """
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



""" dataset processing """

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

""" Benchmark functions """

# plot the graph with total of n samples, x= acc_time for each plot_unit
def benchmark_inference(model, dataloader, n=None, trials=3, warmup=5000, plot_unit=1000, device="cpu"):
    model.eval()
    """ n = Number of inferences, trials = number of trials, warmup = number of warmup trials """


    #warm up our cpu
    validate_model(model, dataloader, n_samples = warmup, device="cuda")
    print("warm up with {warmup} sample finish".format(warmup=warmup))

    accumulated_runtimes_list = [0]

    accumulated_runtime = 0
    n_remaining = n
    while(n_remaining > 0):
        n_remaining -= plot_unit
        elapses = []
        for i in range(trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            # whatever you are timing goes here
            validate_model(model, dataloader, n_samples = plot_unit, device="cuda")
            end.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            elapses.append(start.elapsed_time(end)/1000)
            
        accumulated_runtime += np.mean(elapses)
        accumulated_runtimes_list.append(accumulated_runtime)
    return accumulated_runtimes_list



def validate_model(model, dataloader, n_samples=None, device="cpu"):
    net = model
    correct = 0
    total=0
    BATCH_SIZE = 10

    # Move model to device (CPU or GPU)
    # One time to amortize data movement
    dev = torch.device(device)
    net.to(dev)

    for i, data in enumerate(dataloader, 0):
        n_samples -= BATCH_SIZE
        print("num_ samples remaining: ", n_samples)
        if(n_samples <= 0):
            break
        total = total + len(data[0])
        inputs, labels = data
        inputs = inputs.to(torch.float32)
        # Send inputs to device
        inputs = inputs.to(dev)
        labels = labels.to(dev)


        output=net(inputs)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()

""" main() """
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # instanciate our nets
    vgg16 = vgg(model_name="vgg16", num_classes=10, init_weights=True)  # num_classes=5, init_weights=True保存在**kwargs中
    print("vgg16: \n")
    print(vgg16)

    resnet18 =  ResNet18()
    print("resnet18: \n")
    print(resnet18)

    alexnet = AlexNet()
    print("alexnet: \n")
    print(alexnet)

    # load the dateset
    test_dataset = my_Data_Set()
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # define our benchmarking parameters
    n = 100          # total number of samples to inference
    trials = 3;       # the trial number for calculating average runtime
    warmup = 100       # number of samples to inference for warming up the CPU/GPU
    plot_unit = 10    # plot_unit is the step by which our plots are used
    BLAS = "IntelMKL"
    # run benchmark for vgg16
    accumulated_runtimes_vgg16 = benchmark_inference(vgg16, test_loader, n=n, trials=trials, warmup=warmup, plot_unit=plot_unit)
    
    accumulated_runtimes_resnet18 = benchmark_inference(resnet18, test_loader, n=n, trials=trials, warmup=warmup, plot_unit=plot_unit)
    accumulated_runtimes_alexnet = benchmark_inference(alexnet, test_loader, n=n, trials=trials, warmup=warmup, plot_unit=plot_unit)

    # show the acc runtimes results
    print("vgg16 acc runtimes = ", accumulated_runtimes_vgg16)
    print("resnet18 acc runtimes = ", accumulated_runtimes_resnet18)
    print("alexnet acc runtimes = ", accumulated_runtimes_alexnet)


    x = np.arange(0, n+plot_unit, plot_unit)/1000
    plt.plot(x, accumulated_runtimes_vgg16, marker='o',label='vgg16')
    plt.plot(x, accumulated_runtimes_resnet18, marker='^',label='resnet18')
    plt.plot(x, accumulated_runtimes_alexnet, marker='D',label='alexnet')
  
    # Add labels and title
    plt.title("Comparison of inference runtime between vgg16, resnet18, alexnet")
    plt.xlabel("number of inference samples (k)")
    plt.ylabel("accumulated runtime (second)")
    
    plt.legend()

    plt.savefig(PROJECT_FOLDER_PATH + "graphs/"+BLAS+".png", bbox_inches ="tight", pad_inches = 1)
    # plt.show()
    # print("Task: inference {n} epochs\n average run time for {n_trial} times = {time}s".format(n = 50, n_trial=trials, time=runtime))
    print("end")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
