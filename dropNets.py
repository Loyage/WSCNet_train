import torch
import torch.nn as nn
import torchvision.models as models


class LeNet5(nn.Module):
    """Lenet5"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Resnet18(nn.Module):
    """Resnet18"""

    def __init__(self, num_classes=4):
        super().__init__()
        self.net = models.resnet18()
        self.net.fc = nn.Linear(512, 4)

    def forward(self, x):
        x = self.net(x)
        return x


class Resnet50(nn.Module):
    """Resnet50"""

    def __init__(self):
        super().__init__()
        self.net = models.resnet50()
        self.net.fc = nn.Linear(2048, 4)

    def forward(self, x):
        x = self.net(x)
        return x


class MobileNet(nn.Module):
    """MobileNetV3(small)"""

    def __init__(self, num_classes=4):
        super().__init__()
        self.net = models.mobilenet_v3_small()
        self.net.classifier = nn.Linear(576, 4)

    def forward(self, x):
        x = self.net(x)
        return x


class DropCondNet(nn.Module):
    """DropCondNet
    Combination of Resnet18, Resnet50, MobileNetV3(small), LeNet5
    input size: batch_size * 3 * 32 * 32
    output size: batch_size * 4
    """

    def __init__(self, net_name="Resnet18"):
        super().__init__()
        if net_name == "Resnet18":
            self.net = Resnet18()
        elif net_name == "Resnet50":
            self.net = Resnet50()
        elif net_name == "MobileNet":
            self.net = MobileNet()
        elif net_name == "LeNet5":
            self.net = LeNet5()
        else:
            raise Exception("net_name error")

    def forward(self, x):
        x = self.net(x)
        return x


class WSCNet(nn.Module): #drop_net_0.95
    '''WSCNet
    Weakly supervised cell counting network, can not only classify the droplet type, but also locate the cells. 
    You don't need to label the location of cells, just the number of cells (empty, single, multiple).
    Even if the droplet has > 2 cells, all you need to do is label it as multiple, this net will automaticly identify all the cells by a well-designed loss function.
    '''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,48,3,padding = 1)
        self.conv2 = nn.Conv2d(48,128,3,padding = 1)
        self.conv3 = nn.Conv2d(128,256,3,padding = 1)
        self.conv4 = nn.Conv2d(256,512,3,padding = 1)
        self.deconv1 = nn.ConvTranspose2d(512,128,kernel_size = 3,stride = 2, padding = 1,output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128,48,kernel_size = 3,stride = 2, padding = 1,output_padding=1)
        self.conv5 = nn.Conv2d(48,1,1)

        self.conv31 = nn.Conv2d(256,256,3,padding = 1)
        self.conv32 = nn.Conv2d(256,128,3,padding = 1)

        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(48)

        self.bn31 = nn.BatchNorm2d(256)
        self.bn32 = nn.BatchNorm2d(128)

        self.classifier = nn.Sequential(
                nn.Linear(128*6*6,256),
                nn.ReLU(inplace = True),
                nn.Dropout(),
                nn.Linear(256,256),
                nn.ReLU(inplace = True),
                nn.Dropout(),
                nn.Linear(256,2)
                )
        
    def forward(self,x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.max_pool2d(F.relu(self.bn2(self.conv2(x1))),(2,2))
        x3 = F.max_pool2d(F.relu(self.bn3(self.conv3(x2))),(2,2))

        x_class = F.relu(self.bn31(self.conv31(x3)))
        x_class = F.max_pool2d(F.relu(self.bn32(self.conv32(x_class))),(2,2))
        x_class = x_class.view(x_class.size()[0],-1)
        x_class = self.classifier(x_class)

        x_count = F.relu(self.bn4(self.conv4(x3)))
        x_count = F.relu(self.bn5(self.deconv1(x_count)))
        x_count = F.relu(self.bn6(self.deconv2(x_count)))
        x_count = F.relu(self.conv5(x_count))
        
        return x_class, x_count


class WSCLoss(nn.Module):
    """WSCLoss"""

    def __init__(self):
        super().__init__()
        self.class_loss_func = nn.CrossEntropyLoss()

    def forward(self, output_class, output_count, labels):
        # classification loss
        class_label = t.zeros(labels.shape)
        class_label[labels > 0] = 1.0
        class_loss = self.class_loss_func(output_class, class_label.long())

        # counting loss
        max_pool = t.nn.MaxPool2d(kernel_size=(output_count.shape[2], output_count.shape[3]))
        max_den = max_pool(output_count)
        max_loss = F.relu(max_den-1).mean()

        output_count = output_count.view(output_count.size()[0], -1)
        count_map = 2 - F.leaky_relu(2-output_count.sum(1), negative_slope=0.01)

        count_mask = t.zeros(labels.shape)
        count_mask_0 = t.zeros(labels.shape)
        count_mask_1 = t.zeros(labels.shape)
        count_mask_2 = t.zeros(labels.shape)
        count_mask[labels > 0] = 1.0
        count_mask_0[labels == 1] = 1.0
        count_mask_1[labels == 2] = 1.0
        count_mask_2[labels == 3] = 1.0

        _0_loss = count_mask_0.mul(count_map**2).sum()
        _1_loss = count_mask_1.mul((count_map-1)**2).sum()
        _2_loss = count_mask_2.mul((2-count_map)**2).sum()

        count_loss = (_0_loss + _1_loss + _2_loss)/max(0.001,count_mask.sum()) + max_loss
        
        # sum loss
        sum_loss = class_loss + count_loss
        return sum_loss