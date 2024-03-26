import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
import torch.nn as nn

from torchvision.datasets.cifar import CIFAR100
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
'''
데이터 증강 (크로핑>패딩>좌우대칭) 
: 데이터가 부족하거나 오버피팅을 피하는 기법으로 데이터를 의도적으로 수정해 더 많은 데이터를 확보한다. 
그림을 뒤집고, 색을 바꾸고 하는 등의 수정으로 이미지 하나를 여럿으로 늘리고 이미지의 일부를 제거하기도 한다. 
크로핑 시에 이미지 크기에 변화가 없도록 제로패딩을 사용
'''
transforms = Compose([
    T.ToPILImage(),
    RandomCrop((32, 32), padding=4),
    RandomHorizontalFlip(p=0.5),        #y축 기준 대칭
    T.ToTensor(),

    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),        #이미지 정규화
    T.ToPILImage()
])

training_data = CIFAR100(
    root="./",
    train=True,
    transform=transforms
)

test_data = CIFAR100(
    root="./",
    train=False,
    transform=transforms
)


for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(transforms(training_data.data[i]))
plt.show()


'''
VGG 기본 블록 정의
'''


class BasicBlock(nn.Module):       #기본 블록 정의
    #기본 블록을 구성하는 층 정의
    def __init__(self, in_channels, out_channels, hidden_dim):
        #nn.Module 클래스의 요소 상속
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        return x

class CNN(nn.Module):
    def __init__(self, num_classes):        #num_classes는 클래스 개수
        super(CNN, self).__init__()
        
        self.block1 = BasicBlock(in_channels=3, out_channels=32, hidden_dim=16)
        self.block2 = BasicBlock(in_channels=32, out_channels=128, hidden_dim=64)
        self.block3 = BasicBlock(in_channels=128, out_channels=256, hidden_dim=128)

        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
       

