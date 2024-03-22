import matplotlib.pyplot as plt
import torchvision.transforms as T

from torchvision.datasets.cifar import CIFAR100
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop

#데이터 증강 (크로핑>패딩>좌우대칭)
transforms = Compose([
    T.ToPILImage(),
    RandomCrop((32, 32), padding=4),
    RandomHorizontalFlip(p=0.5),
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
