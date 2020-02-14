from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

"""
Code to create DataLoader objects in the Pytorch format for the available datasets. 

"""

def load_mnist(transform=ToTensor(), custom_path=None, train=True, batch_size=32, shuffle=True, num_workers=2):
    from torchvision.datasets import MNIST
    if custom_path is None: custom_path = './downloaded datasets/mnist/'
    mnist = MNIST(custom_path, train=train, transform=transform, download=True)
    return DataLoader(mnist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_fake_data(num_samples=10000, image_size=(3, 32, 32), num_classes=10, transform=None,
                   batch_size=1, shuffle=True, num_workers=2):
    from torchvision.datasets import FakeData
    if transform is None:  # Assuming Zero-norm
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.29, 0.29, 0.29))])
    fake_data = FakeData(size=num_samples, image_size=image_size, num_classes=num_classes, transform=transform, random_offset=0)
    return DataLoader(fake_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_cifar10(transform=None, custom_path=None, train=True, batch_size=1, shuffle=True, num_workers=2):
    from torchvision.datasets import CIFAR10
    if custom_path is None: custom_path = './downloaded datasets/cifar10/'
    if transform is None:  # Assuming CIFAR-10 normalization.
        transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    cifar10 = CIFAR10(custom_path, train=train, transform=transform, download=True)
    return DataLoader(cifar10, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_cifar100(transform=None, custom_path=None, train=True, batch_size=1, shuffle=True, num_workers=2):
    from torchvision.datasets import CIFAR100
    if custom_path is None: custom_path = './downloaded datasets/cifar100/'
    if transform is None:  # Assuming CIFAR-100 normalization.
        transform = Compose([ToTensor(), Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    cifar100 = CIFAR100(custom_path, train=train, transform=transform, download=True)
    return DataLoader(cifar100, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_tiny_imagenet(transform=ToTensor(), train=True, batch_size=1, shuffle=True, num_workers=2):
    from svc.datasets.utils import TinyImageNet
    tiny = TinyImageNet('./downloaded datasets/tiny-imagenet-200/', transform=transform)
    return DataLoader(tiny, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_svhn(transform=ToTensor(), custom_path=None, split='test', batch_size=1, shuffle=True, num_workers=2):
    from torchvision.datasets import SVHN
    if custom_path is None: custom_path = './downloaded datasets/svhn/'
    svhn = SVHN(root=custom_path, split=split, transform=transform, download=True)
    return DataLoader(svhn, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
