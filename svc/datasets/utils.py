import torch.utils.data as data
from PIL import Image
import os
import sys

class TinyImageNet(data.Dataset):
    def __init__(self, root=None, transform=None):
        self.root = os.path.abspath(root)
        self.transform = transform
        self.paths, self.labels = self.load_paths()
        self.length = len(self.paths)
        self.check_if_downloaded()

    def check_if_downloaded(self):
        if not os.path.isdir(self.root):
            print("Downloading to " + self.root)
            download_tiny_imagenet_validation()
        else:
            print("Dataset already downloaded at " + self.root)

    def load_paths(self):
        paths = []
        image_folder_path = os.path.join(self.root, 'val', 'images')

        for idx in range(10000):
            filename = 'val_' + str(idx) + '.JPEG'
            paths.append(os.path.join(image_folder_path, filename))

        labels = [-1] * len(paths)
        return paths, labels

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label = self.labels[index]
        img = Image.open(self.paths[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def download_tiny_imagenet_validation():
    import zipfile
    import urllib.request
    print("Downloading Tiny ImageNet")
    download = urllib.request.urlretrieve('http://cs231n.stanford.edu/tiny-imagenet-200.zip', filename='./downloaded datasets/tiny-imagenet-200.zip')
    # archive = zipfile.ZipFile(download[0])  # Take the path of the downloaded file

    print("Extracting validation data ...")

    # https://stackoverflow.com/questions/6735917/redirecting-stdout-to-nothing-in-python
    new_target = open(os.devnull, "w")
    old_target = sys.stdout
    sys.stdout = new_target
    with zipfile.ZipFile(download[0]) as archive:
        for file in archive.namelist():
            if 'val/' in file:
                archive.extract(file, './downloaded datasets/')
    sys.stdout = old_target
    os.rename("./downloaded datasets/tiny-imagenet-200.zip", "./downloaded datasets/tiny-imagenet-200/tiny-imagenet-200.zip")
    print("Finished extracting")
