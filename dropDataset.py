import cv2
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as transforms


class DropDataset(Data.Dataset):
    """DropDataset
    Args:
        data: list of tuple (path, label)
        mode: "train" with data enhancement or "test" without
    """

    def __init__(self, data, mode="test"):
        self.data = data
        if mode == "train":
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([32, 32]),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    def __getitem__(self, index):
        img = cv2.imread(self.data[index][0]) / 255
        img = self.transforms(img)
        label = self.data[index][1]
        return img.to(torch.float32), torch.tensor(label).to(torch.float32)

    def __len__(self):
        return len(self.data)


class WSCDataset(Data.Dataset):
    """WSCDataset
    Args:
        data: list of tuple (path, label)
    """

    def __init__(self, data, mode="test"):
        self.data = data
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((48, 48)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __getitem__(self, index):
        img = cv2.imread(self.data[index][0]) / 255
        img = self.transforms(img)
        label = self.data[index][1]
        return img.to(torch.float32), torch.tensor(label).to(torch.float32)

    def __len__(self):
        return len(self.data)

