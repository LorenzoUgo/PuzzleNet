import numpy as np
import torch
import random


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        data_path = "data/np_ob_all_10000_test_2.npy"
        self.img_labels = np.load(data_path, allow_pickle=True)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        print("HAHAHAAH")
        return self.img_labels[idx]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)


data_path = "data/np_ob_all_10000_test_2.npy"

data = np.load(data_path, allow_pickle=True)

print(data, data.shape, data.dtype, data.size, data[0].shape)

datasetPYTORCH = CustomImageDataset()

dataloader = torch.utils.data.DataLoader(
    datasetPYTORCH,
    batch_size=32,
    drop_last=True,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

print(dataloader)
print(len(dataloader))
