import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import *


class LabelledMemory(Dataset):
    def __init__(self, capacity, image_shape, label_shape, device):
        self.capacity = int(capacity)
        self.image_shape = image_shape
        self.label_shape = label_shape
        self.device = device
        self.is_image = len(image_shape) == 3
        self.image_type = np.uint8 if self.is_image else np.float32

        self.transform = Compose([
            # Resize((96, 96)),
            ToTensor(),
            # Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])
        ])
        self.target_transform = Compose([
            # Resize((96, 96)),
            ToTensor(),
            # Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])
        ])

        self.reset()

    def reset(self):
        self._n = 0
        self._p = 0

        self.images = np.empty(
            (self.capacity, *self.image_shape), dtype=self.image_type)
        self.labels = np.empty(
            (self.capacity, *self.label_shape), dtype=np.float32)

    def append(self, images, label):
        self._append(images, label)

    def _append(self, images, label):
        images = np.array(images, dtype=self.image_type)
        label = np.array(label, dtype=np.float32)

        self.images[self._p] = images
        self.labels[self._p] = label

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def get(self):
        valid = slice(0, self._n)
        return (
            self.images[valid], self.labels[valid])

    def load(self, batch):
        num_data = len(batch[0])

        if self._p + num_data <= self.capacity:
            self._insert(
                slice(self._p, self._p+num_data), batch,
                slice(0, num_data))
        else:
            mid_index = self.capacity-self._p
            end_index = num_data - mid_index
            self._insert(
                slice(self._p, self.capacity), batch,
                slice(0, mid_index))
            self._insert(
                slice(0, end_index), batch,
                slice(mid_index, num_data))

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity

    def _insert(self, mem_indices, batch, batch_indices):
        images, labels = batch
        self.images[mem_indices] = images[batch_indices]
        self.labels[mem_indices] = labels[batch_indices]

    def __getitem__(self, idx):
        if self.is_image:
            image = self.images[idx].astype(np.uint8)
        else:
            image = self.images[idx]

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return self._n
