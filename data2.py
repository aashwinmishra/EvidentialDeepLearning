import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_ds = MNIST("./data/mnist", download=True, train=True, transform=transforms.ToTensor())
val_ds = MNIST("./data/mnist", download=True, train=False, transform=transforms.ToTensor())

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=256, num_workers=4)

dataloaders = {"train": train_dl, "val": val_dl}
