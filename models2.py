import torch
import torch.nn as nn



class LeNet(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.fc1 = nn.Linear(20000, 500)
        self.fc2 = nn.Linear(500,10)
        self.dropout = dropout
        