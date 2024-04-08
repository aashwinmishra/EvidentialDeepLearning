import torch
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def relu_evidence(yb):
    return F.relu(yb)


