"""Random initialization."""

import random

import numpy as np
import torch


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if device.startswith("cuda:"):
            torch.cuda.set_device(int(device.split(":")[1]))
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
