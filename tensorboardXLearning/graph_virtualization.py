import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from train_a_model.tools import Net
from tensorboardX import SummaryWriter

dummy_input = torch.rand(4, 3, 32, 32)
model = Net()

with SummaryWriter(comment="net", log_dir="net") as w:
    w.add_graph(model, (dummy_input))



