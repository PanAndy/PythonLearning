import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tools import Net
from tools import TheModelClass

targetModel = TheModelClass()

cifar_net = torch.load('./cifar_net.pth')

for item in cifar_net:
    print('cifar_net \t', item, '\t')

targetModel.load_state_dict(cifar_net, strict=False)

for item in targetModel.state_dict():
    print('targetModel \t', item, '\t')

print('cifar_net \t', cifar_net["fc3.bias"], '\t', cifar_net["fc3.bias"].data)
print('targetModel \t', targetModel.state_dict()["fc_the_model_class.fc.4.bias"], '\t', targetModel.state_dict()["fc_the_model_class.fc.4.bias"].data)

cifar_net["fc_the_model_class.fc.0.weight"] = cifar_net.pop("fc1.weight")
cifar_net["fc_the_model_class.fc.0.bias"] = cifar_net.pop("fc1.bias")
cifar_net["fc_the_model_class.fc.2.weight"] = cifar_net.pop("fc2.weight")
cifar_net["fc_the_model_class.fc.2.bias"] = cifar_net.pop("fc2.bias")
cifar_net["fc_the_model_class.fc.4.weight"] = cifar_net.pop("fc3.weight")
cifar_net["fc_the_model_class.fc.4.bias"] = cifar_net.pop("fc3.bias")

targetModel.load_state_dict(cifar_net, strict=False)
print('cifar_net \t', cifar_net["fc_the_model_class.fc.4.bias"], '\t', cifar_net["fc_the_model_class.fc.4.bias"].data)
print('targetModel \t', targetModel.state_dict()["fc_the_model_class.fc.4.bias"], '\t', targetModel.state_dict()["fc_the_model_class.fc.4.bias"].data)


