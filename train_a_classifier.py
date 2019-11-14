# code from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-a-classifier
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tools import imshow
from tools import Net

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


from IPython import display
from matplotlib import pyplot as plt
import numpy as np


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    # display.set_matplotlib_formats('svg')
    # plt.ion()
    plt.figure(1)

# loading and normalizing CIFAR10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# define a convolutional neural network


net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)


# define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-7, momentum=0.9)


# Train the network
ls = []
for epoch in range(100):  # loop over the dataset multiple times
    if epoch < 10:
        optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    elif 10 <= epoch < 20:
        optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    elif 20 <= epoch < 30:
        optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
    elif 30 <= epoch < 40:
        optimizer = optim.SGD(net.parameters(), lr=1e-6, momentum=0.9)
    elif 40 <= epoch < 50:
        optimizer = optim.SGD(net.parameters(), lr=1e-7, momentum=0.9)
    elif 50 <= epoch < 60:
        optimizer = optim.SGD(net.parameters(), lr=1e-8, momentum=0.9)
    elif 60 <= epoch < 70:
        optimizer = optim.SGD(net.parameters(), lr=1e-9, momentum=0.9)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            ls.append(running_loss/2000)
            set_figsize()
            plt.plot(np.arange(0, len(ls)), ls)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.draw()
            plt.pause(0.001)
            running_loss = 0.0



print('Finished Training')

# save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

