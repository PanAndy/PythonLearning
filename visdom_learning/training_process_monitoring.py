# code from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-a-classifier
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from train_a_model.tools import Net


from matplotlib import pyplot as plt
import numpy as np
import visdom

viz = visdom.Visdom()

# loading and normalizing CIFAR10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()


# define a convolutional neural network
net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

# define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-7, momentum=0.9)

# Train the network
idx = 0
for epoch in range(10):  # loop over the dataset multiple times
    if epoch < 2:
        optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    elif 2 <= epoch < 4:
        optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    elif 4 <= epoch < 6:
        optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
    elif 6 <= epoch < 8:
        optimizer = optim.SGD(net.parameters(), lr=1e-6, momentum=0.9)
    elif 8 <= epoch < 10:
        optimizer = optim.SGD(net.parameters(), lr=1e-7, momentum=0.9)

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
                  (epoch + 1, i + 1, running_loss / 1000))
            l = loss.item()
            idx = idx + 1
            viz.line([running_loss / 1000], [idx], win="train_loss", opts=dict(title='train_loss'), update="append")



print('Finished Training')

# save model
PATH = './cifar_net_monitoring.pth'
torch.save(net.state_dict(), PATH)

