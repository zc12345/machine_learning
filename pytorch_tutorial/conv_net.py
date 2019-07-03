# coding: utf-8
"""
@author: zc12345 
@contact: 18292885866@163.com

@file: conv_net.py
@time: 2019/7/1 16:00
@description:

"""


import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

# Hyper Params
num_epochs = 5
num_classes = 10
batch_size = 8
lr = 0.001


def load_mnist_data(dataset_root='../../dataset/mnist/'):
    # MNIST dataset

    train_dataset = torchvision.datasets.MNIST(root=dataset_root, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = torchvision.datasets.MNIST(root=dataset_root, train=False, transform=transforms.ToTensor(), download=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def show_data():
    train_loader, test_loader = load_mnist_data()

    for i, (images, labels) in enumerate(train_loader):
        if i == 0:
            images = images.to(device)
            images = images.numpy()
            images = images.squeeze()  # (batch_size, 1, 28, 28) -> (batch_size, 28, 28)
            plt.imshow(images[0])
            plt.show()


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train():
    model = ConvNet(num_classes).to(device)
    train_loader, test_loader = load_mnist_data()

    # loss
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, steps, loss.item()))

    torch.save(model.state_dict(), 'model.ckpt')


def test():
    model = ConvNet(num_classes).to(device)
    model.load_state_dict(torch.load('model.ckpt'))
    model.to(device)
    train_loader, test_loader = load_mnist_data()

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicts = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicts == labels).sum().item()

        print('Test Accuracy of the model on the {} test images: {} %'.format(len(test_loader)*batch_size, 100 * correct / total))



if __name__ == '__main__':
    # train()
    # test()
    show_data()