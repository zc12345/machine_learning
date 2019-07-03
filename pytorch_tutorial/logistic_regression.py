# coding: utf-8
"""
@author: zc12345 
@contact: 18292885866@163.com

@file: logistic_regression.py
@time: 2019/7/1 17:41
@description:

"""
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

def load_mnist_data(dataset_root='../dataset/mnist/', batch_size=8):
    # MNIST dataset

    train_dataset = MNIST(root=dataset_root, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = MNIST(root=dataset_root, train=False, transform=transforms.ToTensor(), download=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def train():
    num_classes = 10
    num_epochs = 100
    batch_size = 100
    lr = 0.001
    input_size = 28*28

    train_loader, _ = load_mnist_data('../../dataset/mnist', batch_size=batch_size)

    model = nn.Linear(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        if epoch > 10:
            lr /= 10
        for index, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28*28)
            targets = model(images)
            loss = criterion(targets, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (index + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, index+1, len(train_loader), loss.item()))
    torch.save(model.state_dict(), 'model.ckpt')

    plt.plot(range(num_epochs*len(train_loader)), losses, 'r-', label='loss')
    plt.show()


def validation():
    num_classes = 10
    batch_size = 8
    input_size = 28 * 28
    _, test_loader = load_mnist_data('../../dataset/mnist', batch_size=batch_size)
    model = nn.Linear(input_size, num_classes)
    model.load_state_dict(torch.load('model.ckpt'))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size)
            targets = model(images)
            _, pred = torch.max(targets.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum()
        print('Accuracy of the model on the 10000 test images: {:.4f} %'.format(100.0 * correct / total))


if __name__ == '__main__':
    train()
    validation()