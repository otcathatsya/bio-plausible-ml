from time import time as tm

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torchvision.datasets import CIFAR10

from Torch2PC import TorchSeq2PC as t2PC

torch.manual_seed(0)

train_dataset = CIFAR10('./',
                        train=True,
                        transform=torchvision.transforms.ToTensor(),
                        download=True)

test_dataset = CIFAR10('./',
                       train=False,
                       transform=torchvision.transforms.ToTensor(),
                       download=True
                       )

batch_size = 300

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ', device)

# hyper-parameters
num_epochs = 3
lr = .002

pc_error = "Exact"
pc_eta = .1
pc_n_steps = 20

optimizer_choice = torch.optim.Adam
loss_fun = nn.CrossEntropyLoss()

steps_per_epoch = len(train_loader)
total_num_steps = num_epochs * steps_per_epoch
print("steps per epoch (mini batch size)=", steps_per_epoch)


class BasicConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def reg_inference(model, X, Y):
    outputs = model(X)
    loss_reg = loss_fun(outputs, Y)
    loss_reg.backward()
    with torch.no_grad():
        return loss_reg.item()


def pc_inference(model, X, Y):
    vhat, loss, dLdy, v, epsilon = t2PC.PCInfer(model, loss_fun, X, Y, "Strict", eta=.1, n=20, vinit=None)
    with torch.no_grad():
        return loss.item()


def train(model, custom_inference):
    optimizer = optimizer_choice(model.parameters(), lr=lr)
    hist_loss = []
    t1 = tm()

    for k in range(num_epochs):
        train_iter = iter(train_loader)

        for i in range(steps_per_epoch):
            X, Y = next(train_iter)
            X = X.to(device)
            Y = Y.to(device)

            X.requires_grad = True

            loss = custom_inference(model, X, Y)
            hist_loss += [loss]

            optimizer.step()

            model.zero_grad()
            optimizer.zero_grad()
        print("Average loss in epoch", k, "is", np.mean(hist_loss))

    tTrain = tm() - t1
    print('Training time = ', tTrain, 'sec')
    return hist_loss


def eval(model):
    accs = []
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(test_loader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            predicted = model(inputs)

            accuracy = (torch.softmax(predicted, dim=1).argmax(dim=1) == labels).sum().item() / float(labels.size(0))
            accs += [accuracy]
    print("Mean batch accuracy of", np.mean(accs) * 100, "%")


model_pc = BasicConvNet().to(device)
model_reg = BasicConvNet().to(device)

print("TRAIN PC")
pc_loss = train(model_pc, reg_inference)

print("TRAIN REGULAR")
reg_loss = train(model_reg, reg_inference)

print("EVAL REGULAR")
eval(model_reg)

print("EVAL PC")
eval(model_pc)

# Plot the loss curves
plt.figure()
plt.plot(pc_loss)
plt.plot(reg_loss)
plt.legend(["Loss Pred Coding", "Loss Backprop"])
plt.ylim(bottom=0)
plt.ylabel('Training loss')
plt.xlabel('Steps')
plt.savefig("pc-loss.pdf", bbox_inches='tight')
plt.show()
