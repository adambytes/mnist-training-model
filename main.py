import os
from pickletools import optimize

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# hyperparameters

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 32, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize tesnorboard
writer = SummaryWriter("runs/mnist")

# initialize the model
model = ConvNet().to(device)
# if model.ckpt exists:
if os.path.isfile("model.ckpt"):
    model.load_state_dict(torch.load("model.ckpt"))
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# initialize optimzer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if os.path.isfile("optimizer.ckpt"):
    optimizer.load_state_dict(torch.load("optimizer.ckpt"))


def train():
    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        len(train_loader),
                        loss.item(),
                    )
                )
                # Save the model checkpoint
                torch.save(model.state_dict(), "model.ckpt")
                torch.save(optimizer.state_dict(), "optimizer.ckpt")
        writer.add_scalar("training loss", loss, epoch)
        writer.add_scalar("accuracy", 100 * correct / total, epoch)
        test()


def test():
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            # make grid of images
            grid = torchvision.utils.make_grid(images)
            # write to tensorboard
            writer.add_image("mnist_images", grid)
            writer.add_graph(model, images)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Test accuracy of the model on the 10000 test images: {} %".format(
                100 * correct / total
            )
        )


# Tensorboard


# Run the training loop
test()
train()
