import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import EMNISTModel


def get_loaders():
    affine_transform = torchvision.transforms.RandomAffine(degrees=75, translate=(0.3, 0.3), scale=(0.5, 1.5), shear=30)
    composed_transformer = torchvision.transforms.Compose([
        torchvision.transforms.RandomApply([affine_transform], p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.1736, std=0.3317, inplace=True),
    ])

    emnist_train = torchvision.datasets.EMNIST(root="../data", split="byclass", train=True, download=True, transform=composed_transformer)
    emnist_test = torchvision.datasets.EMNIST(root="../data", split="byclass", train=False, download=True, transform=composed_transformer)

    train_loader = DataLoader(dataset=emnist_train, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(dataset=emnist_test, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    return train_loader, test_loader


def train(network, criterion, optimizer, num_of_epochs, train_loader):
    train_loader = iter(train_loader)
    lowest_loss = 99999
    for epoch in range(num_of_epochs):
        running_loss = 0.0
        for i, sample in enumerate(train_loader, 0):
            data, target = sample[0], sample[1]
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                if running_loss < lowest_loss:
                    lowest_loss = running_loss
                    torch.save(network.state_dict(), "../saved-models/best.pth")
                    print("Best loss so far, saving the model..")
                running_loss = 0.0

    print("Training finished.")


if __name__ == "__main__":
    load_trained = True

    # Get data loaders
    train_loader, test_loader = get_loaders()
    # Create model
    network = EMNISTModel()
    # Set up optimizer and criterion
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)

    if load_trained:
        network.load_state_dict(torch.load("../saved-models/best.pth"))

    train(network, criterion, optimizer, 10, train_loader)

    # TODO: Add test and validation
    # TODO: Use GPU


