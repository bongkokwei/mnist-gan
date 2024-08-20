from mnist_loader import MNISTLoader
from mnist_model import Model

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    batch_size=128,
):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()  # Backpropagation
        optimizer.step()

        if batch % 100 == 0:
            # print("X: ", torch.mean(X))
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def train_and_validate(
    train_dataloader,
    test_dataloader,
    epochs=5,
    batch_size=128,
):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":

    hyper_param = {
        "learning_rate": 0.5,
        "batch_size": 32,
        "epochs": 5,
    }

    # Example usage of the MNISTLoader class
    mnist_loader = MNISTLoader(batch_size=hyper_param["batch_size"])
    train_loader = mnist_loader.load_train_data()
    test_loader = mnist_loader.load_test_data()

    model = Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=hyper_param["learning_rate"],
    )

    train_and_validate(
        train_loader,
        test_loader,
        epochs=hyper_param["epochs"],
        batch_size=hyper_param["batch_size"],
    )
