from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class MNISTLoader:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.trainloader = None
        self.testloader = None

    def load_train_data(self):
        trainset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=self._transform,
        )
        self.trainloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return self.trainloader

    def load_test_data(self):
        testset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=self._transform,
        )
        self.testloader = DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return self.testloader

    def print_batch(self, loader="train", num_images=5):
        if loader == "train":
            dataiter = iter(self.trainloader)
        elif loader == "test":
            dataiter = iter(self.testloader)
        else:
            raise ValueError("Loader must be either 'train' or 'test'.")

        images, labels = next(dataiter)
        images = images[:num_images]
        labels = labels[:num_images]

        # Convert torch tensors to numpy arrays for visualization
        images = images.numpy()
        labels = labels.numpy()

        fig = plt.figure(figsize=(8, 8))
        for idx in range(num_images):
            ax = fig.add_subplot(1, num_images, idx + 1, xticks=[], yticks=[])
            ax.imshow(images[idx].squeeze(), cmap="gray")
            ax.set_title(str(labels[idx]))
        plt.show()


if __name__ == "__main__":

    # Example usage of the MNISTLoader class
    mnist_loader = MNISTLoader(batch_size=128)

    # Load training data
    train_loader = mnist_loader.load_train_data()

    # Load test data
    test_loader = mnist_loader.load_test_data()

    # Display the dataset
    mnist_loader.print_batch(loader="train")
