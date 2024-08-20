import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class WGANGP:
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        lr,
        batch_size,
        num_epochs,
        lambda_gp,
        save_interval,
        save_path="./",
        data_path="./data",
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lambda_gp = lambda_gp
        self.save_interval = save_interval
        self.save_path = save_path
        self.G = Generator(input_dim, output_dim, hidden_dim)
        self.D = Discriminator(output_dim, hidden_dim)
        # self.optimizer_G = optim.RMSprop(self.G.parameters(), lr=lr)
        # self.optimizer_D = optim.RMSprop(self.D.parameters(), lr=lr)
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=lr)
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=lr)
        self.dataloader = self.setup_data(data_path)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """
        Calculates the gradient penalty for enforcing the Lipschitz constraint on the discriminator.
        This is a key component in WGAN-GP for stable training.

        Parameters:
            D (torch.nn.Module): The discriminator (or critic) model.
            real_samples (torch.Tensor): A batch of real data samples.
            fake_samples (torch.Tensor): A batch of generated data samples.

        Returns:
            torch.Tensor: The gradient penalty.
        """

        # If you imagine real_samples and fake_samples as tensors of shape [batch_size, num_features],
        # alpha needs to be of shape [batch_size, 1] initially and then expanded to [batch_size, num_features]
        # to perform element-wise operations correctly across all features of each sample in the batch.

        # Generate random weights alpha with the same batch size as the real samples
        # Alpha is used for convex combination of real and fake samples.
        alpha = torch.rand((real_samples.size(0), 1), device=real_samples.device)

        # Expand alpha to the size of real samples to enable element-wise operations
        alpha = alpha.expand(real_samples.size())

        # Compute the interpolated samples as convex combinations of real and fake samples.
        # The requires_grad_() ensures that gradients can be computed w.r.t. interpolates.
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)

        # Pass the interpolated samples through the discriminator
        d_interpolates = self.D(interpolates)

        # Create a tensor filled with 1.0, with the same size as d_interpolates,
        # which will be used as weights for the gradient computation.
        fake = torch.ones(d_interpolates.size(), device=real_samples.device)

        # Compute the gradients of the discriminator's output w.r.t. the inputs (interpolated samples)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,  # Allows further operations on gradients
            retain_graph=True,  # Keeps the graph, allowing more operations on it
            only_inputs=True,  # Ensures that only gradients w.r.t. interpolates are computed
        )[0]

        # If batch_size is 32 and each gradient is a 3x28x28 tensor,
        # the operation will change each 3x28x28 tensor into a single vector of size 3*28*28 = 2352,
        # resulting in a new tensor shape of [32, 2352].
        # This effectively flattens the gradient tensors for each sample into a single vector,
        # which is useful for operations (like calculating norms) that need to consider the gradient as a whole.

        # Flatten the gradients to simplify further operations like computing the norm
        gradients = gradients.view(gradients.size(0), -1)

        # Calculate the gradient penalty using the 2-norm;
        # penalizes the deviation of gradient norms from 1
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        # Return the gradient penalty component of the loss
        return gradient_penalty

    def setup_data(self, data_path):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        dataset = datasets.MNIST(  # TODO: Add file directory variable
            root=data_path, train=True, transform=transform, download=True
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        for epoch in range(self.num_epochs):
            for i, (real_images, _) in enumerate(self.dataloader):
                # Discriminator training
                self.D.zero_grad()
                real_images = real_images.view(
                    -1, self.output_dim
                )  # Reshape from [batch_size, 1, 28, 28] to [batch_size, 784]
                real_outputs = self.D(real_images)
                real_loss = -torch.mean(real_outputs)
                z = torch.randn(real_images.size(0), self.input_dim)
                fake_images = self.G(z)
                fake_outputs = self.D(
                    fake_images.detach()
                )  # ensuring that updates to the discriminator do not affect the generator.
                fake_loss = torch.mean(fake_outputs)
                gradient_penalty = self.compute_gradient_penalty(
                    real_images.data,  # use real_images.detach() for the same reason above
                    fake_images.data,
                )
                d_loss = real_loss + fake_loss + self.lambda_gp * gradient_penalty
                d_loss.backward()
                self.optimizer_D.step()
                # Generator training
                if i % 5 == 0:
                    self.G.zero_grad()
                    outputs = self.D(fake_images)
                    g_loss = -torch.mean(outputs)
                    g_loss.backward()
                    self.optimizer_G.step()
                # Print and save model
                if (i + 1) % 400 == 0 or (i + 1) == len(self.dataloader):
                    print(
                        f"Epoch [{epoch+1}/{self.num_epochs}], "
                        + f"Step [{i+1}/{len(self.dataloader)}], "
                        + f"D Loss: {d_loss.item()}, G Loss: {g_loss.item()}"
                    )
                if self.save_interval and (epoch + 1) % self.save_interval == 0:
                    self.save_models(epoch + 1)
        # Final model save
        torch.save(self.G.state_dict(), f"{self.save_path}generator_final.pth")
        torch.save(self.D.state_dict(), f"{self.save_path}discriminator_final.pth")
        print("Training completed.")

    def save_models(self, epoch):
        torch.save(
            self.G.state_dict(),
            f"{self.save_path}generator_epoch_{epoch}.pth",
        )
        torch.save(
            self.D.state_dict(),
            f"{self.save_path}discriminator_epoch_{epoch}.pth",
        )

    def load_models(self, path):
        self.G.load_state_dict(torch.load(path))


if __name__ == "__main__":
    # Usage
    gan = WGANGP(
        input_dim=100,
        output_dim=784,
        hidden_dim=128,
        lr=0.00005,
        batch_size=64,
        num_epochs=50,
        lambda_gp=10,
        save_interval=10,
        save_path="./gan/models/",
    )
    gan.train()
