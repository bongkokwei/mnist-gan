import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
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


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1), device=real_samples.device)
    alpha = alpha.expand(real_samples.size())
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_GAN(
    num_epochs,
    dataloader,
    G,
    D,
    optimizer_G,
    optimizer_D,
    lambda_gp,
    save_interval=None,
    save_path="./",
):
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            # Discriminator training
            D.zero_grad()
            real_images = real_images.view(-1, output_dim)
            real_outputs = D(real_images)
            real_loss = -torch.mean(real_outputs)
            z = torch.randn(real_images.size(0), input_dim)
            fake_images = G(z)
            fake_outputs = D(fake_images.detach())
            fake_loss = torch.mean(fake_outputs)
            gradient_penalty = compute_gradient_penalty(
                D, real_images.data, fake_images.data
            )
            d_loss = real_loss + fake_loss + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            # Generator training
            if i % 5 == 0:
                G.zero_grad()
                outputs = D(fake_images)
                g_loss = -torch.mean(outputs)
                g_loss.backward()
                optimizer_G.step()
            # Print and save model
            if (i + 1) % 400 == 0 or (i + 1) == len(dataloader):
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}"
                )
            if save_interval and (epoch + 1) % save_interval == 0:
                torch.save(G.state_dict(), f"{save_path}generator_epoch_{epoch+1}.pth")
                torch.save(
                    D.state_dict(), f"{save_path}discriminator_epoch_{epoch+1}.pth"
                )
    # Final model save
    torch.save(G.state_dict(), f"{save_path}generator_final.pth")
    torch.save(D.state_dict(), f"{save_path}discriminator_final.pth")
    print("Training completed.")


# Initialize and set up for training
input_dim = 100
output_dim = 784  # e.g., 28x28 images
hidden_dim = 128
batch_size = 64
lr = 0.00005
num_epochs = 50
lambda_gp = 10  # Gradient penalty lambda hyperparameter
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1] range
    ]
)
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
G = Generator(input_dim, output_dim, hidden_dim)
D = Discriminator(output_dim, hidden_dim)
optimizer_G = optim.RMSprop(G.parameters(), lr=lr)
optimizer_D = optim.RMSprop(D.parameters(), lr=lr)

# Start training and save models periodically
train_GAN(
    num_epochs,
    dataloader,
    G,
    D,
    optimizer_G,
    optimizer_D,
    lambda_gp,
    save_interval=10,
    save_path="./model_weights/",
)
