import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter


def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
    return layers


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim, batchNorm=False):
        """
        Generator class for a simple GAN.

        Args:
            z_dim (int): The dimension of the latent noise.
            img_dim (int): The dimension of the generated image.
            batchNorm (bool, optional): Whether to use batch normalization. Defaults to False.
        """
        super().__init__()
        if not batchNorm:
            self.gen = nn.Sequential(
                nn.Linear(z_dim, 128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, img_dim),
                nn.Tanh(),  # tanh is used to make sure the output is between -1 and 1
            )
        else:
            self.gen = nn.Sequential(
                nn.Linear(z_dim, 128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.1),
                nn.Linear(256, img_dim),
                nn.Tanh(),  # tanh is used to make sure the output is between -1 and 1
            )

    def forward(self, x):
        """
        Forward pass of the generator.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The generated output tensor.
        """
        return self.gen(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
lr = 3e-4  # best learning rate for Adam optimizer
z_dim = 64  # latent noise dimension
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
step = 0
normalization_params = [
    # {"mean": 0.5, "std": 0.5, "name": "standard", "batchNorm": False},
    # {"mean": 0, "std": 1, "name": "zero_mean_unit_variance", "batchNorm": False},
    # {"mean": 0.1307, "std": 0.3081, "name": "mnist_normalization", "batchNorm": False},
    {"mean": 0.5, "std": 0.5, "name": "standard_BatchNorm", "batchNorm": True},
    # Add more normalization options here
]

for norm_params in normalization_params:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((norm_params["mean"],), (norm_params["std"],)),
        ]
    )  # transform the input image to tensor and normalize it
    # Create a unique SummaryWriter for each normalization setting
    dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake_{norm_params['name']}")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real_{norm_params['name']}")
    disc = Discriminator(image_dim).to(device)
    gen = Generator(z_dim, image_dim, batchNorm=norm_params["batchNorm"]).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.BCELoss()
    step = 0
    for epoch in range(num_epochs):
        for batch_idx, (img, labels) in enumerate(loader):
            real = img.view(-1, 784).to(device)
            batch_size = real.shape[0]

            ### Train Discriminator: max log(D(real)) + log(1-D(G(z)))
            noise = torch.randn(batch_size, z_dim).to(
                device
            )  # this noise is from Guassian Distribution for Mean 0 and Standard Deviation 1
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(
                -1
            )  # can use fake.detach() to aretain computational graph
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            ### Train Generator: min log(1-D(G(z))) <-> max log(D(G(z)))
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \ "
                    f"Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )
                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real.add_image(
                        "Mnist Real Images", img_grid_real, global_step=step
                    )
                    step += 1
