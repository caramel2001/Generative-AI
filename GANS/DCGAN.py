import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator network for DCGAN.

    Args:
        channels_img (int): Number of input channels for the image.
        features_d (int): Number of features in the discriminator.

    Attributes:
        disc (nn.Sequential): Sequential model representing the discriminator network.

    Methods:
        _block: Helper method to create a convolutional block.
        forward: Forward pass of the discriminator network.

    """

    def __init__(self, channels_img, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Helper method to create a convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride value for the convolution.
            padding (int): Padding value for the convolution.

        Returns:
            nn.Sequential: Sequential model representing the convolutional block.

        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),  # 0.2 : from the DCGAN papar
        )

    def forward(self, x):
        """
        Forward pass of the discriminator network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1 (N : Batch Size)
            self._block(z_dim, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # difference between Conv2d and ConvTranspose2d
            # ConvTranspose2d transposed convolution or deconvolution layer, this layer is used for upsampling or expanding the input. It's commonly used in models where you need to increase the spatial resolution, such as in Generative Adversarial Networks (GANs).
            # Conv2d is standard convolutional layer used in Convolutional Neural Networks (CNNs). It's primarily used for downsampling or compressing the input. This operation is useful in feature extraction and learning spatial hierarchies.
            nn.Tanh(),  # [-1,1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Helper method to create a convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride value for the convolution.
            padding (int): Padding value for the convolution.

        Returns:
            nn.Sequential: Sequential model representing the convolutional block.

        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),  # ReLU instead of LeakyReLU : Based on the DCGAN paper
        )

    def forward(self, x):
        """
        Forward pass of the generator network.
        """
        return self.gen(x)


def initialize_weights(model):
    """
    Initialize weights of the model. Based on DCGAN paper.

    Args:
        model (nn.Module): Model to be initialized.

    """
    # Initialize the weights
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Tests passed")


test()

# training Pipeline
import torch.optim
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

celebA = False
# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc , values if from DCGAN paper
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3 if celebA else 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)
# Change dataset to Celeb A
if celebA:
    dataset = datasets.CelebA(
        root="dataset/", train=True, transform=transforms, download=True
    )
else:
    dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms, download=True
    )
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = torch.optim.Adam(
    gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)  # beta values from DCGAN paper. First Beta is beta for momentum and second beta is beta for average of squared gradient
opt_disc = torch.optim.Adam(
    disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)  # beta values from DCGAN paper


criterion = nn.BCELoss()
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = (
    SummaryWriter(f"logs/real") if not celebA else SummaryWriter(f"logs/celebA_real")
)
writer_fake = (
    SummaryWriter(f"logs/fake") if not celebA else SummaryWriter(f"logs/celebA_fake")
)
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)  # detach to not train generator
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()  # Clear all previous gradients
        loss_disc.backward(
            retain_graph=True
        )  # we need to do rretain_graph=True because we are using the same graph for the generator
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            step += 1
