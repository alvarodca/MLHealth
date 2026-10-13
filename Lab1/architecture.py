from torch import nn

# ISHVN images are (32 x 32 x 3)
def svhn_encoder_conv_stack(channels=3):
    return nn.Sequential(
        # Input 3 x 32 x 32 -> Output: 32 x 16 x 16
        nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(4, 32),
        nn.LeakyReLU(0.2, inplace=True),

        # Output 64 x 8 x 8
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, 64),
        nn.LeakyReLU(0.2, inplace=True),

        # Output 128 x 4 x 4
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(16, 128),
        nn.LeakyReLU(0.2, inplace=True),

        # Output 256 x 1 x 1
        nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
        nn.GroupNorm(32, 256),
        nn.LeakyReLU(0.2, inplace=True),
    )

def svhn_decoder_conv_stack(channels=3):
    return nn.Sequential(
        # Input: 256 x 1 x 1 -> Output: 128 x 4 x 4
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0),
        nn.GroupNorm(16, 128),
        nn.LeakyReLU(0.2, inplace=True),

        # Output 64 x 8 x 8
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, 64),
        nn.LeakyReLU(0.2, inplace=True),

        # Output 32 x 16 x 16
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(4, 32),
        nn.LeakyReLU(0.2, inplace=True),

        # Output channels x 32 x 32
        nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid(),
    )


# MNIST images are (28 x 28 x1)
def mnist_encoder_conv_stack(channels=1):
    return nn.Sequential(
        # Input: 1 x 28 x 28 -> Output: 32 x 14 x 14
        nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),

        # Output 64 x 7 x 7
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),

        # Output 128 x 4 x 4
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),

        # Output 256 x 1 x 1
        nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
        nn.LeakyReLU(0.2, inplace=True),
    )

def mnist_decoder_conv_stack(channels=1):
    return nn.Sequential(
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0),
        nn.LeakyReLU(0.2, inplace=True),

        # Output: 64 x 7 x 7
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),

        # Output: 32 x 14 x 14
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),

        # Output: channels x 28 x 28
        nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid(),
    )


def mnist_encoder_linear_stack(latent_dim, private_dim):
    return nn.Sequential(
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, (private_dim + latent_dim) * 2)
    )

def svhn_encoder_linear_stack(latent_dim, private_dim):
    return nn.Sequential(
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, (latent_dim + private_dim) * 2)
    )

def mnist_decoder_linear_stack(latent_dim, private_dim):
    return nn.Sequential(
        nn.Linear(latent_dim + private_dim, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True)
    )

def svhn_decoder_linear_stack(latent_dim, private_dim):
    return nn.Sequential(
        nn.Linear(latent_dim + private_dim, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True)
    )


