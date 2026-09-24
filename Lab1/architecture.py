from torch import nn

# Intended for 64 x 64 input images
def encoder_conv_stack(channels=3):
    return nn.Sequential(
        nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 256, kernel_size=4, stride=1, padding=0),
        nn.ReLU(),
    )


def encoder_linear_stack(latent_dim):
    return nn.Sequential(nn.Linear(256, latent_dim * 2))


def decoder_linear_stack(latent_dim):
    return nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
    )


def decoder_conv_stack(channels=3):
    return nn.Sequential(
        nn.ConvTranspose2d(256, 64, kernel_size=4, stride=1, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
        nn.Tanh(), # Range -1, 1 may have to adjust depending on normalization if not we can change to Sigmoid activation
    )

