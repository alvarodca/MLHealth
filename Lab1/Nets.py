import torch
from torch import nn
import math

class Encoder(nn.Module):
    def __init__(self, latent_dim, conv_stack, linear_stack):
        super().__init__()
        self.latent_dim = latent_dim
        self.flatten = nn.Flatten()
        self.conv_stack = conv_stack      # registered as submodule -> shows up in .parameters(), gets trained
        self.linear_stack = linear_stack  # same as above
    
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.linear_stack(x)

        return x

    def encode_and_sample(self, x):
        x = self.forward(x)

        mu = x[:, :self.latent_dim]      # slice along dim=1 (features), keep batch dim intact
        logvar = x[:, self.latent_dim:]  # network predicts log-variance directly -> unconstrained output is fine
        std =  torch.exp(0.5 * logvar)   # guarantees std > 0, and log(var) is "free" for the KL term below

        # Sample
        eps = torch.randn_like(mu)       # pure noise, no dependency on mu/logvar -> keeps gradient path clean
        z = mu + eps*std                 # reparameterization: z is now a differentiable function of mu, std

        # KL(q(z|x) || p(z))
        kl_per_dim = 1 + logvar - mu**2 - torch.exp(logvar)  # closed-form diagonal Gaussian vs N(0,I), per-dim
        kl = -0.5*torch.sum(kl_per_dim, 1)                   # sum over latent dim only -> one KL value per example

        return z, kl

class Decoder(nn.Module):
    def __init__(self, conv_stack, linear_stack, channels, height, width):
        super().__init__()
        self.conv_stack = conv_stack      # registered as submodule -> shows up in .parameters(), gets trained
        self.linear_stack = linear_stack  # same as above
        self.deflatten = lambda x: torch.reshape(x, (-1, channels, height, width))  # -1 infers batch size at call time

    def forward(self, z):
        x_hat = self.linear_stack(z)
        x_hat = self.deflatten(x_hat)
        x_hat = self.conv_stack(x_hat)

        return x_hat

class VAE(nn.Module):
    def __init__(self, encoder, decoder, var_x):
        super().__init__()
        # Observation variance of p(x|z) (Fixed variance for decoder less training)
        # Large values prioritize KL divergence, small values focus on more sharp visual details
        self.var_x = var_x

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z, kl = self.encoder.encode_and_sample(x)
        x_hat = self.decoder(z)

        return x_hat, z, kl

    def loss_function(self, x, x_hat, kl):
        # Log-Likelihood term
        logp, _ = eval_Gaussian_LL(x, x_hat, self.var_x)

        # KL term
        total_kl = torch.sum(kl)

        # Minimizing ELBO loss
        elbo_loss = -logp + total_kl

        return elbo_loss, -logp, total_kl

# Auxiliary function which evaluates the log-likelihood of an independent Gaussian using given the mean and covariance
def eval_Gaussian_LL(x, mu_x, var_x):
    x_flat = x.flatten(start_dim = 1)
    mu_flat = mu_x.flatten(start_dim = 1)

    # Feature dimension
    D = x_flat.shape[1]

    # Constant term in the Gaussian distribution
    cnt = D * (math.log(2*math.pi) + math.log(var_x))

    # Squared errors
    squared_error = torch.sum((x_flat - mu_flat) ** 2, dim=-1)

    # Log-likelihood per data point
    loglike_point = -0.5* (cnt + squared_error/var_x)

    loglike = torch.sum(loglike_point)

    return loglike, loglike_point

