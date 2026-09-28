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
    
    def encode(self, x):
        x = self.forward(x)
        
        mu = x[:, :self.latent_dim]      # slice along dim=1 (features), keep batch dim intact
        logvar = x[:, self.latent_dim:]  # network predicts log-variance directly -> unconstrained output is fine
        std =  torch.exp(0.5 * logvar)   # guarantees std > 0, and log(var) is "free" for the KL term below

        # KL(q(z|x) || p(z))
        kl_per_dim = 1 + logvar - mu**2 - torch.exp(logvar)  # closed-form diagonal Gaussian vs N(0,I), per-dim
        kl = -0.5*torch.sum(kl_per_dim, 1)                   # sum over latent dim only -> one KL value per example

        return mu, std, kl

    # FOR FUTURE (OPTIONAL) ARCHITECTURE
    # def encode_and_sample(self, x):
    #     x = self.forward(x)

    #     mu = x[:, :self.latent_dim]      # slice along dim=1 (features), keep batch dim intact
    #     logvar = x[:, self.latent_dim:]  # network predicts log-variance directly -> unconstrained output is fine
    #     std =  torch.exp(0.5 * logvar)   # guarantees std > 0, and log(var) is "free" for the KL term below

    #     # Sample
    #     eps = torch.randn_like(mu)       # pure noise, no dependency on mu/logvar -> keeps gradient path clean
    #     z = mu + eps*std                 # reparameterization: z is now a differentiable function of mu, std

    #     # KL(q(z|x) || p(z))
    #     kl_per_dim = 1 + logvar - mu**2 - torch.exp(logvar)  # closed-form diagonal Gaussian vs N(0,I), per-dim
    #     kl = -0.5*torch.sum(kl_per_dim, 1)                   # sum over latent dim only -> one KL value per example

    #     return z, kl


class Decoder(nn.Module):
    def __init__(self, conv_stack, linear_stack, channels, height, width):
        super().__init__()
        self.conv_stack = conv_stack      # registered as submodule -> shows up in .parameters(), gets trained
        self.linear_stack = linear_stack  # same as above
        self.unflatten = nn.Unflatten(1, (channels, height, width))  

    def forward(self, z):
        x_hat = self.linear_stack(z)
        x_hat = self.unflatten(x_hat)
        x_hat = self.conv_stack(x_hat)

        return x_hat


class multiviewVAE(nn.Module):
    def __init__(self, encoder1, encoder2, decoder1, decoder2, var_x1, var_x2):
        super().__init__()
        # Observation variance of p(x|z) (Fixed variance for decoder less training)
        # Large values prioritize KL divergence, small values focus on more sharp visual details
        self.var_x1 = var_x1
        self.var_x2 = var_x2

        self.encoder1 = encoder1
        self.decoder1 = decoder1
        self.encoder2 = encoder2
        self.decoder2 = decoder2

    def forward(self, x):
        x1, x2 = x[0], x[1]
        mu1, std1, kl1 = self.encoder1.encode(x1)
        mu2, std2, kl2 = self.encoder2.encode(x2)

        z1 = self.sample(mu1, std1)    
        z2 = self.sample(mu2, std2)   

        # Cross-reconstruction
        x11_hat = self.decoder1(z1)
        x21_hat = self.decoder2(z1)
        x12_hat = self.decoder1(z2)
        x22_hat = self.decoder2(z2)

        return x11_hat, x21_hat, x12_hat, x22_hat, kl1, kl2
        
    def sample(self, mu, std):
        # Sample
        eps = torch.randn_like(mu)       # pure noise, no dependency on mu/logvar -> keeps gradient path clean
        z = mu + eps*std                 # reparameterization: z is now a differentiable function of mu, std

        return z

    def loss_function(self, x1, x2, x11_hat, x21_hat, x12_hat, x22_hat, kl1, kl2):
        batch_size = kl1.shape[0]

        # Log-Likelihood term
        logp11, _ = eval_Gaussian_LL(x1, x11_hat, self.var_x1)
        logp11 /= batch_size
        logp21, _ = eval_Gaussian_LL(x2, x21_hat, self.var_x2)
        logp21 /= batch_size
        logp12, _ = eval_Gaussian_LL(x1, x12_hat, self.var_x1)
        logp12 /= batch_size
        logp22, _ = eval_Gaussian_LL(x2, x22_hat, self.var_x2)
        logp22 /= batch_size

        # Sum KLs over batch
        kl1 = kl1.sum()/batch_size
        kl2 = kl2.sum()/batch_size

        # Minimizing ELBO loss
        elbo_loss = -0.5 * ((logp11 + logp21 - kl1) + (logp12 + logp22 - kl2)) 

        return elbo_loss, logp11, logp21, logp12, logp22, kl1, kl2


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

