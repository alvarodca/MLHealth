import torch
from torch import nn, optim
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    def __init__(self, latent_dim, private_dim, conv_stack, linear_stack):
        """
        Encoder class. All encoders defined take the vector and project to both a shared latent and a private latent.
        latent_dim : Dimensions for the shared latent
        private_dim : Dimensions for the private latent
        conv_stack: Convolutional NN, declared separately
        linear_stack: FFNN, declared separately"""
        super().__init__()
        self.latent_dim = latent_dim
        self.private_dim = private_dim
        self.flatten = nn.Flatten()
        self.conv_stack = conv_stack      # registered as submodule -> shows up in .parameters(), gets trained
        self.linear_stack = linear_stack  # same as above

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.linear_stack(x)
        return x

    def encode(self, x):
        x = self.forward(x) # Output is a single vector of length 2 (latent_dim + private_dim). It outputs two means and two variances, which we have to split

        # Output is split for the shared latent and the private one
        # We could replace this with two separate linear stacks instead
        x_shared = x[:, : 2 * self.latent_dim]
        x_private = x[:, 2 * self.latent_dim :]

        # Shared Latent will probably provide information about the digits and its structure
        mu_s = x_shared[:, :self.latent_dim]
        logvar_s = x_shared[:, self.latent_dim:]
        std_s = torch.exp(0.5 * logvar_s)
        kl_per_dim_s = 1 + logvar_s - mu_s**2 - torch.exp(logvar_s)
        kl_s = -0.5 * torch.sum(kl_per_dim_s, 1)

        # Private latent will probably provide the information that distinguishes the SVHN from the MNIST
        mu_p = x_private[:, :self.private_dim]
        logvar_p = x_private[:, self.private_dim:]
        std_p = torch.exp(0.5 * logvar_p)
        kl_per_dim_p = 1 + logvar_p - mu_p**2 - torch.exp(logvar_p)
        kl_p = -0.5 * torch.sum(kl_per_dim_p, 1)

        return (mu_s, std_s, kl_s), (mu_p, std_p, kl_p)


class Decoder(nn.Module):
    def __init__(self, conv_stack, linear_stack, channels, height, width):
        """
        Decoder Class. 
        conv_stack and linear_stack same as before.
        Channels, height and width for unflattening to arrive at previous dimensions
        """
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
    def __init__(self, encoder1, encoder2, decoder1, decoder2, var_x1, var_x2, private_dim_mnist, private_dim_svhn):
        """
        MultiviewVAE.
        Consists of 2 encoders and 2 decoders. Each encoder encodes into a private and a shared latent.
        Decoder obtains information from both the shared and the private latent.
        """
        super().__init__()
        # Large values prioritize KL divergence, small values focus on more sharp visual details
        self.var_x1 = var_x1
        self.var_x2 = var_x2

        # MNIST
        self.encoder1 = encoder1 
        self.decoder1 = decoder1
        self.private_dim_mnist = private_dim_mnist

        # SVHN
        self.encoder2 = encoder2
        self.decoder2 = decoder2
        self.private_dim_svhn = private_dim_svhn

    def forward(self, x):
        x1, x2 = x[0], x[1]

        # Encoding MNIST
        (mu_s1, std_s1, kl_s1), (mu_p1, std_p1, kl_p1) = self.encoder1.encode(x1)
        z_s1 = self.sample(mu_s1, std_s1)
        z_p1 = self.sample(mu_p1, std_p1) 

        # Encoding SVHN
        (mu_s2, std_s2, kl_s2), (mu_p2, std_p2, kl_p2) = self.encoder2.encode(x2)
        z_s2 = self.sample(mu_s2, std_s2)
        z_p2 = self.sample(mu_p2, std_p2)   

        # Decoding self reconstruction
        x11_hat = self.decoder1(torch.cat([z_s1, z_p1], dim=-1))
        x22_hat = self.decoder2(torch.cat([z_s2, z_p2], dim=-1))

        # Cross-reconstruction (Prior for private lateents)
        x12_hat = self.decoder1(torch.cat([z_s2, z_p1], dim=-1)) # MNIST decoder using SVHN shared + MNIST private
        x21_hat = self.decoder2(torch.cat([z_s1, z_p2], dim=-1)) # SVHN decoder using MNIST shared + SVHN private


        return x11_hat, x21_hat, x12_hat, x22_hat, kl_s1, kl_p1, kl_s2, kl_p2, mu_s1, mu_s2
        
    def sample(self, mu, std):
        # Sample
        eps = torch.randn_like(mu)       # pure noise, no dependency on mu/logvar -> keeps gradient path clean
        z = mu + eps*std                 # reparameterization: z is now a differentiable function of mu, std

        return z

    def loss_function(self, x1, x2, x11_hat, x21_hat, x12_hat, x22_hat, kl_s1, kl_p1, kl_s2, kl_p2, mu_s1, mu_s2):
        batch_size = x1.shape[0]

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
        kl_s1 = kl_s1.sum() / batch_size
        kl_p1 = kl_p1.sum() / batch_size
        kl_s2 = kl_s2.sum() / batch_size
        kl_p2 = kl_p2.sum() / batch_size

        # Alignment loss, makes sure that the shared mean vector is close in value
        align_loss = F.mse_loss(mu_s1, mu_s2, reduction='sum') / x1.size(0)

        # Minimizing ELBO loss
        elbo_mnist = logp11 + logp12 - kl_s1 - kl_p1
        elbo_svhn = logp22 + logp21 - kl_s2 - kl_p2

        # MoE, value of 0.5. Align loss has a weight of 1.0 
        elbo_loss = -0.5 * (elbo_svhn + elbo_mnist) + align_loss

        return elbo_loss, logp11, logp21, logp12, logp22, kl_s1, kl_p1, kl_s2, kl_p2, align_loss

    def train_loop(self, dataloader,  lr=1e-3, epochs = 200, device="cpu"):
        self.to(device)

        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Stores metrics for plotting
        history = {
            # elbo loss, reconstruction loss and alignment loss
            "elbo": [], "rec": [],"align": [],
            # combined kl terms
            "kl1": [], "kl2": [],
            # separate KL terms
            "kl_s1": [], "kl_p1": [], "kl_s2": [], "kl_p2": [],
            # shared vs. private totals across views
            "kl_shared": [], "kl_private": [],
            # per-view / per-path reconstruction
            "logp11": [], "logp21": [], "logp12": [], "logp22": [],
        }

        for epoch in range(epochs):
            self.train()
            sums = {k: 0.0 for k in history}


            for batch in dataloader:
                x1, x2 = batch[0], batch[1]
                x1 = x1.to(device)
                x2 = x2.to(device)
                optimizer.zero_grad()

                # Forward step
                x11_hat, x21_hat, x12_hat, x22_hat, kl_s1, kl_p1, kl_s2, kl_p2, mu_s1, mu_s2 = self.forward((x1, x2))

                # Loss function and backpropagation
                elbo_loss, logp11, logp21, logp12, logp22, kl_s1, kl_p1, kl_s2, kl_p2, align_loss   = self.loss_function(
                    x1, x2, x11_hat, x21_hat, x12_hat, x22_hat, kl_s1, kl_p1, kl_s2, kl_p2, mu_s1, mu_s2
                )

                elbo_loss.backward()
                optimizer.step()

                # Storing values for plotting
                sums["elbo"] += elbo_loss.item()
                sums["rec"] += (logp11 + logp21 + logp12 + logp22).item()
                sums["align"] += align_loss.item()


                sums["kl1"] += (kl_s1 + kl_p1).item()
                sums["kl2"] += (kl_s2 + kl_p2).item()

                sums["kl_s1"] += kl_s1.item()
                sums["kl_p1"] += kl_p1.item()
                sums["kl_s2"] += kl_s2.item()
                sums["kl_p2"] += kl_p2.item()
                sums["kl_shared"] += (kl_s1 + kl_s2).item()
                sums["kl_private"] += (kl_p1 + kl_p2).item()

                sums["logp11"] += logp11.item()
                sums["logp21"] += logp21.item()
                sums["logp12"] += logp12.item()
                sums["logp22"] += logp22.item()

            n = len(dataloader)
            for k in history:
                history[k].append(sums[k] / n)
                
            print(
            f"Epoch {epoch + 1} | "
            f"ELBO loss: {history['elbo'][-1]:.3f} | "
            f"LL: {history['rec'][-1]:.3f} | "
            f"Align Loss: {history['align'][-1]:.3f} | "
            f"KL shared (s1/s2): {history['kl_s1'][-1]:.3f}/{history['kl_s2'][-1]:.3f} | "
            f"KL private (p1/p2): {history['kl_p1'][-1]:.3f}/{history['kl_p2'][-1]:.3f}"
            )

        return history

    # Generates new images from a Normal Prior
    @torch.no_grad()
    def generation(self, num_imgs=10, device="cpu"):
        self.eval()

        # Providing a random z vector following the prior
        eps_s = torch.randn(num_imgs, self.encoder1.latent_dim, device=device)
        eps_p1 = torch.randn(num_imgs, self.private_dim_mnist, device=device)
        eps_p2 = torch.randn(num_imgs, self.private_dim_svhn, device=device)

        # Decoding the random vector to generate the images
        mnist_gen = self.decoder1(torch.cat([eps_s, eps_p1], dim=-1))
        svhn_gen = self.decoder2(torch.cat([eps_s, eps_p2], dim=-1))

        return mnist_gen, svhn_gen

    # Encodes an image and reconstructs it
    @torch.no_grad()
    def reconstruction(self, x1=None, x2=None):
        self.eval()
        x1_rec, x2_rec = None, None

        if x1 is not None:
            (mu_s1, _, _), (mu_p1, _, _) = self.encoder1.encode(x1)
            x1_rec = self.decoder1(torch.cat([mu_s1, mu_p1], dim=-1))

        if x2 is not None:
            (mu_s2, _, _), (mu_p2, _, _) = self.encoder2.encode(x2)
            x2_rec = self.decoder2(torch.cat([mu_s2, mu_p2], dim=-1))

        return x1_rec, x2_rec

    # Cross-generation, converts images from one domain to another
    @torch.no_grad()
    def cross_generate(self, x1, x2):
        # x1 must be MNIST type and x2 of SVHN type
        self.eval()
        x1_cross, x2_cross = None, None

        # MNIST to SVHN
        if x1 is not None:
            (mu_s1, _, _), _ = self.encoder1.encode(x1)
            # With this random from the private latent space we are picking a random SVHN background style from the learned prior, to combine it with the digit from the MNIST
            # This is because the decoder needs two vectors (one from the shared latent, another from the private latent)
            eps_p2 = torch.randn(x1.size(0), self.private_dim_svhn, device=x1.device)
            x2_cross = self.decoder2(torch.cat([mu_s1, eps_p2], dim=-1))
              
        # SVHN to MNIST
        if x2 is not None:
            (mu_s2, _, _), _ = self.encoder2.encode(x2)
            eps_p1 = torch.randn(x2.size(0), self.private_dim_mnist, device=x2.device)
            x1_cross = self.decoder1(torch.cat([mu_s2, eps_p1], dim=-1))

        return x1_cross, x2_cross

# Auxiliary function which evaluates the log-likelihood of an independent Gaussian using given the mean and covariance
def eval_Gaussian_LL(x, mu_x, var_x):
    x_flat = x.flatten(start_dim = 1)
    mu_flat = mu_x.flatten(start_dim = 1)
    D = x_flat.shape[1] # 784 for MNIST, 3072 for SVHN, with this, no normalization is applied as this already gives higher weights to SVHN

    squared_error = torch.sum((x_flat - mu_flat) ** 2, dim=-1)
    # Constant term from GLL
    cnt = D * (math.log(2 * math.pi) + math.log(var_x))

    # Log-likelihood per data point
    loglike_point = -0.5* (cnt + squared_error/var_x)

    # Normalizing the dimensions
    loglike = torch.sum(loglike_point)

    return loglike, loglike_point

