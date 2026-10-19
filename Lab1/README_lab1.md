## Brief Summary

The following project creates a multi-view Variational Autoencoder using a Mixture of Experts (MoE) approach for encoding both MNIST and SVHN datasets in a latent space. To be able to generate new images and to cross-generate using both datasets.

## Methodology

MNIST consisted of 60,000 images of shape 28*28 with a single color channel, whereas SVHN consisted of 73,527 images of shape 32*32 with three color channels. This higher complexity in SVHN was a key factor considered in our architectural design.

Our multiview VAE used one Encoder and one Decoder initialized for each dataset using a modular approach. The architecture comprised a convolutional stack followed by a linear stack. Both encoders used 4 Conv2D layers and both decoders used 4 ConvTranspose2D layers. For the SVHN dataset, Group Normalization was used instead of Batch Normalization as it performed significantly better; Batch Normalization computes statistics across the entire batch, causing an image's encoding to depend on other samples, leading to information leakage.

LeakyReLU was chosen as the activation function because gradient dominance from SVHN caused large negative errors during early training phases that were zeroed out by standard ReLU, negatively impacting convergence. Finally, our linear stacks incorporated an intermediate hidden layer to prevent excessive information compression. Specific hyperparameter details can be found in the code repository.

Each encoder processed images from its respective dataset, projecting them into two separate latent spaces: a private space unique to that dataset and a shared space common to both datasets. Subsequently, each decoder combined the information from both the shared and private latent spaces to reconstruct the output.

The total loss function combined the Kullback-Leibler (KL) divergence terms across all latent spaces with the log-likelihood reconstruction terms using a Mixture of Experts approach with equal weighting. Additionally, an alignment loss was introduced to enforce that both encoders output similar mean vectors in the shared space. Loss terms were left unnormalized to retain strong penalization on SVHN errors, given its structural complexity.

The underlying design intent was for the shared latent space to capture digit identity while the private latent spaces encoded dataset-specific variation, such as stroke thickness or background color.

Training was conducted in two stages: an initial 30-epoch run followed by a 20-epoch fine-tuning stage with a reduced learning rate.

## Challenges Encountered and Future Improvements.

Our initial implementation relied on two completely independent latent spaces. However, cross-generation produced poor results (Figure~\ref{fig:placeholder}), as the MNIST latent space lacked SVHN domain information and defaulted to decoding mean pixel values (resulting in uniform gray outputs).

This led to the implementation of the shared latent space. The critical factor in achieving cross-generation was the alignment loss, which forced both encoders to project digit identity into the shared space.

Other parameters (such as latent variance and private space capacity) required continuous tuning to prevent the model from shortcutting the alignment loss by encoding digit features into private spaces. This tuning was done manually each iteration, using methods like GridSearch or an Optuna search would yield more accurate parameter values

Architectural adjustments---specifically using LeakyReLU and Group Normalization---were also vital for training stability. Additionally, this current architecture worked reasonably well, however, results could be further improved with improvements such as adding more layers, different values for LeakyReLu or maybe trying different activation functions.

## Files

architecture.py -> Defines the Neural Networks architecture
nets.py -> Contains Encoder, Decoder and multiviewVAE classes
training.ipynb -> Loads data and performs the whole training process
generation.ipynb -> Generation, self-reconstruction, cross-generation and T-SNE.