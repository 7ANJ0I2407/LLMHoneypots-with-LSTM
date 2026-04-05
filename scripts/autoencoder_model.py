import torch
import torch.nn as nn


class SequenceAutoencoder(nn.Module):
    def __init__(
        self,
        window_size: int,
        input_dim: int,
        proj_dim: int = 48,
        latent_dim: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.input_dim = int(input_dim)
        self.proj_dim = int(proj_dim)
        self.latent_dim = int(latent_dim)

        # A compact temporal autoencoder keeps novelty capacity proportional
        # to available normal windows and avoids multi-million-parameter overfit.
        self.in_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.GRU(
            input_size=self.proj_dim,
            hidden_size=self.latent_dim,
            num_layers=1,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=self.latent_dim,
            hidden_size=self.proj_dim,
            num_layers=1,
            batch_first=True,
        )
        self.out_proj = nn.Linear(self.proj_dim, self.input_dim)

    def forward(self, x):
        z_in = self.in_proj(x)
        z_seq, _ = self.encoder(z_in)
        decoded, _ = self.decoder(z_seq)
        return self.out_proj(decoded)


def reconstruction_error(x, recon):
    return ((x - recon) ** 2).mean(dim=(1, 2))
