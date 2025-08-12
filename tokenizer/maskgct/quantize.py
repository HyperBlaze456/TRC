# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        commitment=0.005,
        codebook_loss_weight=1.0,
        use_l2_normlize=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.use_l2_normlize = use_l2_normlize

        if self.input_dim != self.codebook_dim:
            self.in_project = WNConv1d(self.input_dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(
                self.codebook_dim, self.input_dim, kernel_size=1
            )
        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def forward(self, z):
        """
        Parameters
        ----------
        z: torch.Tensor[B x D x T]

        Returns
        -------
        z_q: torch.Tensor[B x D x T]
            Quantized continuous representation of input
        commit_loss: Tensor[B]
            Commitment loss to train encoder to predict vectors closer to codebook entries
        codebook_loss: Tensor[B]
            Codebook loss to update the codebook
        indices: torch.Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        z_e: torch.Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes project input into low-dimensional space if self.input_dim != self.codebook_dim
        z_e = self.in_project(z)
        z_q, indices = self.decode_latents(z_e)

        # Compute commitment loss and codebook loss
        if self.training:
            commit_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
                * self.commitment
            )
            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
                * self.codebook_loss_weight
            )
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)
            codebook_loss = torch.zeros(z.shape[0], device=z.device)

        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_project(z_q)

        return z_q, commit_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        if self.use_l2_normlize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

        # Compute euclidean distance between encodings and codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)

        return z_q, indices

    def vq2emb(self, vq, out_proj=True):
        emb = self.decode_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb


class ResidualVQ(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        quantizer_type: str = "vq",  # "vq" or "fvq"
        quantizer_dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_type = quantizer_type
        self.quantizer_dropout = quantizer_dropout

        if quantizer_type == "fvq":
            VQ = FactorizedVectorQuantize
        else:
            raise ValueError(f"Only 'fvq' quantizer type is supported in this minimal version")

        self.quantizers = nn.ModuleList(
            [
                VQ(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z, n_quantizers: int = None):
        """
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use

        Returns
        -------
        "quantized_out" : Tensor[B x D x T]
            Quantized continuous representation of input
        "all_indices" : Tensor[N x B x T]
            Codebook indices for each codebook
        "all_commit_losses" : Tensor[N]
        "all_codebook_losses" : Tensor[N]
        "all_quantized" : Tensor[N x B x D x T]
        """

        quantized_out = 0.0
        residual = z

        all_commit_losses = []
        all_codebook_losses = []
        all_indices = []
        all_quantized = []

        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.num_quantizers + 1
            dropout = torch.randint(1, self.num_quantizers + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commit_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            quantized_out = quantized_out + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commit_loss_i = (commit_loss_i * mask).mean()
            codebook_loss_i = (codebook_loss_i * mask).mean()

            all_commit_losses.append(commit_loss_i)
            all_codebook_losses.append(codebook_loss_i)
            all_indices.append(indices_i)
            all_quantized.append(z_q_i)

        all_commit_losses, all_codebook_losses, all_indices, all_quantized = map(
            torch.stack,
            (all_commit_losses, all_codebook_losses, all_indices, all_quantized),
        )

        return (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            all_quantized,
        )