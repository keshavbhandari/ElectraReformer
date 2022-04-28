# Mixed-Dimensions Trick
#
# Description: Applies mixed dimension trick to embeddings to reduce
# embedding sizes.
#
# References:
# [1] Antonio Ginart, Maxim Naumov, Dheevatsa Mudigere, Jiyan Yang, James Zou,
# "Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation
# Systems", CoRR, arXiv:1909.11810, 2019
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
from Utils.helper import count_parameters


class PrEmbeddingBag(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=64, base_dim=128):
        super(PrEmbeddingBag, self).__init__()
        self.embs = nn.Embedding(num_embeddings, embedding_dim)
        torch.nn.init.xavier_uniform_(self.embs.weight)
        if embedding_dim < base_dim:
            self.proj = nn.Linear(embedding_dim, base_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.proj.weight)
        elif embedding_dim == base_dim:
            self.proj = nn.Identity()
        else:
            raise ValueError(
                "Embedding dim " + str(embedding_dim) + " > base dim " + str(base_dim)
            )

    def forward(self, input):
        return self.proj(self.embs(input))


if __name__ == '__main__':
    embedding = PrEmbeddingBag(num_embeddings=20000, embedding_dim=64, base_dim=128)
    count_parameters(embedding)
    data = torch.randint(0, 20000, (4,256))
    out = embedding(data)
    print(out.shape)