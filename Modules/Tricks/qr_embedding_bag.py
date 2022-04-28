# Quotient-Remainder Trick
#
# Description: Applies quotient remainder-trick to embeddings to reduce
# embedding sizes.
#
# References:
# [1] Hao-Jun Michael Shi, Dheevatsa Mudigere, Maxim Naumov, Jiyan Yang,
# "Compositional Embeddings Using Complementary Partitions for Memory-Efficient
# Recommendation Systems", CoRR, arXiv:1909.02107, 2019


from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
from Utils.helper import count_parameters


class QREmbeddingBag(nn.Module):
    r"""Computes embeddings, one using the quotient
    of the indices and the other using the remainder of the indices, without
    instantiating the intermediate embeddings, then performs an operation to combine these.
    Args:
        num_categories (int): total number of unique categories. The input indices must be in
                              0, 1, ..., num_categories - 1.
        embedding_dim (int): embedding vector in each table. If ``"add"``
                             or ``"mult"`` operation are used, these embedding dimensions must be
                             the same. If a single embedding_dim is used, then it will use this
                             embedding_dim for both embedding tables.
        num_collisions (int): number of collisions to enforce.
        operation (string, optional): ``"concat"``, ``"add"``, or ``"mult". Specifies the operation
                                      to compose embeddings. ``"concat"`` concatenates the embeddings,
                                      ``"add"`` sums the embeddings, and ``"mult"`` multiplies
                                      (component-wise) the embeddings.
                                      Default: ``"mult"``
    Attributes:
    Inputs: :attr:`input` (LongTensor)
        Input` is 2D of shape `(B, seq_length)`,
        Output shape: `(B, seq_length, embedding_dim)`
    """
    __constants__ = ['num_categories', 'embedding_dim', 'num_collisions',
                     'operation', 'max_norm', 'norm_type', 'scale_grad_by_freq',
                     'mode', 'sparse']

    def __init__(self, num_categories, embedding_dim=128, num_collisions=5, operation='mult'):
        super(QREmbeddingBag, self).__init__()

        assert operation in ['concat', 'mult', 'add'], 'Not valid operation!'

        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.num_collisions = num_collisions
        self.operation = operation

        self.embed_q = nn.Embedding(num_categories//num_collisions, embedding_dim)
        self.embed_r = nn.Embedding(num_categories//num_collisions, embedding_dim)

    def forward(self, input):
        input_q = (input / self.num_collisions).long()
        input_r = torch.remainder(input, self.num_collisions).long()

        embed_q = self.embed_q(input_q)
        embed_r = self.embed_r(input_r)

        if self.operation == 'concat':
            embed = torch.cat((embed_q, embed_r), dim=1)
        elif self.operation == 'add':
            embed = embed_q + embed_r
        elif self.operation == 'mult':
            embed = embed_q * embed_r

        return embed




if __name__ == '__main__':
    embedding = QREmbeddingBag(num_categories=20000, embedding_dim=128, num_collisions=10)
    count_parameters(embedding)
    data = torch.randint(0, 20000, (4,256))
    out = embedding(data)
    print(out.shape)