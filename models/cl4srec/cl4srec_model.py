# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .modules import SelfAttention
from . import config


class CL4SRec(nn.Module):
    def __init__(self, num_items, args):
        super(CL4SRec, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        # Item embeddings cannot be shared between clients, because the number
        # of items in each domain is different.
        self.item_emb = nn.Embedding(num_items + 1, config.hidden_size,
                                     padding_idx=num_items)
        # TODO: loss += args.l2_emb for regularizing embedding vectors during
        # training https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.pos_emb = nn.Embedding(
            args.max_seq_len, config.hidden_size)

        self.encoder = SelfAttention(num_items, args)

        # The last prediction layer cannot be shared between clients, because
        # the number of items in each domain is different.
        self.linear = nn.Linear(config.hidden_size, num_items)
        self.linear_pad = nn.Linear(config.hidden_size, 1)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_rate)

    def add_position_embedding(self, seqs, seq_embeddings):
        seq_length = seqs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=seqs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seqs)
        position_embeddings = self.pos_emb(position_ids)
        seq_embeddings += position_embeddings
        seq_embeddings = self.LayerNorm(seq_embeddings)
        seq_embeddings = self.dropout(seq_embeddings)
        return seq_embeddings  # (batch_size, seq_len, hidden_size)

    def forward(self, seqs, aug_seqs1=None, aug_seqs2=None):
        seqs_emb = self.item_emb(seqs)
        # (batch_size, seq_len, hidden_size)
        seqs_emb *= self.item_emb.embedding_dim ** 0.5
        # (batch_size, seq_len, hidden_size)
        seqs_emb = self.add_position_embedding(seqs, seqs_emb)

        seqs_fea = self.encoder(seqs_emb, seqs)

        if self.training:
            aug_seqs_emb1 = self.item_emb(aug_seqs1)
            aug_seqs_emb2 = self.item_emb(aug_seqs2)
            # (batch_size, seq_len, hidden_size)
            aug_seqs_emb1 *= self.item_emb.embedding_dim ** 0.5
            # (batch_size, seq_len, hidden_size)
            aug_seqs_emb2 *= self.item_emb.embedding_dim ** 0.5
            aug_seqs_emb1 = self.add_position_embedding(
                aug_seqs1, aug_seqs_emb1)  # (batch_size, seq_len, hidden_size)
            aug_seqs_emb2 = self.add_position_embedding(
                aug_seqs2, aug_seqs_emb2)  # (batch_size, seq_len, hidden_size)
            aug_seqs_fea1 = self.encoder(aug_seqs_emb1, aug_seqs1)
            aug_seqs_fea2 = self.encoder(aug_seqs_emb2, aug_seqs2)

        result = self.linear(seqs_fea)
        result_pad = self.linear_pad(seqs_fea)

        if self.training:
            return torch.cat((result, result_pad), dim=-1), \
                aug_seqs_fea1, aug_seqs_fea2
        else:
            return torch.cat((result, result_pad), dim=-1)
