# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .modules import SelfAttention
from . import config


class Encoder(nn.Module):
    def __init__(self, num_items, args):
        super(Encoder, self).__init__()
        self.encoder_mu = SelfAttention(num_items, args)
        self.encoder_logvar = SelfAttention(num_items, args)

    def forward(self,  seqs, seqs_data):
        """
        seqs: (batch_size, seq_len, hidden_size)
        seqs_data: (batch_size, seq_len)
        """
        mu = self.encoder_mu(seqs, seqs_data)
        logvar = self.encoder_logvar(seqs, seqs_data)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, num_items, args):
        super(Decoder, self).__init__()
        self.decoder = SelfAttention(num_items, args)

    def forward(self, seqs, seqs_data):
        """
        seqs: (batch_size, seq_len, hidden_size)
        seqs_data: (batch_size, seq_len)
        """
        feat_seq = self.decoder(seqs, seqs_data)
        return feat_seq


class VSAN(nn.Module):
    def __init__(self, num_items, args):
        super(VSAN, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        # Item embeddings cannot be shared between clients, because the number
        # of items in each domain is different.
        self.item_emb = nn.Embedding(
            num_items + 1, config.hidden_size, padding_idx=num_items)
        # TODO: loss += args.l2_emb for regularizing embedding vectors during
        # training https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.pos_emb = nn.Embedding(
            args.max_seq_len, config.hidden_size)

        self.encoder = Encoder(num_items, args)
        self.decoder = Decoder(num_items, args)
        # The last prediction layer cannot be shared between clients, because
        # the number of items in each domain is different.
        self.linear = nn.Linear(config.hidden_size, num_items)
        self.linear_pad = nn.Linear(config.hidden_size, 1)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.3)

    def add_position_embedding(self, seqs, seq_embeddings):
        seq_length = seqs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=seqs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seqs)
        position_embeddings = self.pos_emb(position_ids)
        seq_embeddings += position_embeddings
        seq_embeddings = self.LayerNorm(seq_embeddings)
        seq_embeddings = self.dropout(seq_embeddings)
        return seq_embeddings  # shape: b*max_Sq*d

    def forward(self, seqs):
        seqs_emb = self.item_emb(seqs)
        # (batch_size, seq_len, hidden_size)
        seqs_emb *= self.item_emb.embedding_dim ** 0.5
        # (batch_size, seq_len, hidden_size)
        seqs_emb = self.add_position_embedding(seqs, seqs_emb)

        mu, logvar = self.encoder(seqs_emb, seqs)
        z = self.reparameterization(mu, logvar)

        reconstructed_seq = self.decoder(z, seqs)
        result = self.linear(reconstructed_seq)
        result_pad = self.linear_pad(reconstructed_seq)
        if self.training:
            return torch.cat((result, result_pad), dim=-1), mu, logvar
        else:
            return torch.cat((result, result_pad), dim=-1)

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            res = mu + std * eps
        else:
            res = mu
        return res
