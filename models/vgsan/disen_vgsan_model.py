# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .modules import SelfAttention
from .gnn import GCNLayer
from . import config


class Encoder(nn.Module):
    def __init__(self, num_items, args):
        super(Encoder, self).__init__()
        self.encoder_mu = SelfAttention(num_items, args)
        self.encoder_logvar = SelfAttention(num_items, args)

    def forward(self, seqs, seqs_data):
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


class DisenVGSAN(nn.Module):
    def __init__(self, num_items, args):
        super(DisenVGSAN, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        # Item embeddings cannot be shared between clients, because the number
        # of items in each domain is different.
        self.item_emb_s = nn.Embedding(
            num_items + 1, config.hidden_size, padding_idx=num_items)
        self.item_emb_e = nn.Embedding(
            num_items + 1, config.hidden_size, padding_idx=num_items)
        # TODO: loss += args.l2_emb for regularizing embedding vectors during
        # training https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.pos_emb_s = nn.Embedding(args.max_seq_len, config.hidden_size)
        self.pos_emb_e = nn.Embedding(args.max_seq_len, config.hidden_size)
        self.GNN_encoder_s = GCNLayer(args)
        self.GNN_encoder_e = GCNLayer(args)

        self.encoder_s = Encoder(num_items, args)
        self.encoder_e = Encoder(num_items, args)
        self.decoder = Decoder(num_items, args)

        # The last prediction layer cannot be shared between clients, because
        # the number of items in each domain is different.
        self.linear = nn.Linear(config.hidden_size, num_items)
        self.linear_pad = nn.Linear(config.hidden_size, 1)

        self.LayerNorm_s = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm_e = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_rate)

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def graph_convolution(self, adj):
        self.item_index_s = torch.arange(
            0, self.item_emb_s.num_embeddings, 1).to(self.device)
        self.item_index_e = torch.arange(
            0, self.item_emb_e.num_embeddings, 1).to(self.device)
        item_embs_s = self.my_index_select_embedding(
            self.item_emb_s, self.item_index_s)
        item_embs_e = self.my_index_select_embedding(
            self.item_emb_e, self.item_index_e)
        self.item_graph_embs_s = self.GNN_encoder_s(item_embs_s, adj)
        self.item_graph_embs_e = self.GNN_encoder_e(item_embs_e, adj)

    def get_position_ids(self, seqs):
        seq_length = seqs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=seqs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seqs)
        return position_ids

    def add_position_embedding_s(self, seqs, seq_embeddings):
        position_ids = self.get_position_ids(seqs)
        position_embeddings = self.pos_emb_s(position_ids)
        seq_embeddings += position_embeddings
        seq_embeddings = self.LayerNorm_s(seq_embeddings)
        seq_embeddings = self.dropout(seq_embeddings)
        return seq_embeddings  # (batch_size, seq_len, hidden_size)

    def add_position_embedding_e(self, seqs, seq_embeddings):
        position_ids = self.get_position_ids(seqs)
        position_embeddings = self.pos_emb_e(position_ids)
        seq_embeddings += position_embeddings
        seq_embeddings = self.LayerNorm_e(seq_embeddings)
        seq_embeddings = self.dropout(seq_embeddings)
        return seq_embeddings  # (batch_size, seq_len, hidden_size)

    def forward(self, seqs, neg_seqs=None, aug_seqs=None):
        # `item_graph_embs` stores the embeddings of all items.
        # Here we need to select the embeddings of items appearing in the
        # sequence
        seqs_emb_s = self.my_index_select(
            self.item_graph_embs_s, seqs) + self.item_emb_s(seqs)
        seqs_emb_e = self.my_index_select(
            self.item_graph_embs_e, seqs) + self.item_emb_e(seqs)
        # (batch_size, seq_len, hidden_size)
        seqs_emb_s *= self.item_emb_s.embedding_dim ** 0.5
        # (batch_size, seq_len, hidden_size)
        seqs_emb_e *= self.item_emb_e.embedding_dim ** 0.5
        seqs_emb_s = self.add_position_embedding_s(
            seqs, seqs_emb_s)  # (batch_size, seq_len, hidden_size)
        seqs_emb_e = self.add_position_embedding_e(
            seqs, seqs_emb_e)  # (batch_size, seq_len, hidden_size)

        # Here is a shortcut operation that adds up the embeddings of items
        # convolved by GNN and those that have not been convolved.
        if self.training:
            neg_seqs_emb = self.my_index_select(
                self.item_graph_embs_e, neg_seqs) + self.item_emb_e(neg_seqs)
            aug_seqs_emb = self.my_index_select(
                self.item_graph_embs_e, aug_seqs) + self.item_emb_e(aug_seqs)
            # (batch_size, seq_len, hidden_size)
            neg_seqs_emb *= self.item_emb_e.embedding_dim ** 0.5
            # (batch_size, seq_len, hidden_size)
            aug_seqs_emb *= self.item_emb_e.embedding_dim ** 0.5
            neg_seqs_emb = self.add_position_embedding_e(
                neg_seqs, neg_seqs_emb)  # (batch_size, seq_len, hidden_size)
            aug_seqs_emb = self.add_position_embedding_e(
                aug_seqs, aug_seqs_emb)  # (batch_size, seq_len, hidden_size)

        mu_s, logvar_s = self.encoder_s(seqs_emb_s, seqs)
        z_s = self.reparameterization(mu_s, logvar_s)

        if self.training:
            neg_mu_e, neg_logvar_e = self.encoder_e(neg_seqs_emb, neg_seqs)
            neg_z_e = self.reparameterization(neg_mu_e, neg_logvar_e)

            aug_mu_e, aug_logvar_e = self.encoder_e(aug_seqs_emb, aug_seqs)
            aug_z_e = self.reparameterization(aug_mu_e, aug_logvar_e)

        mu_e, logvar_e = self.encoder_e(seqs_emb_e, seqs)
        z_e = self.reparameterization(mu_e, logvar_e)
        # if not self.training:
        #     # reconstructed_seq = self.decoder(z_s)
        #     # reconstructed_seq = self.decoder(z_e)
        #     reconstructed_seq = self.decoder(z_s + z_e)
        # else:
        #     reconstructed_seq = self.decoder(z_s + z_e)

        if not self.training:
            # result = self.linear(z_s)
            # result_pad = self.linear_pad(z_s)
            # result = self.linear(z_e)
            # result_pad = self.linear_pad(z_e)
            result = self.linear(z_s + z_e)
            result_pad = self.linear_pad(z_s + z_e)
        else:
            result = self.linear(z_s + z_e)
            result_pad = self.linear_pad(z_s + z_e)
        # reconstructed_seq_exclusive = self.decoder(z_e)
        result_exclusive = self.linear(z_e)
        result_exclusive_pad = self.linear_pad(z_e)
        if self.training:
            return torch.cat((result, result_pad), dim=-1), \
                torch.cat((result_exclusive, result_exclusive_pad), dim=-1), \
                mu_s, logvar_s, z_s, mu_e, logvar_e, z_e, neg_z_e, aug_z_e
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
