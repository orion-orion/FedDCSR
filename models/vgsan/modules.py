# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

###################################
#       Activation Function       #
###################################


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        if isinstance(config.hidden_act, str):
            self.feedforward_act_fn = ACT2FN[config.hidden_act]
        else:
            self.feedforward_act_fn = config.hidden_act
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.feedforward_act_fn(
            self.dropout1(self.conv1(inputs.transpose(-1, -2))))
        outputs = self.dropout2(self.conv2(outputs))
        # As Conv1D requires (N, C, Length)
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SelfAttention(nn.Module):
    def __init__(self, num_items, args):
        super(SelfAttention, self).__init__()
        self.num_items = num_items
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        # To be Q for self-attention
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-8)

        for _ in range(config.num_blocks):
            new_attn_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(config.hidden_size,
                                                   config.num_heads,
                                                   config.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(
                config.hidden_size, config.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seqs, seqs_data):
        timeline_mask = torch.BoolTensor(
            seqs_data.cpu() == self.num_items).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1)  # Broadcast in the last dimension

        # Length of the time dimension for enforce causality
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones(
            (tl, tl), dtype=torch.bool)).to(self.device)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats
