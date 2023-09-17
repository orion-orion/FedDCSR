# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class NCELoss(nn.Module):
    """Noise-Contrastive Estimation (infoNCE).
    """

    def __init__(self, temperature):
        super(NCELoss, self).__init__()
        self.cs_criterion = nn.CrossEntropyLoss(reduction="none")
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)

    # modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    # (batch_size, hidden_size) (batch_size, hidden_size)
    def forward(self, batch_sample_one, batch_sample_two):
        # (batch_size, batch_size)
        sim11 = self.cossim(batch_sample_one.unsqueeze(
            1), batch_sample_one.unsqueeze(0)) / self.temperature
        # (batch_size, batch_size)
        sim22 = self.cossim(batch_sample_two.unsqueeze(
            1), batch_sample_two.unsqueeze(0)) / self.temperature
        # (batch_size, batch_size)
        sim12 = self.cossim(batch_sample_one.unsqueeze(
            1), batch_sample_two.unsqueeze(0)) / self.temperature

        d = sim12.shape[-1]
        # (batch_size, batch_size)，values on the diagonal are set to -inf
        sim11[range(d), range(d)] = float("-inf")
        # (batch_size, batch_size)，values on the diagonal are set to -inf
        sim22[range(d), range(d)] = float("-inf")
        # (batch_size, 2 * batch_size)
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        # (batch_size, 2 * batch_size)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        # (2 * batch_size, 2 * batch_size)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long,
                              device=logits.device)  # (2 * batch_size, )
        nce_loss = self.cs_criterion(logits, labels)  # (2 * batch_size, )
        return nce_loss


class Discriminator(nn.Module):
    def __init__(self, hidden_size, max_seq_len):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_size * 2 * max_seq_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, input1, input2, ground_mask):
        input1 *= ground_mask.unsqueeze(-1)  # broadcast in last dim
        input2 *= ground_mask.unsqueeze(-1)  # broadcast in last dim
        # (batch_size, seq_len * hidden_size)
        input1 = torch.flatten(input1, start_dim=1)
        # (batch_size, seq_len * hidden_size)
        input2 = torch.flatten(input2, start_dim=1)

        input = torch.cat((input1, input2), dim=-1)
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class HingeLoss(nn.Module):
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, pos, neg):
        pos = F.sigmoid(pos)
        neg = F.sigmoid(neg)
        gamma = torch.tensor(self.margin).to(pos.device)
        return F.relu(gamma - pos + neg)


class JSDLoss(torch.nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, pos, neg):
        pos = -F.softplus(-pos)
        neg = F.softplus(neg)
        return neg - pos


def priorKL(alpha):
    """Used in ContrastVAE.
    """
    c1 = 1.161451241083230
    c2 = -1.502041176441722
    c3 = 0.586299206427007
    return -0.5 * torch.log(alpha) + c1 * alpha + c2 * torch.pow(alpha, 2) + c3 * torch.pow(alpha, 3)


class BiDiscriminator(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_in, n_out, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, S, node, s_bias=None):
        score = self.f_k(node, S)
        if s_bias is not None:
            score += s_bias
        return score
