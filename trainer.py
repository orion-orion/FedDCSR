# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from models.vgsan.disen_vgsan_model import DisenVGSAN
from models.vgsan import config
from models.vgsan.vgsan_model import VGSAN
from models.sasrec.sasrec_model import SASRec
from models.vsan.vsan_model import VSAN
from models.contrastvae.contrastvae_model import ContrastVAE
from models.cl4srec.cl4srec_model import CL4SRec
from models.duorec.duorec_model import DuoRec
from utils import train_utils
from losses import NCELoss, HingeLoss, JSDLoss, Discriminator, priorKL


class Trainer(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def train_batch(self, *args, **kwargs):
        raise NotImplementedError

    def test_batch(self, *args, **kwargs):
        raise NotImplementedError

    def update_lr(self, new_lr):
        train_utils.change_lr(self.optimizer, new_lr)


class ModelTrainer(Trainer):
    def __init__(self, args, num_items, max_seq_len):
        self.args = args
        self.method = args.method
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        if self.method == "FedDCSR":
            self.model = DisenVGSAN(num_items, args).to(self.device)
            # Here we set `self.z_s[:], self.z_g = [None], [None]` so that
            # we can use `self.z_s[:] = ...`, `self.z_g[:] = ...` to modify
            # them later.
            # Note that if we set `self.z_s, self.z_g = None, None`,
            # then `self.z_s = obj` / `self.z_g = obj` will just refer to a
            # new object `obj`, rather than modify `self.z_s` / `self.z_g`
            # itself
            self.z_s, self.z_g = [None], [None]
            self.discri = Discriminator(
                config.hidden_size, max_seq_len).to(self.device)
        elif "VGSAN" in self.method:
            self.model = VGSAN(num_items, args).to(self.device)
        elif "VSAN" in self.method:
            self.model = VSAN(num_items, args).to(self.device)
        elif "SASRec" in self.method:
            self.model = SASRec(num_items, args).to(self.device)
        elif "CL4SRec" in self.method:
            self.model = CL4SRec(num_items, args).to(self.device)
        elif "DuoRec" in self.method:
            self.model = DuoRec(num_items, args).to(self.device)
        elif "ContrastVAE" in self.method:
            self.model = ContrastVAE(num_items, args).to(self.device)

        self.bce_criterion = nn.BCEWithLogitsLoss(
            reduction="none").to(self.device)
        self.cs_criterion = nn.CrossEntropyLoss(
            reduction="none").to(self.device)
        self.cl_criterion = NCELoss(
            temperature=args.temperature).to(self.device)
        self.jsd_criterion = JSDLoss().to(self.device)
        self.hinge_criterion = HingeLoss(margin=0.3).to(self.device)

        if args.method == "FedDCSR":
            self.params = list(self.model.parameters()) + \
                list(self.discri.parameters())
        else:
            self.params = list(self.model.parameters())
        self.optimizer = train_utils.get_optimizer(
            args.optimizer, self.params, args.lr)
        self.step = 0

    def train_batch(self, sessions, adj, num_items, args, global_params=None):
        """Trains the model for one batch.

        Args:
            sessions: Input user sequences.
            adj: Adjacency matrix of the local graph.
            num_items: Number of items in the current domain.
            args: Other arguments for training.
            global_params: Global model parameters used in `FedProx` method.
        """
        self.optimizer.zero_grad()

        if (self.method == "FedDCSR") or ("VGSAN" in self.method):
            # Here the items are first sent to GNN for convolution, and then
            # the resulting embeddings are sent to the self-attention module.
            # Note that each batch must be convolved once, and the
            # item_embeddings input to the convolution layer are updated from
            # the previous batch.
            self.model.graph_convolution(adj)

        sessions = [torch.LongTensor(x).to(self.device) for x in sessions]

        if self.method == "FedDCSR":
            # seq: (batch_size, seq_len), ground: (batch_size, seq_len),
            # ground_mask:  (batch_size, seq_len),
            # js_neg_seqs: (batch_size, seq_len),
            # contrast_aug_seqs: (batch_size, seq_len)
            # Here `js_neg_seqs` is used for computing similarity loss,
            # `contrast_aug_seqs` is used for computing contrastive infomax
            # loss
            seq, ground, ground_mask, js_neg_seqs, contrast_aug_seqs = sessions
            result, result_exclusive, mu_s, logvar_s, self.z_s[0], mu_e, \
                logvar_e, z_e, neg_z_e, aug_z_e = self.model(
                    seq,
                neg_seqs=js_neg_seqs,
                aug_seqs=contrast_aug_seqs)
            # Broadcast in last dim. it well be used to compute `z_g` by
            # federated aggregation later
            self.z_s[0] *= ground_mask.unsqueeze(-1)
            loss = self.disen_vgsan_loss_fn(result, result_exclusive, mu_s,
                                            logvar_s,  mu_e, logvar_e,
                                            ground, self.z_s[0], self.z_g[0],
                                            z_e, neg_z_e, aug_z_e, ground_mask,
                                            num_items, self.step)

        elif "VGSAN" in self.method:
            seq, ground, ground_mask = sessions
            result, mu, logvar = self.model(seq)
            loss = self.vgsan_loss_fn(
                result, mu, logvar, ground, ground_mask, num_items, self.step)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "VSAN" in self.method:
            seq, ground, ground_mask = sessions
            result, mu, logvar = self.model(seq)
            loss = self.vsan_loss_fn(
                result, mu, logvar, ground, ground_mask, num_items, self.step)
            if self.method == "FedVSAN" and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "SASRec" in self.method:
            seq, ground, ground_mask = sessions
            # result： (batch_size, seq_len, hidden_size)
            result = self.model(seq)
            loss = self.sasrec_loss_fn(result, ground, ground_mask, num_items)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "CL4SRec" in self.method:
            seq, ground, ground_mask, aug_seqs1, aug_seqs2 = sessions
            # result： (batch_size, seq_len, hidden_size)
            result, aug_seqs_fea1, aug_seqs_fea2 = self.model(
                seq, aug_seqs1=aug_seqs1, aug_seqs2=aug_seqs2)
            loss = self.cl4srec_loss_fn(
                result, aug_seqs_fea1, aug_seqs_fea2, ground, ground_mask,
                num_items)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "DuoRec" in self.method:
            seq, ground, ground_mask, aug_seqs = sessions
            # result： (batch_size, seq_len, hidden_size)
            result, seqs_fea, aug_seqs_fea = self.model(seq, aug_seqs=aug_seqs)
            loss = self.duorec_loss_fn(
                result, seqs_fea, aug_seqs_fea, ground, ground_mask, num_items)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        elif "ContrastVAE" in self.method:
            seq, ground, ground_mask, aug_seqs = sessions
            # result： (batch_size, seq_len, hidden_size)
            result, aug_result, mu, logvar, z, aug_mu, aug_logvar, aug_z, \
                alpha = self.model(seq, aug_seqs=aug_seqs)
            loss = self.contrastvae_loss_fn(result, aug_result, mu, logvar, z,
                                            aug_mu, aug_logvar, aug_z, alpha,
                                            ground, ground_mask, num_items,
                                            self.step)
            if "Fed" in self.method and args.mu:
                loss += self.prox_reg(
                    [dict(self.model.encoder.named_parameters())],
                    global_params, args.mu)

        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss.item()

    def disen_vgsan_loss_fn(self, result, result_exclusive, mu_s, logvar_s,
                            mu_e, logvar_e, ground, z_s, z_g, z_e, neg_z_e,
                            aug_z_e, ground_mask, num_items, step):
        """Overall loss function of FedDCSR (our method).
        """

        def sim_loss_fn(self, z_s, z_g, neg_z_e, ground_mask):
            pos = self.discri(z_s, z_g, ground_mask)
            neg = self.discri(neg_z_e, z_g, ground_mask)

            # pos_label, neg_label = torch.ones(pos.size()).to(self.device), \
            #     torch.zeros(neg.size()).to(self.device)
            # sim_loss = self.bce_criterion(pos, pos_label) \
            #     + self.bce_criterion(neg, neg_label)

            # sim_loss = self.jsd_criterion(pos, neg)
            sim_loss = self.hinge_criterion(pos, neg)

            sim_loss = sim_loss.mean()

            return sim_loss

        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()

        recons_loss_exclusive = self.cs_criterion(
            result_exclusive.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss_exclusive = (
            recons_loss_exclusive * (ground_mask.reshape(-1))).mean()

        kld_loss_s = -0.5 * \
            torch.sum(1 + logvar_s - mu_s ** 2 -
                      logvar_s.exp(), dim=-1).reshape(-1)
        kld_loss_s = (kld_loss_s * (ground_mask.reshape(-1))).mean()

        kld_loss_e = -0.5 * \
            torch.sum(1 + logvar_e - mu_e ** 2 -
                      logvar_e.exp(), dim=-1).reshape(-1)
        kld_loss_e = (kld_loss_e * (ground_mask.reshape(-1))).mean()

        # If it is the first training round
        if z_g is not None:
            sim_loss = sim_loss_fn(self, z_s, z_g, neg_z_e, ground_mask)
        else:
            sim_loss = 0

        alpha = 1.0  # 1.0 for all scenarios

        kld_weight = self.kl_anneal_function(
            self.args.anneal_cap, step, self.args.total_annealing_step)

        beta = 2.0  # 2.0 for FKCB, 0.5 for BMG and SGH

        gamma = 1.0  # 1.0 for all scenarios

        lam = 1.0  # 1.0 for FKCB and BMG, 0.1 for SGH

        user_representation1 = z_e[:, -1, :]
        user_representation2 = aug_z_e[:, -1, :]
        contrastive_loss = self.cl_criterion(
            user_representation1, user_representation2)
        contrastive_loss = contrastive_loss.mean()

        loss = alpha * (recons_loss + kld_weight * kld_loss_s + kld_weight
                        * kld_loss_e) \
            + beta * sim_loss \
            + gamma * recons_loss_exclusive \
            + lam * contrastive_loss

        return loss

    def vgsan_loss_fn(self, result, mu, logvar, ground, ground_mask, num_items,
                      step):
        """Compute kl divergence, reconstruction.
        result: (batch_size, seq_len, hidden_size),
        mu: (batch_size, seq_len, hidden_size),
        log_var: (batch_size, seq_len, hidden_size),
        ground: (batch_size, seq_len)
        ground_mask: (batch_size, seq_len)
        """
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()

        kld_loss = -0.5 * torch.sum(1 + logvar -
                                    mu ** 2 - logvar.exp(), dim=-1).reshape(-1)
        kld_loss = (kld_loss * (ground_mask.reshape(-1))).mean()

        kld_weight = self.kl_anneal_function(
            self.args.anneal_cap, step, self.args.total_annealing_step)

        loss = recons_loss + kld_weight * kld_loss
        return loss

    def vsan_loss_fn(self, result, mu, logvar, ground, ground_mask, num_items,
                     step):
        """Compute kl divergence, reconstruction.
        result: (batch_size, seq_len, hidden_size),
        mu: (batch_size, seq_len, hidden_size),
        log_var: (batch_size, seq_len, hidden_size),
        ground: (batch_size, seq_len),
        ground_mask: (batch_size, seq_len)
        """
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()

        kld_loss = -0.5 * torch.sum(1 + logvar -
                                    mu ** 2 - logvar.exp(), dim=-1).reshape(-1)
        kld_loss = (kld_loss * (ground_mask.reshape(-1))).mean()

        kld_weight = self.kl_anneal_function(
            self.args.anneal_cap, step, self.args.total_annealing_step)

        loss = recons_loss + kld_weight * kld_loss
        return loss

    def sasrec_loss_fn(self, result, ground, ground_mask, num_items):
        """Compute cross entropy loss for next item prediction.
        result: (batch_size, seq_len, hidden_size),
        ground: (batch_size, seq_len),
        ground_mask: (batch_size, seq_len)
        """
        loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # （batch_size * seq_len, ）
        loss = 1.0 * (loss * (ground_mask.reshape(-1))).mean()
        return loss

    def duorec_loss_fn(self, result, seqs_fea, aug_seqs_fea, ground,
                       ground_mask, num_items):
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()

        user_representation1 = seqs_fea[:, -1, :]
        user_representation2 = aug_seqs_fea[:, -1, :]
        contrastive_loss = self.cl_criterion(
            user_representation1, user_representation2)
        contrastive_loss = contrastive_loss.mean()

        lam = 0.1  # 0.1 is the best
        loss = recons_loss + lam * contrastive_loss
        return loss

    def cl4srec_loss_fn(self, result, aug_seqs_fea1, aug_seqs_fea2, ground,
                        ground_mask, num_items):
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()

        user_representation1 = aug_seqs_fea1[:, -1, :]
        user_representation2 = aug_seqs_fea2[:, -1, :]
        contrastive_loss = self.cl_criterion(
            user_representation1, user_representation2)
        contrastive_loss = contrastive_loss.mean()

        lam = 0.1  # 0.1 is the best
        loss = recons_loss + lam * contrastive_loss
        return loss

    def contrastvae_loss_fn(self, result, aug_result, mu, logvar, z, aug_mu,
                            aug_logvar, aug_z, alpha, ground, ground_mask,
                            num_items, step):
        recons_loss = self.cs_criterion(
            result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss = (recons_loss * (ground_mask.reshape(-1))).mean()
        aug_recons_loss = self.cs_criterion(
            aug_result.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        aug_recons_loss = (aug_recons_loss * (ground_mask.reshape(-1))).mean()

        kld_loss = -0.5 * torch.sum(1 + logvar -
                                    mu ** 2 - logvar.exp(), dim=-1).reshape(-1)
        kld_loss = (kld_loss * (ground_mask.reshape(-1))).mean()
        aug_kld_loss = -0.5 * \
            torch.sum(1 + aug_logvar - aug_mu ** 2 -
                      aug_logvar.exp(), dim=-1).reshape(-1)
        aug_kld_loss = (aug_kld_loss * (ground_mask.reshape(-1))).mean()

        kld_weight = self.kl_anneal_function(
            self.args.anneal_cap, step, self.args.total_annealing_step)

        lam = 1.0  # 1.0 is the best

        mask = ground_mask.float().sum(-1).unsqueeze(-1)\
            .repeat(1, ground_mask.size(-1))
        mask = 1 / mask
        mask = ground_mask * mask  # For mean
        mask = ground_mask.unsqueeze(-1).repeat(1, 1, z.size(-1))
        # user representation1: (batch_size, hidden_size)
        # user_representation2: (batch_size, hidden_size)
        user_representation1 = (z * mask).sum(1)
        user_representation2 = (aug_z * mask).sum(1)

        contrastive_loss = self.cl_criterion(
            user_representation1, user_representation2)
        contrastive_loss = contrastive_loss.mean()

        loss = recons_loss + aug_recons_loss + kld_weight * (kld_loss
                                                             + aug_kld_loss) \
            + lam * contrastive_loss
        # Compute priorKL loss
        if alpha:
            adaptive_alpha_loss = priorKL(alpha).mean()
            loss += adaptive_alpha_loss
        return loss

    def kl_anneal_function(self, anneal_cap, step, total_annealing_step):
        """
        step: increment by 1 for every forward-backward step.
        total annealing steps: pre-fixed parameter control the speed of
        anealing.
        """
        # borrows from https://github.com/timbmg/Sentence-VAE/blob/master/train.py
        return min(anneal_cap, step / total_annealing_step)

    @ staticmethod
    def flatten(source):
        return torch.cat([value.flatten() for value in source])

    def prox_reg(self, params1, params2, mu):
        params1_values, params2_values = [], []
        # Record the model parameter aggregation results of each branch
        # separately
        for branch_params1, branch_params2 in zip(params1, params2):
            branch_params2 = [branch_params2[key]
                              for key in branch_params1.keys()]
            params1_values.extend(branch_params1.values())
            params2_values.extend(branch_params2)

        # Multidimensional parameters should be compressed into one dimension
        # using the flatten function
        s1 = self.flatten(params1_values)
        s2 = self.flatten(params2_values)
        return mu/2 * torch.norm(s1 - s2)

    def test_batch(self, sessions):
        """Tests the model for one batch.

        Args:
            sessions: Input user sequences.
        """
        sessions = [torch.LongTensor(x).to(self.device) for x in sessions]

        # seq: (batch_size, seq_len), ground_truth: (batch_size, ),
        # neg_list: (batch_size, num_test_neg)
        seq, ground_truth, neg_list = sessions
        # result: (batch_size, seq_len, num_items)
        result = self.model(seq)

        pred = []
        for id in range(len(result)):
            # result[id, -1]: (num_items, )
            score = result[id, -1]
            cur = score[ground_truth[id]]
            # score_larger = (score[neg_list[id]] > (cur + 0.00001))\
            # .data.cpu().numpy()
            score_larger = (score[neg_list[id]] > (cur)).data.cpu().numpy()
            true_item_rank = np.sum(score_larger) + 1
            pred.append(true_item_rank)

        return pred
