# -*- coding: utf-8 -*-
import os
import gc
import copy
import logging
import numpy as np
import torch
from dataloader import SeqDataloader
from utils.io_utils import ensure_dir


class Client:
    def __init__(self, model_fn, c_id, args, adj, train_dataset, valid_dataset, test_dataset):
        # Used for computing the mask in self-attention module
        self.num_items = train_dataset.num_items
        self.domain = train_dataset.domain
        # Used for computing the positional embeddings
        self.max_seq_len = args.max_seq_len
        self.trainer = model_fn(args, self.num_items, self.max_seq_len)
        self.model = self.trainer.model
        self.method = args.method
        self.checkpoint_dir = args.checkpoint_dir
        self.model_id = args.id if len(args.id) > 1 else "0" + args.id
        if args.method == "FedDCSR":
            self.z_s = self.trainer.z_s
            self.z_g = self.trainer.z_g
        self.c_id = c_id
        self.args = args
        self.adj = adj

        self.train_dataloader = SeqDataloader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_dataloader = SeqDataloader(
            valid_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = SeqDataloader(
            test_dataset, batch_size=args.batch_size, shuffle=False)

        # Compute the number of samples for each client
        self.n_samples_train = len(train_dataset)
        self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)
        # The aggretation weight
        self.train_pop, self.valid_weight, self.test_weight = 0.0, 0.0, 0.0
        # Model evaluation results
        self.MRR, self.NDCG_5, self.NDCG_10, self.HR_1, self.HR_5, self.HR_10 \
            = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def train_epoch(self, round, args, global_params=None):
        """Trains one client with its own training data for one epoch.

        Args:
            round: Training round.
            args: Other arguments for training.
            global_params: Global model parameters used in `FedProx` method.
        """
        self.trainer.model.train()
        for _ in range(args.local_epoch):
            loss = 0
            step = 0
            for _, sessions in self.train_dataloader:
                if ("Fed" in args.method) and args.mu:
                    batch_loss = self.trainer.train_batch(
                        sessions, self.adj, self.num_items, args,
                        global_params=global_params)
                else:
                    batch_loss = self.trainer.train_batch(
                        sessions, self.adj, self.num_items, args)
                loss += batch_loss
                step += 1

            gc.collect()
        logging.info("Epoch {}/{} - client {} -  Training Loss: {:.3f}".format(
            round, args.epochs, self.c_id, loss / step))
        return self.n_samples_train

    def evaluation(self, mode="valid"):
        """Evaluates one client with its own valid/test data for one epoch.

        Args:
            mode: `valid` or `test`.
        """
        if mode == "valid":
            dataloader = self.valid_dataloader
        elif mode == "test":
            dataloader = self.test_dataloader

        self.trainer.model.eval()
        if (self.method == "FedDCSR") or ("VGSAN" in self.method):
            self.trainer.model.graph_convolution(self.adj)
        pred = []
        for _, sessions in dataloader:
            predictions = self.trainer.test_batch(sessions)
            pred = pred + predictions

        gc.collect()
        self.MRR, self.NDCG_5, self.NDCG_10, self.HR_1, self.HR_5, self.HR_10 \
            = self.cal_test_score(pred)
        return {"MRR": self.MRR, "HR @1": self.HR_1, "HR @5": self.HR_5,
                "HR @10":  self.HR_10, "NDCG @5":  self.NDCG_5,
                "NDCG @10": self.NDCG_10}

    def get_old_eval_log(self):
        """Returns the evaluation result of the lastest epoch.
        """
        return {"MRR": self.MRR, "HR @1": self.HR_1, "HR @5": self.HR_5,
                "HR @10":  self.HR_10, "NDCG @5":  self.NDCG_5,
                "NDCG @10": self.NDCG_10}

    @ staticmethod
    def cal_test_score(predictions):
        MRR = 0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        valid_entity = 0.0
        # `pred` indicates the rank of groundtruth items in the recommendation
        # list
        for pred in predictions:
            valid_entity += 1
            MRR += 1 / pred
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
                HR_5 += 1
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
                HR_10 += 1
        return MRR/valid_entity, NDCG_5 / valid_entity, \
            NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / \
            valid_entity, HR_10 / valid_entity

    def get_params(self):
        """Returns the model parameters that need to be shared between clients.
        """
        if self.method == "FedDCSR":
            return copy.deepcopy([self.model.encoder_s.state_dict()])
        elif "VGSAN" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif "SASRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif "VSAN" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict(),
                                  self.model.decoder.state_dict()])
        elif "ContrastVAE" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict(),
                                  self.model.decoder.state_dict()])
        elif "CL4SRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])
        elif "DuoRec" in self.method:
            return copy.deepcopy([self.model.encoder.state_dict()])

    def get_reps_shared(self):
        """Returns the user sequence representations that need to be shared
        between clients.
        """
        assert (self.method == "FedDCSR")
        return copy.deepcopy(self.z_s[0].detach())

    def set_global_params(self, global_params):
        """Assign the local shared model parameters with global model
        parameters.
        """
        assert (self.method in ["FedDCSR", "FedVGSAN", "FedSASRec", "FedVSAN",
                                "FedContrastVAE", "FedCL4SRec", "FedDuoRec"])
        if self.method == "FedDCSR":
            self.model.encoder_s.load_state_dict(global_params[0])
        elif self.method == "FedVGSAN":
            self.model.encoder.load_state_dict(global_params[0])
            # self.model.decoder.load_state_dict(global_params[1])
        elif self.method == "FedSASRec":
            self.model.encoder.load_state_dict(global_params[0])
        elif self.method == "FedVSAN":
            self.model.encoder.load_state_dict(global_params[0])
            self.model.decoder.load_state_dict(global_params[1])
        elif self.method == "FedContrastVAE":
            self.model.encoder.load_state_dict(global_params[0])
            self.model.decoder.load_state_dict(global_params[1])
        elif self.method == "FedCL4SRec":
            self.model.encoder.load_state_dict(global_params[0])
        elif self.method == "FedDuoRec":
            self.model.encoder.load_state_dict(global_params[0])

    def set_global_reps(self, global_rep):
        """Copy global user sequence representations to local.
        """
        assert (self.method == "FedDCSR")
        self.z_g[0] = copy.deepcopy(global_rep)

    def save_params(self):
        method_ckpt_path = os.path.join(self.checkpoint_dir,
                                        "domain_" +
                                        "".join([domain[0]
                                                for domain
                                                 in self.args.domains]),
                                        self.method + "_" + self.model_id)
        ensure_dir(method_ckpt_path, verbose=True)
        ckpt_filename = os.path.join(
            method_ckpt_path, "client%d.pt" % self.c_id)
        params = self.trainer.model.state_dict()
        try:
            torch.save(params, ckpt_filename)
            print("Model saved to {}".format(ckpt_filename))
        except IOError:
            print("[ Warning: Saving failed... continuing anyway. ]")

    def load_params(self):
        ckpt_filename = os.path.join(self.checkpoint_dir,
                                     "domain_" +
                                     "".join([domain[0]
                                             for domain in self.args.domains]),
                                     self.method + "_" + self.model_id,
                                     "client%d.pt" % self.c_id)
        try:
            checkpoint = torch.load(ckpt_filename)
        except IOError:
            print("[ Fail: Cannot load model from {}. ]".format(ckpt_filename))
            exit(1)
        if self.trainer.model is not None:
            self.trainer.model.load_state_dict(checkpoint)
