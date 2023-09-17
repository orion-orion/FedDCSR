# -*- coding: utf-8 -*-
"""Customized dataset.
"""
import math
import random
import os
import copy
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    """A customized dataset reading and preprocessing data of a certain domain
    from ".txt" files.
    """
    data_dir = "data"
    prep_dir = "prep_data"
    # The number of negative samples to test all methods (including ours)
    num_test_neg = 999

    def __init__(self, domain, model="SAN", mode="train", max_seq_len=16,
                 load_prep=True):
        assert (model in ["DisenVGSAN", "VGSAN", "SASRec",
                "VSAN", "ContrastVAE", "CL4SRec", "DuoRec"])
        assert (mode in ["train", "valid", "test"])

        self.domain = domain
        self.model = model
        self.mode = mode
        self.dataset_dir = os.path.join(self.data_dir, self.domain)
        self.user_ids, self.sessions, self.num_items \
            = self.read_data(self.dataset_dir)
        self.max_seq_len = max_seq_len
        self.prep_sessions = self.preprocess(
            self.sessions, self.dataset_dir, load_prep)

    def read_data(self, dataset_dir):
        with open(os.path.join(dataset_dir, "num_items.txt"),
                  "rt", encoding="utf-8") as infile:
            num_items = int(infile.readline())

        with open(os.path.join(self.data_dir, self.domain,
                               "%s_data.txt" % self.mode), "rt",
                  encoding="utf-8") as infile:
            user_ids, sessions = [], []
            for line in infile.readlines():
                session = []
                line = line.strip().split("\t")
                # Note that the ground truth is included when computing the
                # sequence lengths of domain A and domain B
                for item in line[1:]:  # Start from index 1 to exclude user ID
                    item = int(item)
                    session.append(item)
                user_ids.append(int(line[0]))
                sessions.append(session)
        print("Successfully load %s %s data!" % (self.domain, self.mode))

        return user_ids, sessions, num_items

    def preprocess(self, sessions, dataset_dir, load_prep):
        prep_functions = {"DisenVGSAN": self.preprocess_disen_vgsan,
                          "VGSAN": self.preprocess_vgsan,
                          "SASRec": self.preprocess_sasrec,
                          "VSAN": self.preprocess_vgsan,
                          "ContrastVAE": self.preprocess_contrastvae,
                          "CL4SRec": self.preprocess_cl4srec,
                          "DuoRec": self.preprocess_duorec}
        if not os.path.exists(os.path.join(dataset_dir, self.prep_dir)):
            os.makedirs(os.path.join(dataset_dir, self.prep_dir))

        self.prep_data_path = os.path.join(
            dataset_dir, self.prep_dir, "%s_%s_data.pkl" % (self.model,
                                                            self.mode))
        if os.path.exists(self.prep_data_path) and load_prep:
            with open(os.path.join(self.prep_data_path), "rb") as infile:
                prep_sessions = pickle.load(infile)
            print("Successfully load preprocessed %s %s data!" %
                  (self.domain, self.mode))
        else:
            prep_sessions = prep_functions[self.model](
                sessions, mode=self.mode)
            with open(self.prep_data_path, "wb") as infile:
                pickle.dump(prep_sessions, infile)
            print("Successfully preprocess %s %s data!" %
                  (self.domain, self.mode))
        return prep_sessions

    @ staticmethod
    def random_neg(left, right, excl):  # [left, right)
        sample = np.random.randint(left, right)
        while sample in excl:
            sample = np.random.randint(left, right)
        return sample

    def preprocess_disen_vgsan(self, data, mode="train"):
        prep_sessions = []
        for session in data:  # The pad is needed
            temp = []
            if mode == "train":
                items_input = session[:-1]
                ground_truths = session[1:]
                # Here `js_neg_seqs` is used for computing similarity loss,
                # `contrast_aug_seqs` is used for computing contrastive infomax loss
                js_neg_seq = copy.deepcopy(items_input)
                contrast_aug_seq = copy.deepcopy(items_input)
                random.shuffle(contrast_aug_seq)
            else:
                items_input = session[:-1]
                ground_truth = session[-1]

            pad_len = self.max_seq_len - len(items_input)
            items_input = [self.num_items] * pad_len + items_input
            temp.append(items_input)
            if mode == "train":
                pad_len1 = self.max_seq_len - len(js_neg_seq)
                pad_len2 = self.max_seq_len - len(contrast_aug_seq)
                ground_mask = [0] * pad_len + [1] * len(ground_truths)
                ground_truths = [self.num_items] * pad_len + ground_truths
                js_neg_seq = [self.num_items] * pad_len1 + js_neg_seq
                contrast_aug_seq = [self.num_items] * \
                    pad_len2 + contrast_aug_seq
                temp.append(ground_truths)
                temp.append(ground_mask)
                temp.append(js_neg_seq)
                temp.append(contrast_aug_seq)
            else:
                temp.append(ground_truth)
                neg_samples = []
                for _ in range(self.num_test_neg):
                    # Negative samples must be generated in the corresponding
                    # domain
                    neg_sample = self.random_neg(
                        0, self.num_items, excl=[ground_truth])
                    neg_samples.append(neg_sample)
                temp.append(neg_samples)

            prep_sessions.append(temp)
        return prep_sessions

    def preprocess_vgsan(self, data, mode="train"):
        prep_sessions = []
        for session in data:  # The pad is needed
            temp = []
            if mode == "train":
                items_input = session[:-1]
                ground_truths = session[1:]
            else:
                items_input = session[:-1]
                ground_truth = session[-1]

            pad_len = self.max_seq_len - len(items_input)
            items_input = [self.num_items] * pad_len + items_input
            temp.append(items_input)
            if mode == "train":
                ground_mask = [0] * pad_len + [1] * len(ground_truths)
                ground_truths = [self.num_items] * pad_len + ground_truths
                temp.append(ground_truths)
                temp.append(ground_mask)
            else:
                temp.append(ground_truth)
                neg_samples = []
                for _ in range(self.num_test_neg):
                    # Negative samples must be generated in the corresponding
                    # domain
                    neg_sample = self.random_neg(
                        0, self.num_items, excl=[ground_truth])
                    neg_samples.append(neg_sample)
                temp.append(neg_samples)

            prep_sessions.append(temp)
        return prep_sessions

    def preprocess_sasrec(self, data, mode="train"):
        prep_sessions = []
        for session in data:  # The pad is needed
            temp = []
            if mode == "train":
                items_input = session[:-1]
                ground_truths = session[1:]
            else:
                items_input = session[:-1]
                ground_truth = session[-1]

            pad_len = self.max_seq_len - len(items_input)
            items_input = [self.num_items] * pad_len + items_input
            temp.append(items_input)

            if mode == "train":
                ground_mask = [0] * pad_len + [1] * len(ground_truths)
                ground_truths = [self.num_items] * pad_len + ground_truths
                temp.append(ground_truths)
                temp.append(ground_mask)
            else:
                temp.append(ground_truth)
                neg_samples = []
                for _ in range(self.num_test_neg):
                    # Negative samples must be generated in the corresponding
                    # domain
                    neg_sample = self.random_neg(
                        0, self.num_items, excl=[ground_truth])
                    neg_samples.append(neg_sample)
                temp.append(neg_samples)

            prep_sessions.append(temp)
        return prep_sessions

    def preprocess_contrastvae(self, data, mode="train"):
        prep_sessions = []
        for session in data:  # The pad is needed
            temp = []
            if mode == "train":
                items_input = session[:-1]
                ground_truths = session[1:]
                switch = random.sample(range(3), k=1)
                if switch[0] == 0:
                    aug_seq = self.item_crop(items_input)
                elif switch[0] == 1:
                    aug_seq = self.item_mask(items_input)
                elif switch[0] == 2:
                    aug_seq = self.item_reorder(items_input)
            else:
                items_input = session[:-1]
                ground_truth = session[-1]

            pad_len = self.max_seq_len - len(items_input)
            items_input = [self.num_items] * pad_len + items_input
            temp.append(items_input)
            if mode == "train":
                aug_pad_len = self.max_seq_len - len(aug_seq)
                ground_mask = [0] * pad_len + [1] * len(ground_truths)
                ground_truths = [self.num_items] * pad_len + ground_truths
                aug_seq = [self.num_items] * aug_pad_len + aug_seq
                temp.append(ground_truths)
                temp.append(ground_mask)
                temp.append(aug_seq)
            else:
                temp.append(ground_truth)
                neg_samples = []
                for _ in range(self.num_test_neg):
                    # Negative samples must be generated in the corresponding
                    # domain
                    neg_sample = self.random_neg(
                        0, self.num_items, excl=[ground_truth])
                    neg_samples.append(neg_sample)
                temp.append(neg_samples)

            prep_sessions.append(temp)
        return prep_sessions

    def preprocess_cl4srec(self, data, mode="train"):
        prep_sessions = []
        for session in data:  # The pad is needed
            temp = []
            if mode == "train":
                items_input = session[:-1]
                ground_truths = session[1:]
                switch = random.sample(range(3), k=2)
                if switch[0] == 0:
                    aug_seq_1 = self.item_crop(items_input)
                elif switch[0] == 1:
                    aug_seq_1 = self.item_mask(items_input)
                elif switch[0] == 2:
                    aug_seq_1 = self.item_reorder(items_input)

                if switch[1] == 0:
                    aug_seq_2 = self.item_crop(items_input)
                elif switch[1] == 1:
                    aug_seq_2 = self.item_mask(items_input)
                elif switch[1] == 2:
                    aug_seq_2 = self.item_reorder(items_input)
            else:
                items_input = session[:-1]
                ground_truth = session[-1]

            pad_len = self.max_seq_len - len(items_input)
            items_input = [self.num_items] * pad_len + items_input
            temp.append(items_input)
            if mode == "train":
                aug_pad_len1 = self.max_seq_len - len(aug_seq_1)
                aug_pad_len2 = self.max_seq_len - len(aug_seq_2)
                ground_mask = [0] * pad_len + [1] * len(ground_truths)
                ground_truths = [self.num_items] * pad_len + ground_truths
                aug_seq_1 = [self.num_items] * aug_pad_len1 + aug_seq_1
                aug_seq_2 = [self.num_items] * aug_pad_len2 + aug_seq_2
                temp.append(ground_truths)
                temp.append(ground_mask)
                temp.append(aug_seq_1)
                temp.append(aug_seq_2)
            else:
                temp.append(ground_truth)
                neg_samples = []
                for _ in range(self.num_test_neg):
                    # Negative samples must be generated in the corresponding
                    # domain
                    neg_sample = self.random_neg(
                        0, self.num_items, excl=[ground_truth])
                    neg_samples.append(neg_sample)
                temp.append(neg_samples)

            prep_sessions.append(temp)
        return prep_sessions

    def preprocess_duorec(self, data, mode="train"):
        if mode == "train":
            last_item_to_seq = defaultdict(list)
            for idx, session in enumerate(data):
                items_input = session[:-1]
                last_item = session[-1]
                last_item_to_seq[last_item].append(items_input)

        prep_sessions = []
        for session in data:  # The pad is needed
            temp = []
            if mode == "train":
                items_input = session[:-1]
                ground_truths = session[1:]
                seqs_same_last_item = last_item_to_seq[session[-1]]
                if len(seqs_same_last_item) == 1:
                    aug_seq = copy.deepcopy(items_input)
                else:
                    idx = np.random.randint(0, len(seqs_same_last_item))
                    while seqs_same_last_item[idx] == items_input:
                        idx = np.random.randint(0, len(seqs_same_last_item))
                    aug_seq = copy.deepcopy(seqs_same_last_item[idx])
            else:
                items_input = session[:-1]
                ground_truth = session[-1]

            pad_len = self.max_seq_len - len(items_input)
            items_input = [self.num_items] * pad_len + items_input
            temp.append(items_input)
            if mode == "train":
                aug_pad_len = self.max_seq_len - len(aug_seq)
                ground_mask = [0] * pad_len + [1] * len(ground_truths)
                ground_truths = [self.num_items] * pad_len + ground_truths
                aug_seq = [self.num_items] * aug_pad_len + aug_seq
                temp.append(ground_truths)
                temp.append(ground_mask)
                temp.append(aug_seq)
            else:
                temp.append(ground_truth)
                neg_samples = []
                for _ in range(self.num_test_neg):
                    # Negative samples must be generated in the corresponding
                    # domain
                    neg_sample = self.random_neg(
                        0, self.num_items, excl=[ground_truth])
                    neg_samples.append(neg_sample)
                temp.append(neg_samples)

            prep_sessions.append(temp)
        return prep_sessions

    def item_crop(self, item_seq, eta=0.6):
        item_seq_len = len(item_seq)
        num_left = math.floor(item_seq_len * eta)
        crop_begin = random.randint(0, item_seq_len - num_left)
        if crop_begin + num_left < len(item_seq):
            croped_item_seq = copy.deepcopy(
                item_seq[crop_begin: crop_begin + num_left])
        else:
            croped_item_seq = copy.deepcopy(item_seq[crop_begin:])
        return croped_item_seq

    def item_mask(self, item_seq, gamma=0.3):
        item_seq_len = len(item_seq)
        num_mask = math.floor(item_seq_len * gamma)
        mask_index = random.sample(range(item_seq_len), k=num_mask)
        masked_item_seq = copy.deepcopy(item_seq)
        masked_item_seq = np.array(masked_item_seq)
        # Token [num_items] has been used for semantic masking
        masked_item_seq[mask_index] = self.num_items
        return masked_item_seq.tolist()

    def item_reorder(self, item_seq, beta=0.6):
        item_seq_len = len(item_seq)
        num_reorder = math.floor(item_seq_len * beta)
        reorder_begin = random.randint(0, item_seq_len - num_reorder)
        reordered_item_seq = copy.deepcopy(item_seq)
        reordered_item_seq = np.array(reordered_item_seq)
        shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
        random.shuffle(shuffle_index)
        reordered_item_seq[reorder_begin:reorder_begin +
                           num_reorder] = reordered_item_seq[shuffle_index]
        return reordered_item_seq.tolist()

    def __len__(self):
        return len(self.prep_sessions)

    def __getitem__(self, idx):
        user_ids = self.user_ids[idx]
        session = self.prep_sessions[idx]
        return user_ids, session

    def __setitem__(self, idx, value):
        """To support shuffle operation.
        """
        self.user_ids[idx] = value[0]
        self.prep_sessions[idx] = value[1]

    def __add__(self, other):
        """To support concatenation operation.
        """
        user_ids, prep_sessions = other
        self.user_ids += user_ids
        self.prep_sessions += prep_sessions
        return self
