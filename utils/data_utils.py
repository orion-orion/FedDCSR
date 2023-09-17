# -*- coding: utf-8 -*-
import numpy as np
import torch
from dataset import SeqDataset
from local_graph import LocalGraph


def load_dataset(args):
    client_train_datasets = []
    client_valid_datasets = []
    client_test_datasets = []
    for domain in args.domains:
        if args.method == "FedDCSR":
            model = "DisenVGSAN"
        else:
            model = args.method.replace("Fed", "")
            model = model.replace("Local", "")

        train_dataset = SeqDataset(
            domain, model, mode="train", max_seq_len=args.max_seq_len,
            load_prep=args.load_prep)
        valid_dataset = SeqDataset(
            domain, model, mode="valid", max_seq_len=args.max_seq_len,
            load_prep=args.load_prep)
        test_dataset = SeqDataset(
            domain, model, mode="test", max_seq_len=args.max_seq_len,
            load_prep=args.load_prep)

        client_train_datasets.append(train_dataset)
        client_valid_datasets.append(valid_dataset)
        client_test_datasets.append(test_dataset)

    adjs = []
    for train_dataset, domain in zip(client_train_datasets, args.domains):
        local_graph = LocalGraph(args, domain, train_dataset.num_items)
        adjs.append(local_graph.adj)
        print("%s graph loaded!" % domain)

    if args.cuda:
        torch.cuda.empty_cache()
        device = "cuda:%s" % args.gpu
    else:
        device = "cpu"
    for idx, adj in enumerate(adjs):
        adjs[idx] = adj.to(device)

    return client_train_datasets, client_valid_datasets, \
        client_test_datasets, adjs


def init_clients_weight(clients):
    """Initialize the aggretation weight, which is the ratio of the number of
    samples per client to the total number of samples.
    """
    client_n_samples_train = [client.n_samples_train for client in clients]

    samples_sum_train = np.sum(client_n_samples_train)
    for client in clients:
        client.train_weight = client.n_samples_train / samples_sum_train
        client.valid_weight = 1 / len(clients)
        client.test_weight = 1 / len(clients)
