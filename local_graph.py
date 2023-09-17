# -*- coding: utf-8 -*-
"""Local Graph class.
"""
import numpy as np
import scipy.sparse as sp
import torch
import os


def normalize(mx):
    """Row-normalize sparse matrix.
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class LocalGraph(object):
    """A local graph data structure class reading training data of a certain
    domain from ".txt" files, and preprocess it into a local graph.
    """
    data_dir = "data"

    def __init__(self, args, domain, num_items):
        self.args = args
        self.dataset_dir = os.path.join(self.data_dir, domain)
        self.raw_data = self.read_train_data(self.dataset_dir)
        self.num_items = num_items
        self.adj = self.preprocess(self.raw_data)

    def read_train_data(self, dataset_dir):
        with open(os.path.join(dataset_dir, "train_data.txt"),
                  "rt", encoding="utf-8") as infile:
            train_data = []
            for line in infile.readlines():
                session = []
                line = line.strip().split("\t")
                for item in line[1:]:  # Start from index 1 to exclude user ID
                    item = int(item)
                    session.append(item)
                train_data.append(session)
        return train_data

    def preprocess(self, data):
        VV_edges = []
        for session in data:
            source = -1
            for item in session:
                if source != -1:
                    VV_edges.append([source, item])
                source = item

        VV_edges = np.array(VV_edges)
        adj = sp.coo_matrix((np.ones(VV_edges.shape[0]), (VV_edges[:, 0],
                                                          VV_edges[:, 1])),
                            shape=(self.num_items + 1, self.num_items + 1),
                            dtype=np.float32)

        adj = normalize(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        return adj
