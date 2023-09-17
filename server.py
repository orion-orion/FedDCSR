# -*- coding: utf-8 -*-
import math
import numpy as np


class Server(object):
    def __init__(self, args, init_global_params):
        self.args = args
        self.global_params = init_global_params
        if args.method == "FedDCSR":
            self.global_reps = None

    def aggregate_params(self, clients, random_cids):
        """Sums up parameters of models shared by all active clients at each
        epoch.

        Args:
            clients: A list of clients instances.
            random_cids: Randomly selected client ID in each training round.
        """
        # Record the model parameter aggregation results of each branch
        # separately
        num_branchs = len(self.global_params)
        for branch_idx in range(num_branchs):
            client_params_sum = None
            for c_id in random_cids:
                # Obtain current client's parameters
                current_client_params = clients[c_id].get_params()[branch_idx]
                # Sum it up with weights
                if client_params_sum is None:
                    client_params_sum = dict((key, value
                                              * clients[c_id].train_weight)
                                             for key, value
                                             in current_client_params.items())
                else:
                    for key in client_params_sum.keys():
                        client_params_sum[key] += clients[c_id].train_weight \
                            * current_client_params[key]
            self.global_params[branch_idx] = client_params_sum

    def aggregate_reps(self, clients, random_cids):
        """Sums up representations of user sequences shared by all active
        clients at each epoch.

        Args:
            clients: A list of clients instances.
            random_cids: Randomly selected client ID in each training round.
        """
        # Record the user sequence aggregation results of each branch
        # separately
        client_reps_sum = None
        for c_id in random_cids:
            # Obtain current client's user sequence representations
            current_client_reps = clients[c_id].get_reps_shared()
            # Sum it up with weights
            if client_reps_sum is None:
                client_reps_sum = current_client_reps * \
                    clients[c_id].train_weight
            else:
                client_reps_sum += clients[c_id].train_weight * \
                    current_client_reps
        self.global_reps = client_reps_sum

    def choose_clients(self, n_clients, ratio=1.0):
        """Randomly chooses some clients.
        """
        choose_num = math.ceil(n_clients * ratio)
        return np.random.permutation(n_clients)[:choose_num]

    def get_global_params(self):
        """Returns a reference to the parameters of the global model.
        """
        return self.global_params

    def get_global_reps(self):
        """Returns a reference to the parameters of the global model.
        """
        return self.global_reps
