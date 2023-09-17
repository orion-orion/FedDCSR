# -*- coding: utf-8 -*-
import logging
from tqdm import tqdm
from utils.train_utils import EarlyStopping, LRDecay


def evaluation_logging(eval_logs, round, weights, mode="valid"):
    if mode == "valid":
        logging.info("Epoch%d Valid:" % round)
    else:
        logging.info("Test:")

    avg_eval_log = {}
    for metric_name in list(eval_logs.values())[0].keys():
        avg_eval_val = 0
        for domain in eval_logs.keys():
            avg_eval_val += eval_logs[domain][metric_name] * weights[domain]
        avg_eval_log[metric_name] = avg_eval_val

    logging.info("MRR: %.4f" % avg_eval_log["MRR"])
    logging.info("HR @1|5|10: %.4f \t %.4f \t %.4f \t" %
                 (avg_eval_log["HR @1"], avg_eval_log["HR @5"],
                     avg_eval_log["HR @10"]))
    logging.info("NDCG @5|10: %.4f \t %.4f" %
                 (avg_eval_log["NDCG @5"], avg_eval_log["NDCG @10"]))

    for domain, eval_log in eval_logs.items():
        logging.info("%s MRR: %.4f \t HR @10: %.4f \t NDCG @10: %.4f"
                     % (domain, eval_log["MRR"], eval_log["HR @10"],
                         eval_log["NDCG @10"]))

    return avg_eval_log


def load_and_eval_model(n_clients, clients, args):
    eval_logs = {}
    for c_id in tqdm(range(n_clients), ascii=True):
        clients[c_id].load_params()
        eval_log = clients[c_id].evaluation(mode="test")
        eval_logs[clients[c_id].domain] = eval_log
    weights = dict((client.domain, client.test_weight) for client in clients)
    evaluation_logging(eval_logs, 0, weights, mode="test")


def run_fl(clients, server, args):
    n_clients = len(clients)
    if args.do_eval:
        load_and_eval_model(n_clients, clients, args)
    else:
        early_stopping = EarlyStopping(
            args.checkpoint_dir, patience=args.es_patience, verbose=True)
        lr_decay = LRDecay(args.lr, args.decay_epoch,
                           args.optimizer, args.lr_decay,
                           patience=args.ld_patience, verbose=True)
        for round in range(1, args.epochs + 1):
            random_cids = server.choose_clients(n_clients, args.frac)

            # Train with these clients
            for c_id in tqdm(random_cids, ascii=True):
                if "Fed" in args.method:
                    # Restore global parameters to client's model
                    clients[c_id].set_global_params(server.get_global_params())
                    if args.method == "FedDCSR":
                        clients[c_id].set_global_reps(server.get_global_reps())

                # Train one client
                clients[c_id].train_epoch(
                    round, args, global_params=server.global_params)

            if "Fed" in args.method:
                server.aggregate_params(clients, random_cids)
                if args.method == "FedDCSR":
                    server.aggregate_reps(clients, random_cids)

            if round % args.eval_interval == 0:
                eval_logs = {}
                for c_id in tqdm(range(n_clients), ascii=True):
                    if "Fed" in args.method:
                        clients[c_id].set_global_params(
                            server.get_global_params())
                    if c_id in random_cids:
                        eval_log = clients[c_id].evaluation(mode="valid")
                    else:
                        eval_log = clients[c_id].get_old_eval_log()
                    eval_logs[clients[c_id].domain] = eval_log

                weights = dict((client.domain, client.valid_weight)
                               for client in clients)
                avg_eval_log = evaluation_logging(
                    eval_logs, round, weights, mode="valid")

                # Early Stopping. Here only compare the current results with
                # the best results
                early_stopping(avg_eval_log, clients)
                if early_stopping.early_stop:
                    logging.info("Early stopping")
                    break

                # Learning rate decay. Here only compare the current results
                # with the latest results
                lr_decay(round, avg_eval_log, clients)

        load_and_eval_model(n_clients, clients, args)
