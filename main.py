# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import argparse
import torch
from trainer import ModelTrainer
import logging
from client import Client
from server import Server
from utils.data_utils import load_dataset, init_clients_weight
from utils.io_utils import save_config, ensure_dir
from fl import run_fl


def arg_parse():
    parser = argparse.ArgumentParser()

    # Dataset part
    parser.add_argument(dest="domains", metavar="domains", nargs="*",
                        help="`Food Kitchen Clothing Beauty` or "
                        "`Movies Books Games` or `Sports Garden Home`")
    parser.add_argument("--load_prep", dest="load_prep", action="store_true",
                        default=False,
                        help="Whether need to load preprocessed the data. If "
                        "you want to load preprocessed data, add it")
    parser.add_argument("--max_seq_len", type=int,
                        default=16, help="maxisum sequence length")

    # Training part
    parser.add_argument("--method", type=str, default="FedDCSR",
                        help="method, possible are `FedDCSR`(ours), "
                        "`FedVGSAN`, `LocalVGSAN`, `FedSASRec`, "
                        "`LocalSASRec`, `FedVSAN`, `LocalVSAN`, "
                        "`FedContrastVAE`, `LocalContrastVAE`, `FedCL4SRec`, "
                        "`LocalCL4SRec`, `FedDuoRec`, `LocalDuoRec`")
    parser.add_argument("--log_dir", type=str,
                        default="log", help="directory of logs")
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of total training iterations.")
    parser.add_argument("--local_epoch", type=int, default=3,
                        help="Number of local training epochs.")
    parser.add_argument("--optimizer", choices=["sgd", "adagrad", "adam",
                                                "adamax"], default="adam",
                        help="Optimizer: sgd, adagrad, adam or adamax.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Applies to sgd and adagrad.")  # 0.001
    parser.add_argument("--lr_decay", type=float, default=1,
                        help="Learning rate decay rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--decay_epoch", type=int, default=10,
                        help="Decay learning rate after this epoch.")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int,
                        default=1, help="Interval of evalution")
    parser.add_argument("--frac", type=float, default=1,
                        help="Fraction of participating clients")
    parser.add_argument("--mu", type=float, default=0,
                        help="hyper parameter for FedProx")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoint", help="Checkpoint Dir")
    parser.add_argument("--id", type=str, default="00",
                        help="Model ID under which to save models.")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--es_patience", type=int,
                        default=5, help="Early stop patience.")
    parser.add_argument("--ld_patience", type=int, default=1,
                        help="Learning rate decay patience.")

    # KL annealing arguments for variantional method (including ours)
    parser.add_argument("--anneal_cap", type=float, default=1.0, help="KL "
                        "annealing arguments for variantional method "
                        "(including ours). 1.0 for FKCB is the best, 0.01 for "
                        "MBG and SGH is the best")
    parser.add_argument("--total_annealing_step", type=int, default=10000)

    # Contrastive arguments for contrastive method (including ours)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Contrastive arguments for contrastive method "
                        "(including ours)")
    parser.add_argument("--sim", type=str, default="dot")

    args = parser.parse_args()
    assert (args.method in ["FedDCSR", "FedVGSAN", "LocalVGSAN", "FedSASRec",
                            "LocalSASRec", "FedVSAN", "LocalVSAN",
                            "FedContrastVAE", "LocalContrastVAE", "FedCL4SRec",
                            "LocalCL4SRec", "FedDuoRec", "LocalDuoRec"])
    return args


def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_logger(args):
    """Init a file logger that opens the file periodically and write to it.
    """
    log_path = os.path.join(args.log_dir,
                            "domain_" + "".join([domain[0] for domain
                                                 in args.domains]))
    ensure_dir(log_path, verbose=True)

    model_id = args.id if len(args.id) > 1 else "0" + args.id
    log_file = os.path.join(log_path, args.method + "_" + model_id + ".log")

    logging.basicConfig(
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode="w+"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def main():
    args = arg_parse()

    seed_everything(args)

    init_logger(args)

    train_datasets, valid_datasets, test_datasets, adjs = load_dataset(args)

    n_clients = len(args.domains)
    clients = [Client(ModelTrainer, c_id, args, adjs[c_id],
                      train_datasets[c_id], valid_datasets[c_id],
                      test_datasets[c_id]) for c_id in range(n_clients)]
    # Initialize the aggretation weight
    init_clients_weight(clients)

    # Save the config of input arguments
    save_config(args)

    server = Server(args, clients[0].get_params())

    run_fl(clients, server, args)


if __name__ == "__main__":
    main()
