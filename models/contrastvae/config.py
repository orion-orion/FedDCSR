# -*- coding: utf-8 -*-
hidden_size = 256  # Lantent dim
num_heads = 4  # Number of self-attention heads
# Number of self-attention blocks,
# because ContrastVAE has both encoder and decoder,
# we set it to 1 for fairness
num_blocks = 1
num_gnn_layers = 1  # GNN depth
dropout_rate = 0.3  # Dropout rate p
leakey = 0.1  # Hyperparameter of LeakyReLU
# aug_method = "data_augmentation"
# aug_method = "model_augmentation"
# aug_method = "variational_augmentation"
aug_method = "variational_and_data_augmentation"
# aug_method = "None"
hidden_act = "relu"  # Hidden layer activation function
