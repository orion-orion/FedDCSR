<p align="center">
<img src="pic/FedDCSR-Framework.png" width="700" height="450">
</p>

<div align="center">

# FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning
*[Hongyu Zhang](https://github.com/orion-orion), Dongyi Zheng, Xu Yang, Jiyuan Feng, [Qing Liao](http://liaoqing.hitsz.edu.cn/)\**

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/orion-orion/FedDCSR)[![LICENSE](https://img.shields.io/github/license/orion-orion/FedDCSR)](https://github.com/orion-orion/FedDCSR/blob/main/LICENSE)[![TinyPy](https://img.shields.io/github/stars/orion-orion/FedDCSR?style=social)](https://github.com/orion-orion/FedDCSR)
<br/>
[![FedDCSR](https://img.shields.io/github/directory-file-count/orion-orion/FedDCSR)](https://github.com/orion-orion/FedDCSR) [![FedDCSR](https://img.shields.io/github/languages/code-size/orion-orion/FedDCSR)](https://github.com/orion-orion/FedDCSR)
</div>

## 1 Introduction

This is the source code and baselines of our paper *[FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning](https://arxiv.org/abs/2309.08420)*. In this paper, we propose **FedDCSR**, a novel federated cross-domain sequential recommendation framework via disentangled representation learning. 

## 2 Dependencies

Run the following command to install dependencies:
```bash
pip install -r requirements.txt
```
Note that my Python version is `3.8.13`. In addition, it is especially important to note that the Pytorch version needs to be `<=1.7.1`, otherwise the autograd engine of Pytorch will report an error.

## 3 Dataset

As used in many cross-domain recommendation methods, we utilize the publicly available datasets from [Amazon](https://jmcauley.ucsd.edu/data/amazon/}{https://jmcauley.ucsd.edu/data/amazon/) (an e-commerce platform) to construct the federated CSR scenarios. We select ten domains to generate three cross-domain scenarios: Food-Kitchen-Cloth-Beauty (FKCB), Movie-Book-Game (MBG), and Sports-Garden-Home (SGH). 

The preprocessed CSR datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1NnZN3LhzdpxwaHiOW8GAUS8noTbdLlQt?usp=drive_link). You can download them and place them in the `./data` path of this project.

## 4 Code Structure

```bash
FedDCSR
├── LICENSE                                     LICENSE file
├── README.md                                   README file 
├── checkpoint                                  Model checkpoints saving directory
│   └── ...
├── data                                        Data directory
│   └── ...
├── log                                         Log directory
│   └── ...
├── models                                      Local model packages
│   ├── __init__.py                             Package initialization file
│   ├── cl4srec                                 CL4SRec package
│   │   ├── __init__.py                         Package initialization
│   │   ├── cl4srec_model.py                    Model architecture
│   │   ├── config.py                           Model configuration file
│   │   └── modules.py                          Backbone modules (such as self-attention)
│   └── ...
├── pic                                         Picture directory
│   └── FedDCSR-Framework.png                   Model framework diagram
├──  utils                                      Tools such as data reading, IO functions, training strategies, etc.
│    ├── __init__.py                            Package initialization file
│    ├── data_utils.py                          Data reading
│    ├── io_utils.py                            IO functions
│    └── train_utils.py                         Training strategies
├── client.py                                   Client architecture   
├── dataloader.py                               Customized dataloader
├── dataset.py                                  Customized dataset          
├── fl.py                                       The overall process of federated learning
├── local_graph.py                              Local graph data structure
├── losses.py                                   Loss functions
├── main.py                                     Main function, including the complete data pipeline
├── requirements.txt                            Dependencies installation
├── server.py                                   Server-side model parameters and user representations aggregation
├── trainer.py                                  Training and test methods of FedDCSR and other baselines
└── .gitignore                                  .gitignore file
```


## 5 Train & Eval

### 5.1 Our method

To train FedDCSR (ours), you can run the following command:

```bash
python -u main.py \
        --epochs 40 \
        --local_epoch 3 \
        --eval_interval 1 \
        --frac 1.0 \
        --batch_size 256 \
        --log_dir log \
        --method FedDCSR \
        --anneal_cap 1.0 \
        --lr 0.001 \
        --seed 42 \
        Food Kitchen Clothing Beauty
```
There are a few points to note:

- the positional arguments `Food Kitchen Clothing Beauty` indicates training FedDCSR in FKCB scenario. If you want to choose another scenario, you can change it to `Move Book Game` (MBG) or `Sports Garden Home` (SGH).

- The argument `--anneal_cap` is used to control KL annealing for variantional method (including ours). For FKCB, `1.0` is the best; for MBG and SGH, `0.01` is the best.

- If you restart training the model in a certain scenario, you can add the parameter `--load_prep` to load the dataset preprocessed in the previous training to avoid repeated data preprocessing

To test FedDCSR, you can run the following command:
```bash
python -u main.py \
        --log_dir log \
        --method FedDCSR \
        --do_eval \
        --seed 42 \
        Food Kitchen Clothing Beauty
```
### 5.2 Baselines

To train other baselines (FedSASRec, FedVSAN, FedContrastVAE, FedCL4SRec, FedDuoRec), you can run the following command:
```bash
python -u main.py \
        --epochs 40 \
        --local_epoch 3 \
        --eval_interval 1 \
        --frac 1.0 \
        --batch_size 256 \
        --log_dir log \
        --method FedContrastVAE \
        --anneal_cap 1.0 \
        --lr 0.001 \
        --seed 42 \
        Food Kitchen Clothing Beauty
```
For the local version without federated aggregation, you can run the following command:

```bash
python -u main.py \
        --epochs 40 \
        --local_epoch 3 \
        --eval_interval 1 \
        --frac 1.0 \
        --batch_size 256 \
        --log_dir log \
        --method LocalContrastVAE \
        --anneal_cap 1.0 \
        --lr 0.001 \
        --seed 42 \
        Food Kitchen Clothing Beauty
```



## 6 Citation
If you find this work useful for your research, please kindly cite FedDCSR by:
```text
@misc{zhang2023feddcsr,
      title={FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning}, 
      author={Hongyu Zhang and Dongyi Zheng and Xu Yang and Jiyuan Feng and Qing Liao},
      year={2023},
      eprint={2309.08420},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```