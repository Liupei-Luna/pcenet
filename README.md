# SolarPCE-Net: Accelerating High-Efficiency Donor-Acceptor Pair Discovery in Organic Photovoltaics via Deep Learning
<p align="center"> <a href="https://doi.org/10.1039/d5ta04854k"> <img alt="DOI" src="https://img.shields.io/badge/DOI-10.1039%2Fd5ta04854k-blue.svg"> </a> <a href="https://github.com/your-username/SolarPCE-Net"> <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-success.svg"> </a> <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"> <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue.svg"> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg"> </p>

## News
[2025/10/21] SolarPCE-Net is now available on GitHub!

[2025/10/08]🎉🎉 Paper published in Journal of Materials Chemistry A: Accelerating the discovery of high-efficiency donor–acceptor pairs in organic photovoltaics via SolarPCE-Net guided screening.

[2025/10/06]📄 Paper accepted by Journal of Materials Chemistry A.

## Release Plan
Research Paper

Model Architecture and Training Code

HOPV15 Dataset Integration

Extended Dataset with Additional OPV Materials

Web-based Interactive Screening Platform

## Introduction
We present SolarPCE-Net, a novel deep learning framework for predicting power conversion efficiency (PCE) of donor-acceptor (D-A) pairs in organic photovoltaics (OPVs). The model integrates residual networks with self-attention mechanisms in a dual-channel architecture to capture intricate D-A coupling effects that critically influence device performance. SolarPCE-Net achieves state-of-the-art prediction accuracy (test R² = 0.81) on the HOPV15 dataset and enables high-throughput virtual screening of unexplored D-A combinations, accelerating the discovery of high-performance OPV materials.

![Figures/Fig 1.png](https://github.com/Liupei-Luna/pcenet/blob/main/Figures/Fig%201.png)


## Installation
Create a Conda environment and install dependencies using requirements.txt:

```bash
conda create --name solarpce python=3.10
conda activate solarpce
pip install -r requirements.txt
```
Or setup environment with provided YML file:
```bash
conda env create -f environment.yml
```

## Usage
Data Preparation
If you want to work with the HOPV15 dataset used in our study, you can download and preprocess it using:
```bash
python src/data_preparation.py --data_path ./data/HOPV15.csv
```
Model Training
Train the SolarPCE-Net model with default hyperparameters:
```bash
python src/train.py --data_path ./data/HOPV15_processed.csv --epochs 100 --batch_size 32 --lr 1e-3
```
PCE Prediction
Use the trained model to predict PCE for new donor-acceptor pairs:
```bash
python src/predict.py --model_path ./models/solarpce_net.pth --input_file ./data/new_pairs.csv
```
High-Throughput Screening
Perform virtual screening of unexplored D-A combinations:
```bash
python src/screen.py --model_path ./models/solarpce_net.pth --donor_library ./data/donors.csv --acceptor_library ./data/acceptors.csv
```

## Performance
SolarPCE-Net demonstrates superior performance compared to traditional machine learning and graph neural network methods:
![](https://github.com/Liupei-Luna/pcenet/blob/main/Figures/Fig%202.png)
![](https://github.com/Liupei-Luna/pcenet/blob/main/Figures/Fig%203.png)

## Interpretability and Key Molecular Descriptors
Through attention weight analysis and SHAP values, SolarPCE-Net identifies critical molecular substructures that significantly influence PCE, including sulfur-containing five-membered rings, enhanced conjugated systems, and linear conjugated structures. These insights provide actionable guidance for rational OPV material design.

## Citation
If you find this repository useful, please consider giving a star ⭐ and citing our paper:

```
text
@article{liu2025accelerating,
  title={Accelerating the discovery of high-efficiency donor--acceptor pairs in organic photovoltaics via SolarPCE-Net guided screening},
  author={Liu, Xingyu and Hu, Bo and Liu, Pei and Huang, Meng and Li, Ming and Wan, Yuwei and Hoex, Bram and Xie, Tong},
  journal={Journal of Materials Chemistry A},
  year={2025},
  doi={10.1039/d5ta04854k}
}
```

## Acknowledgments
This research was supported by:

National Natural Science Foundation of China (Grant 12401679)

Teaching Reform Research Project for Jiangsu Province Academic Degrees and Graduate Education of China (Grant JGKT25_B053)

Graduate Research and Practice Innovation Program of Jiangsu Province SJCX25_2117

Jiangsu Provincial Higher Education Basic Science (Natural Science) Research Project (Grant 25KJD520001)

Haiyan Project (Grant KK25015) funded by the Lianyungang government, China
