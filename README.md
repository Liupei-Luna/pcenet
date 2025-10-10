<h1 align="center">Accelerating discovery of high-efficiency donor-acceptor pairs in organic photovoltaics via SolarPCE-Net guided screening</h1>

<p align="center">    
    <a href="https://github.com/ocean-luna/pcenet">
        <img alt="Build" src="https://img.shields.io/badge/Github-Code-blue">
    </a>
</p>

[![DOI](https://zenodo.org/badge/DOI/10.xxxx/xxxx.svg)](https://doi.org/10.xxxx/xxxx) <!-- If published, replace xxxx with your DOI (如果已发表，请将 xxxx 替换为您的 DOI) -->
[![arXiv](https://img.shields.io/badge/arXiv-23xx.xxxx-b31b1b.svg)](https://arxiv.org/abs/23xx.xxxx) <!-- If available, replace 23xx.xxxx with your arXiv ID (如果可用，请将 23xx.xxxx 替换为您的 arXiv ID) -->
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) <!-- Or Apache 2.0, etc. (或者 Apache 2.0 等许可证) -->

## Table of Contents
- [Introduction](#introduction)
- [Key Contributions](#key-contributions)
- [Install](#install)
- [Data](#data)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact Us](#contact-us)

## Introduction
We propose the Solar Power Conversion Efficiency Network (SolarPCE-Net), a novel deep learning-based framework for OPV material screening that captures the intricate dynamics within D-A pairs. By integrating a residual network with self-attention mechanism, SolarPCE-Net employs a dual-channel architecture to process molecular descriptor signatures of D-A while quantifying interfacial coupling effects through attention-weighted feature fusion. We apply the proposed method to the HOPV15 dataset. Experimental results show that our proposed SolarPCE-Net exhibits certain advantages in terms of accuracy and generalization ability compared to traditional methods. Interpretability analysis by attention weighting reveals key molecular descriptors that influence performance. Our work screens undeveloped D-A combinations, demonstrating its potential to accelerate high-performance OPV material discovery.

**Abstract:** Organic photovoltaic (OPV) materials hold great potential in accelerating solar energy conversion. Rapid screening of high-performance donor-acceptor (D-A) materials helps reduce the cost and time consumption associated with traditional experimental trial-and-error methods. However, for predicting the power conversion efficiency (PCE) of D-A in OPV, the existing approaches focus on efficiency prediction of single-component material and neglect synergistic D-A coupling effects critical to device performance. Here, we propose the Solar Power Conversion Efficiency Network (SolarPCE-Net), a novel deep learning-based framework for OPV material screening that captures the intricate dynamics within D-A pairs. By integrating a residual network with self-attention mechanism, SolarPCE-Net employs a dual-channel architecture to process molecular descriptor signatures of D-A while quantifying interfacial coupling effects through attention-weighted feature fusion. We apply the proposed method to the HOPV15 dataset. Experimental results show that our proposed SolarPCE-Net exhibits certain advantages in terms of accuracy and generalization ability compared to traditional methods. Interpretability analysis by attention weighting reveals key molecular descriptors that influence performance. Our work screens undeveloped D-A combinations, demonstrating its potential to accelerate high-performance OPV material discovery.

![Figure 1: SolarPCE-Net framework](Fig 1.jpg) <!-- Ensure this path is correct relative to README.md (确保此路径相对于 README.md 正确) -->

## Key Contributions
The main contributions of this study include:

*   **Novel Framework**: Introduction of SolarPCE-Net, a framework specifically tailored for modeling donor-acceptor pairs in OPV materials, addressing the material-level coupling effect often overlooked in prior methods.
*   **Efficient Predictive Model**: Development of a predictive model capable of facilitating rapid screening and design of high-performance OPV material pairs, thereby accelerating solar cell development.
*   **Interpretability Analysis**: Insights into critical factors (molecular descriptors) influencing PCE through interpretability analysis enabled by the self-attention mechanism, guiding material optimization.
*   **Uncertainty Quantification**: Comprehensive quantification of prediction uncertainties, enhancing the reliability and trustworthiness of model predictions.
*   **Application Potential**: Successful application in screening unexplored D-A combinations, identifying D-A pairs with high PCE potential.

## Install
You can create a Conda environment and install dependencies using `requirements.txt`:
```bash
conda create --name SolarPCE python=3.10
conda activate SolarPCE
pip install -r requirements.txt
Installation & Running
This section guides you on how to set up the environment and run the project code.

## Data
We used the Harvard Organic Photovoltaics dataset (HOPV15) as the benchmark data, which comprises detailed information on 350 small-molecule and polymer electron donors, along with their corresponding 6 acceptor materials (PC61BM, PC71BM, TiO2, C60, PDI, ICB). The dataset reports molecular structures, experimentally determined power conversion efficiency (PCE), and quantum chemical electronic features (e.g., HOMO/LUMO).
Data Preprocessing
Three donor entries lacking acceptor information and three duplicates were removed, resulting in a refined dataset of 344 unique donor-acceptor (D-A) pairs.
Donor molecules were characterized using the MolSig program to generate signature descriptors (path lengths of 0-4 bonds), yielding 625 descriptors.
Acceptor molecules were represented using a simple but effective 1-hot encoding scheme, capturing essential differences, most relevantly their LUMO energy levels.
The dataset was split into an 80% training set and a 20% test set using the k-means clustering algorithm to ensure representative data partitioning.

## Results
Comparative Performance Analysis
SolarPCE-Net demonstrated superior predictive performance and generalization ability on the HOPV15 dataset.

![Figure 2: Comparison of different methods on the HOPV15 dataset](Fig 2.jpg) <!

SolarPCE-Net achieved the highest R² (0.81) and lowest MAE (0.35) and SE (0.45) on the test set, significantly outperforming traditional regression models, ensemble methods, and existing deep learning models (including GNNs). It strikes an optimal balance between training performance and test generalization, avoiding the overfitting issues prevalent in other deep learning approaches.

![Figure 3: Performance comparison of methods on the HOPV15 dataset using 5-fold cross-validation](Fig 3.jpg) <!

In 5-fold cross-validation, SolarPCE-Net also demonstrated superior generalization capabilities, achieving the highest average R² (0.6218 ± 0.1404) and the lowest MAE (0.4362 ± 0.0772) and SE (0.6480 ± 0.1867).

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
We would like to thank [the National Natural Science Foundation of China (Grant 12401679), the Teaching Reform Research Project for Jiangsu Province Academic Degrees and Graduate Education of China (Grant JGKT25_B053), the Graduate Research and Practice Innovation Program of Jiangsu Province SJCX25_2117, and the Haiyan Project (Grant KK25015) funded by the Lianyungang 867 government, China.].

## Contact Us
For any inquiries, please feel free to contact:

[Meng Huang] - [huangmeng@jou.edu.cn]
