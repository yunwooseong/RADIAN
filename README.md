# [IEIE'25] Relational Alignment Graph Neural Network for Multimodal Recommendation
[![View Paper](https://img.shields.io/badge/View%20Paper-PDF-E24D35)](https://ieeexplore.ieee.org/document/11023251) [![DOI](https://img.shields.io/badge/DOI-10.1109/ACCESS.2025.3576445-blue)](https://doi.org/10.1109/ACCESS.2025.3576445)

### 📄 Paper 

This repository contains the implementation code for the Radian models as proposed in our paper.

### Datasets
We used the Baby, Sports and Clothing in our experiments. 
```
./datasets include download links.
```

### Requirements
All dependencies are listed in the `requirements.txt` file.  
Please install them using the following command:
```
pip install -r requirements.txt
```

## Run

### Baby
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset baby --model RADIAN
```
### Sports
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset sports --model RADIAN
```
### Clothing
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset clothing --model RADIAN
```
### Acknowledgements
