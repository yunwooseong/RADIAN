# [IEIE'25] Relational Alignment Graph Neural Network for Multimodal Recommendation
[![View Paper](https://img.shields.io/badge/View%20Paper-PDF-E24D35)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12332645)

## 📄 Paper 

This repository contains the implementation code for the Radian models as proposed in our paper.

## Datasets
We used the Baby, Sports and Clothing in our experiments. 
```
./datasets include download links.
```

## Requirements
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
## Acknowledgements
본  연구는  과학기술정보통신부  및  정보통신기획평가원의    대학  ICT  연구센터사업의  연구결과  (IITP-2024-RS-2024-00438056)와  정부(과학기술정보통신부)의  재원으로  한국연구재단의  지원을  받아  수행된  연구임(No.RS-2024-00419201),  2025년도  정부(과학기술정보부)의  재원으로  정보통신기획평가원의 지원(RS-2021-II211341,  인공지능대학원지원(중앙대학교))을  받아  수행된  연구임.   
