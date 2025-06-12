# [IEIE'25] Relational Alignment Graph Neural Network for Multimodal Recommendation

## 📄 Paper 

## Requirements
All dependencies are listed in the `requirements.txt` file.  
Please install them using the following command:
```
pip install -r requirements.txt
```
### Datasets
We used the Baby, Sports and Clothing in our experiments. 
```
./datasets include download links.
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
