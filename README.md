# Unsupervised QSM
Pytorch training script for unsupervised physics informed QSM reconstruction


## USAGE

#### Training
```
python train.py --epochs 10 --resume True --bs 10
```
#### Testing
```
python test.py
```

## DATASET
The dataset subdirectories consists of volume patches made from full volume Brain local field images. The images were acquired from Sree Chitra Tirunal Institute for Medical Sciences and Technology (SCTIMST)

## MODEL
The model architechture used for training is inspired from : https://github.com/dlQSM/dlQSM?tab=readme-ov-file
