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
The dataset subdirectories consists of volume patches made from full volume Brain local field images. The images were from : Yoon, Jaeyeon & Gong, Enhao & Chatnuntawech, Itthi & Bilgic, Berkin & Jingu, Lee & Jung, Woojin & Ko, Jingyu & Jung, Hosan & Setsompop, Kawin & Zaharchuk, Greg & Kim, Eung Yeop & Pauly, John & Lee, Jongho. (2018). Quantitative Susceptibility Mapping using Deep Neural Network: QSMnet. NeuroImage. 179. 10.1016/j.neuroimage.2018.06.030. 

## MODEL
The model architechture used for training is inspired from : https://github.com/dlQSM/dlQSM?tab=readme-ov-file
