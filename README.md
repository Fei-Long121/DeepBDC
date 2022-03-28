# DeepBDC in Pytorch
We provide a PyTorch implementation of DeepBDC for few-shot learning. <br>
If you use this code for your research, please cite our paper.<br>
```
@inproceedings{xie2022DeepBDC,
  title={Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification},
  author={Xie, Jiangtao and Long, Fei and Lv, Jiaming and Wang, Qilong and Li, peihua}, 
  booktitle={CVPR},
  year={2022}
 }
```
[Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification.](www.baidu.com)<br>
[Jiangtao Xie*](www.biying.com), FeiLong*, Jiaming Lv, Qilong Wang, Peihua Li
## Prerequisites
- Linux
- Python 3.5
- Pytorch 1.3
- GPU + CUDA CuDNN
- pillow, torchvision, scipy, numpy

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/WenbinLee/DeepBDC.git
cd DeepBDC
```
### Datasets
- [miniImageNet](https://drive.google.com/file/d/1fUBrpv8iutYwdL4xE1rX_R9ef6tyncX9/view). 
- [tieredImageNet](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0).<br> 
  Note that all the images need to be stored into a file named "images", and the data splits are stored into "train.csv", "val.csv" and "test.csv", respectively.

