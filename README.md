# DeepBDC in Pytorch
<div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="http://peihuali.org/pictures/BDC.JPG" width="80%"/>
</div>


## Introduction
We provide a PyTorch implementation of DeepBDC for few-shot learning:<br>
   [Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification.](www.baidu.com). <br>
If you use this code for your research, please cite our paper.<br>
```
@inproceedings{xie2022DeepBDC,
  title={Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification},
  author={Xie, Jiangtao and Long, Fei and Lv, Jiaming and Wang, Qilong and Li, Peihua}, 
  booktitle={CVPR},
  year={2022}
 }
```
## Few-shot classification Results
Experimental results on few-shot learning datasets with ResNet-12 backbone and ResNet-18 backbone. We report average results with 2,000 randomly sampled episodes for both 1-shot and  5-shot evaluation.
<table>
         <tr>
             <th rowspan="3" style="text-align:center;">Method</th>
             <th colspan="4" style="text-align:center;">ResNet-12</th>
             <th colspan="4" style="text-align:center;">ResNet-34</th>
         </tr>
         <tr>
             <td colspan="2" style="text-align:center;">1-shot</th>
             <td colspan="2" style="text-align:center;">5-shot</th>
             <td colspan="2" style="text-align:center;">1-shot</th>
             <td colspan="2" style="text-align:center;">5-shot</th>
         </tr>
         <tr>
             <td style="text-align:center">paper</td>
             <td style="text-align:center;">reproduce</td>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reperduce</td>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
         </tr>
         <tr>
             <td style="text-align:center">ProtoNet</td>
             <td style="text-align:center;">22.14</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">6.22</td>
             <td style="text-align:center;"><b>6.13</b></td>
             <td style="text-align:center;"><a href="https://drive.google.com/open?id=19TWen7p9UDyM0Ueu9Gb22NtouR109C6j">217.3MB</a></td>
             <td style="text-align:center;"><a href="https://pan.baidu.com/s/17TxANPJg_j2VyYgXV05OOQ">217.3MB</a></td>
             <td style="text-align:center;">6.22</td>
             <td style="text-align:center;"><b>6.13</b></td>
         </tr>
         <tr>
             <td style="text-align:center">Good-Embed</td>
             <td style="text-align:center;">21.21</td>
             <td style="text-align:center;"><b>20.99</b></td>
             <td style="text-align:center;">5.68</td>
             <td style="text-align:center;"><b>5.56</b></td>
             <td style="text-align:center;"><a href="https://drive.google.com/open?id=1riur7v3rZ7vnrdj2UZ7EBaTKEGSccYwg">289.9MB</a></td>
             <td style="text-align:center;"><a href="https://pan.baidu.com/s/1_H8MosgzPH0BBmlKw2sr5A">289.9MB</a></td>
             <td style="text-align:center;">6.22</td>
             <td style="text-align:center;"><b>6.13</b></td>
         </tr>
</table>

## Prerequisites
- Linux
- Python 3.5
- Pytorch 1.3
- GPU + CUDA CuDNN
- pillow, torchvision, scipy, numpy

## Implementation details
### Installation

- Clone this repo:
```bash
git clone https://github.com/longfei-png/DeepBDC.git
cd DeepBDC
```
### Datasets
- [miniImageNet](https://drive.google.com/file/d/1fUBrpv8iutYwdL4xE1rX_R9ef6tyncX9/view). 
- [tieredImageNet](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0).<br> 
  Note that all the images need to be stored into a file named "images", and the data splits are stored into "train.csv", "val.csv" and "test.csv", respectively.
 ## Train and Test
### **Meta DeepBDC**
#### Pretraining
```
python train_pretrain.py --dataset [DATASET_NAME] --data_path [DATA_PATH] --model [BACKBONE_NAME] --method [METHOD_NAME] --image_size [IMAGE_SIZE] --reduce_dim [REDUCE_DIM] --[OTHER_OPTISIONS]
```

For example, run
```
python train_pretrain.py --dataset miniImagenet --data_path /data/miniImagenet --model ResNet12 --method meta_bdc --image_size 84 --reduce_dim 640 --train_aug --gpu 0
```
#### Meta training
```
python train_meta.py --dataset [DATASET_NAME] --data_path [DATA_PATH] --model [BACKBONE_NAME] --method [METHOD_NAME] --image_size [IMAGE_SIZE] --reduce_dim [REDUCE_DIM] --n_shot [N_SHOT] --pretrain_path [PRETRAIN_MODEL_PATH]
```

For example, run
```
python train_meta.py --dataset miniImagenet --data_path /data/miniImagenet --model ResNet12 --method meta_bdc --image_size 84 --reduce_dim 640 --train_aug --gpu 0 --n_shot 1 --pretrain_path /pretrain_pth_save_path
```

#### Meta testing
```
python test_finetuning.py --dataset [DATASET_NAME] --data_path [DATA_PATH] --model [BACKBONE_NAME] --method [METHOD_NAME] --image_size [IMAGE_SIZE] --reduce_dim [REDUCE_DIM] --n_shot [N_SHOT] --model_path [MODEL_PATH]
```

For example, run
```
python test_finetuning.py --dataset miniImagenet --data_path /data/miniImagenet --model ResNet12 --method meta_bdc --image_size 84 --reduce_dim 640 --train_aug --gpu 0 --n_shot 1 --model_path /model_pth_save_path
```

### **STL DeepBDC**
#### Pretraining
```
python train_pretrain.py --dataset [DATASET_NAME] --data_path [DATA_PATH] --model [BACKBONE_NAME] --method [METHOD_NAME] --image_size [IMAGE_SIZE] --reduce_dim [REDUCE_DIM] --[OTHER_OPTISIONS]
```

For example, run
```
python train_pretrain.py --dataset miniImagenet --data_path /data/miniImagenet --model ResNet12 --method stl_bdc --image_size 84 --reduce_dim 128 --train_aug --gpu 0
```
#### Self distillation
```
python train_distillation.py --dataset [DATASET_NAME] --data_path [DATA_PATH] --model [BACKBONE_NAME] --method [METHOD_NAME] --image_size [IMAGE_SIZE] --reduce_dim [REDUCE_DIM] --teacher_path [TEACHER_MODEL_PATH]
```

For example, run
```
python train_meta.py --dataset miniImagenet --data_path /data/miniImagenet --model ResNet12 --method stl_bdc --image_size 84 --reduce_dim 128 --train_aug --gpu 0 --teacher_path /teacher_pth_save_path
```

#### Meta testing
```
python test_finetuning.py --dataset [DATASET_NAME] --data_path [DATA_PATH] --model [BACKBONE_NAME] --method [METHOD_NAME] --image_size [IMAGE_SIZE] --reduce_dim [REDUCE_DIM] --n_shot [N_SHOT] --model_path [MODEL_PATH]
```

For example, run
```
python test_finetuning.py --dataset miniImagenet --data_path /data/miniImagenet --model ResNet12 --method stl_bdc --image_size 84 --reduce_dim 128 --train_aug --gpu 0 --n_shot 1 --model_path /model_pth_save_path
```


