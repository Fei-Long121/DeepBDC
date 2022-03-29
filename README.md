# DeepBDC for few-shot learning
<div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="http://peihuali.org/pictures/DeepBDC.jpg" width="80%"/>
</div>


## Introduction
We provide a PyTorch implementation of DeepBDC for few-shot learning:[Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification.](www.baidu.com)
In this paper, we propose deep Brown Distance Covariance (DeepBDC) for few-shot classification. DeepBDC can effectively learn image representations by measuring,for the query and support images, the discrepancy between the joint distribution of their embedded features and product of the marginals. The core of DeepBDC is formulated as a modular and efficient layer, which can be flexibly inserted into deep networks, suitable not only for metalearning framework based on episodic training, but also for the simple transfer learning framework that relies on nonepisodic training.<br>
```
@inproceedings{xie2022DeepBDC,
  title={Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification},
  author={Xie, Jiangtao and Long, Fei and Lv, Jiaming and Wang, Qilong and Li, Peihua}, 
  booktitle={CVPR},
  year={2022}
 }
```
If you use this code for your research, please cite our paper.<br>
## Few-shot classification Results
Experimental results on CUB and MiniImageNet with ResNet-12 , ResNet-18 and ResNet-34. We report average results with 2,000 randomly sampled episodes for both 1-shot and  5-shot evaluation.More details about the experiments can be seen in the paper, and We will update some of our results on other datasets in the future.
### MiniImageNet
<table>
         <tr>
             <th rowspan="3" style="text-align:center;">Method</th>
             <th colspan="4" style="text-align:center;">ResNet-12</th>
             <th colspan="4" style="text-align:center;">ResNet-34</th>
         </tr>
         <tr>
             <th colspan="2" style="text-align:center;">1-shot</th>
             <th colspan="2" style="text-align:center;">5-shot</th>
             <th colspan="2" style="text-align:center;">1-shot</th>
             <th colspan="2" style="text-align:center;">5-shot</th>
         </tr>
         <tr>
             <td style="text-align:center">paper</td>
             <td style="text-align:center;">reproduce</td>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
         </tr>
         <tr>
             <td style="text-align:center">ProtoNet</td>
             <td style="text-align:center;">62.11</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">80.77</td>
             <td style="text-align:center;"><b>6.13</b></td>
             <td style="text-align:center;">64.56</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">81.16</td>
             <td style="text-align:center;"><b>6.13</b></td>
         </tr>
         <tr>
             <td style="text-align:center">Good-Embed</td>
             <td style="text-align:center;">64.98</td>
             <td style="text-align:center;"><b>20.99</b></td>
             <td style="text-align:center;">82.10</td>
             <td style="text-align:center;"><b>5.56</b></td>
             <td style="text-align:center;">66.14</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">82.39</td>
             <td style="text-align:center;"><b>6.13</b></td>
         </tr>
         <tr>
             <td style="text-align:center">Meta DeepBDC</td>
             <td style="text-align:center;">67.34</td>
             <td style="text-align:center;"><b>20.99</b></td>
             <td style="text-align:center;">84.46</td>
             <td style="text-align:center;"><b>5.56</b></td>
             <td style="text-align:center;">68.20</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">84.97</td>
             <td style="text-align:center;"><b>6.13</b></td>
         </tr>
         <tr>
             <td style="text-align:center">STL DeepBDC</td>
             <td style="text-align:center;">67.83</td>
             <td style="text-align:center;"><b>20.99</b></td>
             <td style="text-align:center;">85.45</td>
             <td style="text-align:center;"><b>5.56</b></td>
             <td style="text-align:center;">68.66</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">85.47</td>
             <td style="text-align:center;"><b>6.13</b></td>
         </tr>
</table>

### CUB
<table>
         <tr>
             <th rowspan="3" style="text-align:center;">Method</th>
             <th colspan="4" style="text-align:center;">ResNet-18</th>
             <th colspan="4" style="text-align:center;">ResNet-34</th>
         </tr>
         <tr>
             <th colspan="2" style="text-align:center;">1-shot</th>
             <th colspan="2" style="text-align:center;">5-shot</th>
             <th colspan="2" style="text-align:center;">1-shot</th>
             <th colspan="2" style="text-align:center;">5-shot</th>
         </tr>
         <tr>
             <td style="text-align:center">paper</td>
             <td style="text-align:center;">reproduce</td>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
         </tr>
         <tr>
             <td style="text-align:center">ProtoNet</td>
             <td style="text-align:center;">80.90</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">89.81</td>
             <td style="text-align:center;"><b>6.13</b></td>
             <td style="text-align:center;">80.58</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">90.11</td>
             <td style="text-align:center;"><b>6.13</b></td>
         </tr>
         <tr>
             <td style="text-align:center">Good-Embed</td>
             <td style="text-align:center;">77.92</td>
             <td style="text-align:center;"><b>20.99</b></td>
             <td style="text-align:center;">89.94</td>
             <td style="text-align:center;"><b>5.56</b></td>
             <td style="text-align:center;">79.33</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">90.10</td>
             <td style="text-align:center;"><b>6.13</b></td>
         </tr>
         <tr>
             <td style="text-align:center">Meta DeepBDC</td>
             <td style="text-align:center;">83.55</td>
             <td style="text-align:center;"><b>20.99</b></td>
             <td style="text-align:center;">93.82</td>
             <td style="text-align:center;"><b>5.56</b></td>
             <td style="text-align:center;">85.25</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">94.31</td>
             <td style="text-align:center;"><b>6.13</b></td>
         </tr>
         <tr>
             <td style="text-align:center">STL DeepBDC</td>
             <td style="text-align:center;">84.01</td>
             <td style="text-align:center;"><b>20.99</b></td>
             <td style="text-align:center;">94.02</td>
             <td style="text-align:center;"><b>5.56</b></td>
             <td style="text-align:center;">84.69</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">94.33</td>
             <td style="text-align:center;"><b>6.13</b></td>
         </tr>
</table>

## Implementation details
### Prerequisites
- Linux
- Python 3.5
- Pytorch 1.3
- GPU + CUDA CuDNN
- sklearn, torchvision, scipy, numpy
### Installation

- Clone this repo:
```bash
git clone https://github.com/Fei-Long121/DeepBDC.git
cd DeepBDC
```
### Datasets
- [miniImageNet](www.biying.com)
- [tieredImageNet](www.biying.com)<br> 
- [CUB](www.biying.com)<br>
- [Aircraft](www.biying.com)<br>
- [Cars](www.biying.com)<br>
### **For Meta DeepBDC**
1. `cp run/miniImagenet/meta_deepbdc.sh ./`
2.  modify the dataset path in `meta_deepbdc.sh`
3. `sh meta_deepbdc.sh`


### **For STL DeepBDC**
1. `cp run/miniImagenet/stl_deepbdc.sh ./`
2.  modify the dataset path in `stl_deepbdc.sh`
3. `sh stl_deepbdc.sh`
## Download  Models


[Pre-trained Models](www.biying.com)
(or run `bash download_pretrain_model.sh`)

[Meta-trained Models](www.biying.com)
(or run `bash download_trained_model.sh`)

## Acknowledgment
Our project references the codes in the following repos.
- [Baseline++](www.biying.com)

## Contact

**If you have any questions or suggestions, please contact me**

`longfei121@mail.dlut.edu.cn`



