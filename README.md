# DeepBDC for few-shot learning
<div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="http://peihuali.org/DeepBDC/illustration.gif" width="80%"/>
</div>


## Introduction
In this repo, we provide the implementation of the following paper:"Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification"   [Project.](http://peihuali.org/DeepBDC/index.html),[Paper.](https://arxiv.org/pdf/2204.04567.pdf)
 In this paper, we propose deep Brown Distance Covariance (DeepBDC) for few-shot classification. DeepBDC can effectively learn image representations by measuring, for the query and support images, the discrepancy between the joint distribution of their embedded features and product of the marginals. The core of DeepBDC is formulated as a modular and efficient layer, which can be flexibly inserted into deep networks, suitable not only for meta-learning framework based on episodic training, but also for the simple transfer learning framework of pretraining plus linear classifier.<br>
```
@inproceedings{DeepBDC-CVPR2022,
  title={Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification},
  author={Jiangtao Xie and Fei Long and Jiaming Lv and Qilong Wang and Peihua Li}, 
  booktitle={CVPR},
  year={2022}
 }
```

If you find this repo helpful for your research, please consider citing our paper.<br>
## Few-shot classification Results
Experimental results on miniImageNet and CUB with ResNet-12 , ResNet-18 and ResNet-34. We report average results with 2,000 randomly sampled episodes for both 1-shot and  5-shot evaluation. More details on the experiments can be seen in the paper, and we will update some of our results on other datasets in the future.
### miniImageNet
<table>
         <tr>
             <th rowspan="2" style="text-align:center;">Method</th>
             <th colspan="2" style="text-align:center;">ResNet-12</th>
             <th colspan="2" style="text-align:center;">ResNet-34</th>
         </tr>
         <tr>
             <th colspan="1" style="text-align:center;">5-way-1-shot</th>
             <th colspan="1" style="text-align:center;">5-way-5-shot</th>
             <th colspan="1" style="text-align:center;">5-way-1-shot</th>
             <th colspan="1" style="text-align:center;">5-way-5-shot</th>
         </tr>
         <tr>
             <td style="text-align:center">ProtoNet</td>
             <td style="text-align:center;">62.11</td>
             <td style="text-align:center;">80.77</td>
             <td style="text-align:center;">64.56</td>
             <td style="text-align:center;">81.16</td>
         </tr>
         <tr>
             <td style="text-align:center">Good-Embed</td>
             <td style="text-align:center;">64.98</td>
             <td style="text-align:center;">82.10</td>
             <td style="text-align:center;">66.14</td>
             <td style="text-align:center;">82.39</td>
         </tr>
         <tr>
             <td style="text-align:center">Meta DeepBDC</td>
             <td style="text-align:center;">67.34</td>
             <td style="text-align:center;">84.46</td>
             <td style="text-align:center;">68.20</td>
             <td style="text-align:center;">84.97</td>
         </tr>
         <tr>
             <td style="text-align:center">STL DeepBDC</td>
             <td style="text-align:center;">67.83</td>
             <td style="text-align:center;">85.45</td>
             <td style="text-align:center;">68.66</td>
             <td style="text-align:center;">85.47</td>
         </tr>
</table>

### CUB
<table>
         <tr>
             <th rowspan="2" style="text-align:center;">Method</th>
             <th colspan="2" style="text-align:center;">ResNet-18</th>
             <th colspan="2" style="text-align:center;">ResNet-34</th>
         </tr>
         <tr>
             <th colspan="1" style="text-align:center;">5-way-1-shot</th>
             <th colspan="1" style="text-align:center;">5-way-5-shot</th>
             <th colspan="1" style="text-align:center;">5-way-1-shot</th>
             <th colspan="1" style="text-align:center;">5-way-5-shot</th>
         </tr>
         <tr>
             <td style="text-align:center">ProtoNet</td>
             <td style="text-align:center;">80.90</td>
             <td style="text-align:center;">89.81</td>
             <td style="text-align:center;">80.58</td>
             <td style="text-align:center;">90.11</td>
         </tr>
         <tr>
             <td style="text-align:center">Good-Embed</td>
             <td style="text-align:center;">77.92</td>
             <td style="text-align:center;">89.94</td>
             <td style="text-align:center;">79.33</td>
             <td style="text-align:center;">90.10</td>
         </tr>
         <tr>
             <td style="text-align:center">Meta DeepBDC</td>
             <td style="text-align:center;">83.55</td>
             <td style="text-align:center;">93.82</td>
             <td style="text-align:center;">85.25</td>
             <td style="text-align:center;">94.31</td>
         </tr>
         <tr>
             <td style="text-align:center">STL DeepBDC</td>
             <td style="text-align:center;">84.01</td>
             <td style="text-align:center;">94.02</td>
             <td style="text-align:center;">84.69</td>
             <td style="text-align:center;">94.33</td>
         </tr>
</table>

### Cross-domain
<table>
         <tr>
             <th rowspan="3" style="text-align:center;">Method</th>
             <th colspan="1" style="text-align:center;">miniImageNet->CUB</th>
             <th colspan="1" style="text-align:center;">miniImageNet->Aircraft</th>
             <th colspan="1" style="text-align:center;">miniImageNet->Cars</th>
         </tr>
         <tr>
             <th colspan="3" style="text-align:center;">ResNet-12</th>
         </tr>
         <tr>
             <th colspan="3" style="text-align:center;">5-way-5-shot</th>
         </tr>
         <tr>
             <td style="text-align:center">ProtoNet</td>
             <td style="text-align:center;">67.19</td>
             <td style="text-align:center;">55.96</td>
             <td style="text-align:center;">46.30</td>
         </tr>
         <tr>
             <td style="text-align:center">Good-Embed</td>
             <td style="text-align:center;">67.43</td>
             <td style="text-align:center;">58.95</td>
             <td style="text-align:center;">50.18</td>
         </tr>
         <tr>
             <td style="text-align:center">Meta DeepBDC</td>
             <td style="text-align:center;">77.87</td>
             <td style="text-align:center;">68.67</td>
             <td style="text-align:center;">54.61</td>
         </tr>
         <tr>
             <td style="text-align:center">STL DeepBDC</td>
             <td style="text-align:center;">80.16</td>
             <td style="text-align:center;">69.07</td>
             <td style="text-align:center;">58.09</td>
         </tr>
</table>

## Implementation details
### Prerequisites
- Linux
- Python 3.8.3
- torch 1.7.1
- GPU + CUDA11.0 CuDNN
- sklearn1.0.1,  pillow8.0.0, numpy1.19.2
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
### **For Meta DeepBDC on general object recognition**
1. `cd scripts/miniImagenet/run_meta_deepbdc`
2.  modify the dataset path in `run_pretrain.sh`、`run_metatrain.sh` and `run_test.sh`
3. `bash run_all.sh`


### **For STL DeepBDC on general object recognition**
1. `cd scripts/miniImagenet/run_stl_deepbdc`
2.  modify the dataset path in `run_pretrain.sh`、`run_metatrain.sh` and `run_test.sh`
3. `bash run_all.sh`
## Download  Models


[Pre-trained Models](www.biying.com)
(or run `bash download_pretrain_model.sh`)

[Meta-trained Models](www.biying.com)
(or run `bash download_trained_model.sh`)

## Acknowledgment
Our project references the codes in the following repos.
- [Baseline++](https://github.com/wyharveychen/CloserLookFewShot)

## Contact

**If you have any questions or suggestions, please contact us**

`longfei121@mail.dlut.edu.cn`<br>
`ljm_vlg@mail.dlut.edu.cn`



