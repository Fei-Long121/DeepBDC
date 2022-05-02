import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time
import os
from data.datamgr import SetDataManager

from methods.protonet import ProtoNet
from methods.good_embed import GoodEmbed
from methods.meta_deepbdc import MetaDeepBDC
from methods.stl_deepbdc import STLDeepBDC

from utils import *
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=84, type=int, choices=[84, 224])
parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet', 'tiered_imagenet', 'cub'])
parser.add_argument('--data_path', type=str)
parser.add_argument('--model', default='ResNet12', choices=['ResNet12', 'ResNet18'])
parser.add_argument('--method', default='stl_deepbdc', choices=['meta_deepbdc', 'stl_deepbdc', 'protonet', 'good_embed'])

parser.add_argument('--test_n_way', default=5, type=int, help='number of classes used for testing (validation)')
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')

parser.add_argument('--test_n_episode', default=2000, type=int, help='number of episodes in test')
parser.add_argument('--model_path', default='', help='meta-trained or pre-trained model .tar file path')
parser.add_argument('--test_task_nums', default=5, type=int, help='test numbers')
parser.add_argument('--gpu', default='0', help='gpu id')

parser.add_argument('--penalty_C', default=0.1, type=float, help='logistic regression penalty parameter')
parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimensions of BDC dimensionality reduction layer')
parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')

params = parser.parse_args()
num_gpu = set_gpu(params)

json_file_read = False
if params.dataset == 'cub':
    novel_file = 'novel.json'
    json_file_read = True
else:
    novel_file = 'test'


novel_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
novel_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.test_n_episode, json_read=json_file_read,  **novel_few_shot_params)
novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)

if params.method == 'protonet':
    model = ProtoNet(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'good_embed':
    model = GoodEmbed(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'meta_deepbdc':
    model = MetaDeepBDC(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'stl_deepbdc':
    model = STLDeepBDC(params, model_dict[params.model], **novel_few_shot_params)

# model save path
model = model.cuda()
model.eval()

print(params.model_path)
model_file = os.path.join(params.model_path)
model = load_model(model, model_file)

print(params)
iter_num = params.test_n_episode
acc_all_task = []
for _ in range(params.test_task_nums):
    acc_all = []
    test_start_time = time.time()
    tqdm_gen = tqdm.tqdm(novel_loader)
    for _, (x, _) in enumerate(tqdm_gen):
        with torch.no_grad():
            model.n_query = params.n_query
            scores = model.set_forward(x, False)
        if params.method in ['meta_deepbdc', 'protonet']:
            pred = scores.data.cpu().numpy().argmax(axis=1)
        else:
            pred = scores
        y = np.repeat(range(params.test_n_way), params.n_query)
        acc = np.mean(pred == y) * 100

        acc_all.append(acc)
        tqdm_gen.set_description(f'avg.acc:{(np.mean(acc_all)):.2f} (curr:{acc:.2f})')

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%% (Time uses %.2f minutes)'
        % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num), (time.time() - test_start_time) / 60))
    acc_all_task.append(acc_all)

acc_all_task_mean = np.mean(acc_all_task)
print('%d test mean acc = %4.2f%%' % (params.test_task_nums, acc_all_task_mean))

