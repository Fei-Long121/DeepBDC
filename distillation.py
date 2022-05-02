from curses.panel import top_panel
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import argparse

from data.datamgr import SimpleDataManager, SetDataManager
from methods.template import BaselineTrain
from utils import *

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


def train_distill(params, base_loader, val_loader, model, model_t, stop_epoch):   

    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    if params.method in ['stl_deepbdc']:
        bas_params = filter(lambda p: id(p) != id(model.dcov.temperature), model.parameters())
        optimizer = torch.optim.SGD([
            {'params': bas_params}, 
            {'params': model.dcov.temperature, 'lr': params.t_lr}], lr=params.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=params.milestones, gamma=params.gamma)

    loss_fn = nn.CrossEntropyLoss()
    loss_div_fn = DistillKL(4)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    for epoch in range(0, stop_epoch):
        epoch_time = time.time()
        
        model.train()
        
        print_freq = 200
        avg_loss = 0
        avg_acc = 0
        start_time = time.time()
        for i, (x, y) in enumerate(base_loader):
            
            x = Variable(x.cuda())
            y = Variable(y.cuda())

            with torch.no_grad():
                scores_t = model_t(x)
            scores = model(x)
            loss_cls = loss_fn(scores, y)
            pred = scores.data.max(1)[1]
            train_acc = pred.eq(y.data.view_as(pred)).sum()
 
            loss_div = loss_div_fn(scores, scores_t)
            loss = loss_cls * 0.5 + loss_div * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            avg_acc = avg_acc + train_acc.item()

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Time {:.2f}'.format(epoch, i, len(base_loader), avg_loss / float(i + 1), time.time()-start_time))
            start_time = time.time()

        model.eval()
        if params.val in ['meta']:
            valObj, acc = model.meta_test_loop(val_loader)
            trlog['val_loss'].append(valObj)
            trlog['val_acc'].append(acc)
            if acc > trlog['max_acc']:
                print("best model! save...")
                trlog['max_acc'] = acc
                trlog['max_acc_epoch'] = epoch
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
                
            print("val loss is {:.2f}, val acc is {:.2f}".format(valObj, acc))
            print("model best acc is {:.2f}, best acc epoch is {}".format(trlog['max_acc'], trlog['max_acc_epoch']))

        if epoch == params.save_freq:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        
        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        avg_acc = avg_acc / (len(base_loader)*params.batch_size) * 100
        avg_loss = avg_loss / len(base_loader)

        trlog['train_loss'].append(avg_loss)
        trlog['train_acc'].append(avg_acc)

        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))

        lr_scheduler.step()  # lr decreased
    
        print('1 epoch use {:.2f}mins '.format((time.time() - epoch_time)/60))
        print("train loss is {:.2f}, train acc is {:.2f}".format(avg_loss, avg_acc))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', default=84, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
    parser.add_argument('--batch_size', default=64, type=int, help='pre-training batch size')
    parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate of the backbone')
    parser.add_argument('--t_lr', type=float, default=0.05, help='initial learning rate uesd for the temperature of bdc module')

    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--milestones', nargs='+', type=int, default=[80, 120], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=180, type=int, help='stopping epoch')
    parser.add_argument('--gpu', default='0', help='gpu id')

    parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet','tiered_imagenet','cub'])
    parser.add_argument('--data_path', type=str, help='dataset path')

    parser.add_argument('--model', default='ResNet12', choices=['ResNet12', 'ResNet18'])
    parser.add_argument('--method', default='stl_deepbdc', choices=['stl_deepbdc', 'good_embed'])

    parser.add_argument('--val', default='meta', choices=['meta', 'last'], help='validation method')
    parser.add_argument('--val_n_episode', default=1000, type=int, help='number of episode in meta validation')
    parser.add_argument('--val_n_way', default=5, type=int, help='class num to classify in meta validation')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support during meta validation')
    parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')

    parser.add_argument('--extra_dir', default='', help='record additional information')

    parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in training')
    parser.add_argument('--save_freq', default=50, type=int, help='saving model .pth file frequency')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    parser.add_argument('--teacher_path', default='', help='teacher model .tar file path')
    parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimensions of BDC dimensionality reduction layer')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    params = parser.parse_args()

    num_gpu = set_gpu(params)
    set_seed(params.seed)

    if params.val == 'last':
        val_file = None
    elif params.val == 'meta':
        val_file = 'val'

    json_file_read = False
    if params.dataset == 'mini_imagenet':
        base_file = 'train'
        params.num_classes = 64
    elif params.dataset == 'cub':
        base_file = 'base.json'
        val_file = 'val.json'
        json_file_read = True
        params.num_classes = 200
    elif params.dataset == 'tiered_imagenet':
        base_file = 'train'
        params.num_classes = 351
    else:
        ValueError('dataset error')


    base_datamgr = SimpleDataManager(params.data_path, params.image_size, batch_size=params.batch_size, json_read=json_file_read)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    if params.val == 'meta':
        test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    else:
        val_loader = None

    model = BaselineTrain(params, model_dict[params.model], params.num_classes)
    model_t = BaselineTrain(params, model_dict[params.model], params.num_classes)

    model = model.cuda()
    model_t = model_t.cuda()

    # model save path
    params.checkpoint_dir = './checkpoints/%s/%s_%s' % (params.dataset, params.model, params.method)
    params.checkpoint_dir += '_distill'
    params.checkpoint_dir += '_born{}'.format(params.trial)
    params.checkpoint_dir += params.extra_dir
    print(params.checkpoint_dir)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # teacher model load
    modelfile = os.path.join(params.teacher_path)
    tmp = torch.load(modelfile)
    state = tmp['state']
    model_t.load_state_dict(state)

    print(params)
    model = train_distill(params, base_loader, val_loader, model, model_t, params.epoch)
