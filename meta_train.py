import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

from data.datamgr import SetDataManager

from methods.protonet import ProtoNet
from methods.meta_deepbdc import MetaDeepBDC
from utils import *


def train(params, base_loader, val_loader, model, stop_epoch):

    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    for epoch in range(0, stop_epoch):
        start = time.time()
        model.train()
        trainObj, top1 = model.train_loop(epoch, base_loader, optimizer)

        model.eval()
        valObj, acc = model.test_loop(val_loader)
        if acc > trlog['max_acc']:
            print("best model! save...")
            trlog['max_acc'] = acc
            trlog['max_acc_epoch'] = epoch
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if epoch % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        trlog['train_loss'].append(trainObj)
        trlog['train_acc'].append(top1)
        trlog['val_loss'].append(valObj)
        trlog['val_acc'].append(acc)
        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))

        lr_scheduler.step()

        print("This epoch use %.2f minutes" % ((time.time() - start) / 60))
        print("train loss is {:.2f}, train acc is {:.2f}".format(trainObj, top1))
        print("val loss is {:.2f}, val acc is {:.2f}".format(valObj, acc))
        print("model best acc is {:.2f}, best acc epoch is {}".format(trlog['max_acc'], trlog['max_acc_epoch']))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', default=84, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate of the backbone')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 80], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=100, type=int, help='Stopping epoch')
    parser.add_argument('--gpu', default='0', help='gpu id')

    parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet','tiered_imagenet','cub'])
    parser.add_argument('--val_dataset', default='mini_imagenet', choices=['mini_imagenet','tiered_imagenet','cub'])

    parser.add_argument('--data_path', type=str, help='dataset path')
    parser.add_argument('--val_data_path', type=str, help='validation dataset path')
    parser.add_argument('--model', default='ResNet12', choices=['ResNet12','ResNet18'])
    parser.add_argument('--method', default='meta_deepbdc', choices=['meta_deepbdc', 'protonet'])

    parser.add_argument('--train_n_episode', default=600, type=int, help='number of episodes in meta train')
    parser.add_argument('--val_n_episode', default=300, type=int, help='number of episodes in meta val')
    parser.add_argument('--train_n_way', default=5, type=int, help='number of classes used for meta train')
    parser.add_argument('--val_n_way', default=5, type=int, help='number of classes used for meta val')
    parser.add_argument('--n_shot', default=5, type=int,
                        help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=16, type=int,
                        help='number of unlabeled data in each class')

    parser.add_argument('--extra_dir', default='', help='record additional information')

    parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in pretrain')
    parser.add_argument('--pretrain_path', default='', help='pre-trained model .tar file path')
    parser.add_argument('--save_freq', default=50, type=int, help='the frequency of saving model .pth file')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimension of BDC dimensionality reduction layer')
    params = parser.parse_args()

    params.val_dataset = params.dataset
    params.val_data_path = params.data_path

    num_gpu = set_gpu(params)
    set_seed(params.seed)

    if params.dataset == 'mini_imagenet':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 64
    elif params.dataset == 'cub':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 100
    elif params.dataset == 'tiered_imagenet':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 351
    else:
        ValueError('dataset error')

    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.train_n_episode, dataset=params.dataset, **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(params.val_data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode,dataset=params.val_dataset, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

    if params.method == 'protonet':
        model = ProtoNet(params, model_dict[params.model], **train_few_shot_params)
    elif params.method == 'meta_deepbdc':
        model = MetaDeepBDC(params, model_dict[params.model], **train_few_shot_params)

    model = model.cuda()

    # model save path
    params.checkpoint_dir = './checkpoints/%s/%s_%s' % (params.dataset, params.model, params.method)
    params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    params.checkpoint_dir += '_metatrain'
    params.checkpoint_dir += params.extra_dir
    print(params.checkpoint_dir)

    model = model.cuda()
    
    print(params.pretrain_path)
    modelfile = os.path.join(params.pretrain_path)
    model = load_model(model, modelfile)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(params)
    model = train(params, base_loader, val_loader, model, params.epoch)
