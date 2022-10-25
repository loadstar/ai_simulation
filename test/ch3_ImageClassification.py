#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2022/10/24 11:31 下午
# @Author:Andrew duokunzhang@163.com
# @File：ch3_ImageClassification.py 使用ResNet（2015年ImageNet冠军）做猫vs狗的分类

import os
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder


# %matplotlib inline


# def main():
#     # Check if GPU is present
#     if torch.cuda.is_available():
#         is_cuda = True
#
#     # Load data into PyTorch tensors
#     simple_transform = transforms.Compose([transforms.Resize((224, 224))
#                                               , transforms.ToTensor()
#                                               , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     train = ImageFolder('/Users/andrewzhang/Documents/dev/Datasets/Kaggledogandcat/train/train/', simple_transform)
#     valid = ImageFolder('/Users/andrewzhang/Documents/dev/Datasets/Kaggledogandcat/train/valid/', simple_transform)
#
#     print(train.class_to_idx)
#     print(train.classes)
#
#     imshow(train[50][0])
#
#     # Create data generators
#     train_data_gen = torch.utils.data.DataLoader(train, shuffle=True, batch_size=64, num_workers=3)
#     valid_data_gen = torch.utils.data.DataLoader(valid, batch_size=64, num_workers=3)
#     dataset_sizes = {'train': len(train_data_gen.dataset), 'valid': len(valid_data_gen.dataset)}
#
#     dataloaders = {'train': train_data_gen, 'valid': valid_data_gen}
#
#     # Create a network
#
#     model_ft = models.resnet18(pretrained=True)
#     num_ftrs = model_ft.fc.in_features
#     model_ft.fc = nn.Linear(num_ftrs, 2)
#
#     if torch.cuda.is_available():
#         model_ft = model_ft.cuda()
#
#     print(model_ft)
#
#     # Loss and Optimizer
#     learning_rate = 0.001
#     criterion = nn.CrossEntropyLoss()
#     optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#
#     model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                            num_epochs=2)
#
#     print('done')


def prepare():
    # 读取所有文件
    path = '/Users/andrewzhang/Documents/dev/Datasets/Kaggledogandcat/train/'
    files = glob(os.path.join(path, '*.jpg'))
    print(f'Total no of images {len(files)}')


    # 创建可用于创建验证数据集的混合索引
    no_of_images = len(files) # no_of_images = 25000
    shuffle = np.random.permutation(no_of_images)

    # 创建保存验证集和训练集的文件夹
    os.mkdir(os.path.join(path, 'valid'))
    os.mkdir(os.path.join(path, 'train'))

    # 创建对应目录
    for t in ['train', 'valid']:
        for folder in ['dog/', 'cat/']:
            os.mkdir(os.path.join(path, t, folder))

    # 随机将2000图片子集复制到验证文件夹
    for i in shuffle[:2000]:
        # shutil.copyfile(files[i],'../chapter3/dogsandcats/valid/')
        folder = files[i].split('/')[-1].split('.')[0]
        image = files[i].split('/')[-1]
        os.rename(files[i], os.path.join(path, 'valid', folder, image))

    # 复制到train文件夹
    for i in shuffle[2000:]:
        # shutil.copyfile(files[i],'../chapter3/dogsandcats/valid/')
        folder = files[i].split('/')[-1].split('.')[0]
        image = files[i].split('/')[-1]
        os.rename(files[i], os.path.join(path, 'train', folder, image))


# 3-d张量数据可视化，变形并反值归一化
def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase 仅在训练阶段反向优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    # Check if GPU is present
    if torch.cuda.is_available():
        is_cuda = True

    # Load data into PyTorch tensors，用ImageFolder做图像的统一大小、归一化
    simple_transform = transforms.Compose([transforms.Resize((224, 224))
                                              , transforms.ToTensor()
                                              , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train = ImageFolder('/Users/andrewzhang/Documents/dev/Datasets/Kaggledogandcat/train/train/', simple_transform)
    valid = ImageFolder('/Users/andrewzhang/Documents/dev/Datasets/Kaggledogandcat/train/valid/', simple_transform)

    # 保留类别标签
    print(train.class_to_idx)
    print(train.classes)

    # 3-d张量数据可视化
    imshow(train[50][0])

    # Create data generators，数据集转换到数据加载器（data loader）中，num_workers并发，少于机器核数即可
    train_data_gen = torch.utils.data.DataLoader(train, shuffle=True, batch_size=64, num_workers=3)
    valid_data_gen = torch.utils.data.DataLoader(valid, batch_size=64, num_workers=3)
    dataset_sizes = {'train': len(train_data_gen.dataset), 'valid': len(valid_data_gen.dataset)}

    dataloaders = {'train': train_data_gen, 'valid': valid_data_gen}

    # Create a network 创建ResNet网络，最后一层输出特征调整为2，因为是二分类问题
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    if torch.cuda.is_available():
        model_ft = model_ft.cuda()

    print(model_ft)

    # Loss and Optimizer
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 动态修改学习率
    # exp_lr_scheduler = ExponentialLR(optimizer_ft, gamma=0.9)

    # 训练模型
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=2)

    print(model_ft)
    print('done')
