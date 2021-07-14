#coding=utf8
import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
from itertools import cycle
import tensorflow as tf
from torchvision import transforms
import torch.utils.data
import torchvision.datasets as datasets
import torch
from torch.autograd import Variable
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision
from math import log
import math
from scipy import interp
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
import shutil
import argparse
from sklearn.metrics import roc_curve, auc
from gfnet import GFNet
import csv

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if USE_CUDA:
    torch.cuda.set_device(0)
tf.gfile.DeleteRecursively('./logs')  # 删除之前的记录
n_class = 9


def softmax(x):
    return np.exp(x) / np.exp(x).sum(-1, keepdims=True)


def onehot(x, n_class=n_class):
    return np.eye(n_class)[x].tolist()

choose_list=[1,2,3,4,5]
for i in choose_list:

    BATCH_SIZE = 8
    EPOCH_NUM = 15
    LR = 0.1
    model_type = ["gfnet","res2net","res2net_bi","res2net_mish","res2net_conv7to3","resnet","res2net_mish+bi+73","res2net_bi+73","res2net_mish+73"]
    choose = i
    print(i,model_type[choose])
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--image_dir', type=str, default='train_val')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--resume', default='./runs/' +model_type[choose]+ '_/checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--action', default='', type=str,
                        help='other information.')
    parser.add_argument('--arch', metavar='ARCH', type=str, default=model_type[choose], help="")

    from res2net_v3_0 import res2net50_26w_4s,res2net50_26w_4s_bi
    from resnet import resnet50


    def getdata(path,batch_size=BATCH_SIZE):
        '''
        ImageFolder方法根据文件夹命名label，可在下面迭代器中以label.data返回类别标签
        按随机顺序读取图片
        :param path:
        :param batch_size:
        :return:
        '''
        traindir = path+"/train"
        valdir = path+"/val"

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),  # 随机裁剪成224个像素
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
             pin_memory=True)

        return train_loader, val_loader


    def eca_resnet50(k_size=[3, 3, 3, 3], num_classes=1000, mode='eca'):
        '''
        resnet50模型构建函数
        :param k_size:
        :param num_classes:
        :param mode:
        :return:
        '''
        print("Constructing eca_resnet50 with block-{}......".format(mode))
        if mode == 'eca':
            model = ResNet(ECABottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
        elif mode == 'eca_ns':
            model = ResNet(ECA_NS_Bottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
        else:
            print("输入模型错误，请重新输入.")

        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model

    def eca_resnet18(k_size=[3, 3, 3, 3], num_classes=1000, mode='eca'):
        '''
        resnet50模型构建函数
        :param k_size:
        :param num_classes:
        :param mode:
        :return:
        '''
        print("Constructing eca_resnet50 with block-{}......".format(mode))
        if mode == 'eca':
            model = ResNet(ECABottleneck, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
        elif mode == 'eca_ns':
            model = ResNet(ECA_NS_Bottleneck, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
        else:
            print("输入模型错误，请重新输入.")

        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model

    def eca_resnet101(k_size=[3, 3, 3, 3], num_classes=n_class, pretrained=False):
        """Constructs a ResNet-101 model.

        Args:
            k_size: Adaptive selection of kernel size
            num_classes:The classes of classification
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = ResNet(ECABottleneck, [3, 4, 23, 3], num_classes=num_classes, k_size=k_size)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model


    def eca_resnet152(k_size=[3, 3, 3, 3], num_classes=n_class, pretrained=False):
        """Constructs a ResNet-152 model.

        Args:
            k_size: Adaptive selection of kernel size
            num_classes:The classes of classification
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = ResNet(ECABottleneck, [3, 8, 36, 3], num_classes=num_classes, k_size=k_size)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model




    use_gpu = torch.cuda.is_available()


    def train(train_loader, model, criterion, optimizer, epoch):
        model.train()
        dataset_sizes = len(train_loader.dataset)
        # print(dataset_sizes)
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for i, (inputs, labels) in enumerate(train_loader):
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # scheduler.step(epoch)

            # statistics
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == labels.data)


            batch_loss = running_loss / ((i + 1) * BATCH_SIZE)
            batch_acc = float(running_corrects) / ((i + 1) * BATCH_SIZE)

            '''if i % 50 == 0:
                print('[Epoch {}/{}]-[batch:{}/{}]  Loss: {:.6f}  Acc: {:.4f}'.format(
                    epoch, EPOCH_NUM, i, round(dataset_sizes / BATCH_SIZE) - 1, batch_loss, batch_acc))'''


        epoch_loss = running_loss / dataset_sizes
        epoch_acc = float(running_corrects) / dataset_sizes


        print('[Epoch {}/{}] - train Loss: {:.4f} Acc: {:.4f}'.format(epoch, EPOCH_NUM,
            epoch_loss, epoch_acc))
        return epoch_loss, epoch_acc



    def val(val_loader, model, criterion, optimizer):
        model.eval()
        dataset_sizes = len(val_loader.dataset)
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = model(inputs)
                # 找到概率最大的下标
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)



        epoch_loss = running_loss / dataset_sizes
        epoch_acc = float(running_corrects) / dataset_sizes

        print('[Epoch {}/{}] - test Loss: {:.4f} Acc: {:.4f}'.format(epoch, EPOCH_NUM,
            epoch_loss, epoch_acc))
        return epoch_loss, epoch_acc

    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        directory = "runs/%s/" % (args.arch + '_' + args.action)
        if not os.path.exists(directory):  # 如果路径不存在
            os.makedirs(directory)
        filename = directory + filename
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, directory + 'model_best.pth.tar')


    def plot_roc_auc(arch, y_preds, y_test,y_score):
        print(y_test)
        # print(y_preds.shape)
        # print(y_test.shape)
        print(y_score)
        # 计算每一类的ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_class):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area（方法二）
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area（方法一）
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_class):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_class
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw = 2
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'pink', 'purple', 'brown'])
        for i, color in zip(range(n_class), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(arch+'figure.jpg')



    def val_last(val_loader, model, criterion, optimizer, arch):
        model.eval()
        y = []
        pred = []
        scores = []
        # Iterate over data.
        with torch.no_grad():
            for inputs, labels in val_loader:
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                outputs = model(inputs)
                # 找到概率最大的下标
                _, preds = torch.max(outputs.data, 1)
                for i in preds.cpu():
                    # print('preds ', i.shape)  # []
                    pred.append(onehot(i))

                for i in labels.cpu():
                    y.append(onehot(i))
                for i in outputs.data.cpu():
                #     print('outputs, i.shape', i.shape)  # 9
                #     print('softmax(i).shape',softmax(i).shape)  # 9
                    scores.append(softmax(i).tolist())


                # pred.append(np.array((preds.cpu().numpy())))
                # y.append(np.array((labels.cpu().numpy())))
                # scores.append(np.array((outputs.data.cpu().numpy())))
        print(np.array(scores).shape)
        plot_roc_auc(arch,y_preds= np.array(pred),y_score=np.array(scores),y_test=np.array(y))



    if __name__ == '__main__':
        best_prec1 = 0
        args = parser.parse_args()
        writer = SummaryWriter('logs', comment=args.arch)
        # ----------------eca嵌入块
        # 读取数据
        # path = './Tiny-image'
        train_loader, val_loader= getdata(args.image_dir,args.batch_size)

        #improve = 0 无改进
        #improve = 1 Mish
        #improve = 2 卷积层从7x7换为3x3
        # mode可选参数eca, eca_ns
        # model_type = ["res2net", "res2net_bi", "res2net_mish", "res2net_conv7to3", "resnet", "res2net_mish+bi+73",
        #             "res2net_bi+73", "res2net_mish+73"]
        if args.arch =="res2net":
            model = res2net50_26w_4s(pretrained=False,improve=0,num_classes=9)
        elif args.arch =="res2net_bi":
            model = res2net50_26w_4s_bi(pretrained=True,num_classes=9)
        elif args.arch == "res2net_mish":
            model = res2net50_26w_4s(pretrained=False, improve=1, num_classes=9)
        elif args.arch == "res2net_conv7to3":
            model = res2net50_26w_4s(pretrained=False, improve=2, num_classes=9)
        elif args.arch == "resnet":
            model = resnet50(num_classes=9)
        elif args.arch == "res2net_mish+bi+73":
            model = res2net50_26w_4s_bi(pretrained=False, improve=3, num_classes=9)
        elif args.arch == "res2net_bi+73":
            model = res2net50_26w_4s_bi(pretrained=False, improve=4, num_classes=9)
        elif args.arch == "res2net_mish+73":
            model = res2net50_26w_4s(pretrained=False, improve=5, num_classes=9)
        elif args.arch == "gfnet":
            model = GFNet()
        if use_gpu:
            model = model.cuda()

        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 定义优化器
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.00004)


        eca_train_loss = []
        eca_train_acc = []
        eca_test_loss = []
        eca_test_acc = []


        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                del checkpoint
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))



        for epoch in range(1, EPOCH_NUM + 1):
            t_l, t_a = train(train_loader,model,criterion,optimizer,epoch)
            eca_train_loss.append(t_l)
            eca_train_acc.append(t_a)
            v_l, v_a = val(val_loader, model, criterion, optimizer)
            eca_test_loss.append(v_l)
            eca_test_acc.append(v_a)
            writer.add_scalar('train_loss', t_l, epoch)
            writer.add_scalar('train_acc', t_a, epoch)
            writer.add_scalar('val_loss', v_l, epoch)
            writer.add_scalar('val_acc', v_a, epoch)

            # checkpoint
            is_best = v_l > best_prec1
            best_prec1 = max(v_l, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


        # 输出四个评价指标，用于比较
        output = pd.DataFrame(columns=['train_loss', 'train_acc', 'test_loss', 'test_acc'])
        output['train_loss'] = eca_train_loss
        output['train_acc'] = eca_train_acc
        output['test_loss'] = eca_test_loss
        output['test_acc'] = eca_test_acc
        val_last(val_loader, model, criterion, optimizer, args.arch)

        save_file_name = args.arch +".txt"
        output.to_csv(save_file_name, index=False, header=True)
        # output_ns.to_csv('output_eca_ns.txt', index=False, header=True)

