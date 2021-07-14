import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
from res2net_v3_0 import res2net50_26w_4s,res2net50_26w_4s_bi
from resnet import resnet50

# model_type = ["res2net", "res2net_bi", "res2net_mish", "res2net_conv7to3", "resnet", "res2net_mish+bi+73",
        #             "res2net_bi+73", "res2net_mish+73"]
isres2net = False
model_name = "resnet"

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        name_list = []
        module_list = []
        for name, module in self.submodule._modules.items():

            name_list.append(name)
            module_list.append(module)
        if isres2net == True:
            name_list[0], name_list[1] = name_list[1], name_list[0]
            module_list[0],module_list[1] = module_list[1], module_list[0]
        for i in range(len(name_list)):
            name = name_list[i]
            module = module_list[i]
            if "fc" in name:
                x = x.view(x.size(0), -1)

            x = module(x)
            print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs


def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (256, 256))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature():
    pic_dir = './Tiny-image/train/n01443537/val_611.JPEG'
    transform = transforms.ToTensor()
    img = get_picture(pic_dir, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 插入维度
    img = img.unsqueeze(0)

    img = img.to(device)
    if model_name == "res2net":
        model = res2net50_26w_4s(pretrained=False, improve=0, num_classes=50)
    elif model_name == "res2net_bi":
        model = res2net50_26w_4s_bi(pretrained=False, num_classes=50)
    elif model_name == "res2net_mish":
        model = res2net50_26w_4s(pretrained=False, improve=1, num_classes=50)
    elif model_name == "res2net_conv7to3":
        model = res2net50_26w_4s(pretrained=False, improve=2, num_classes=50)
    elif model_name == "resnet":
        model = resnet50(num_classes=50)
    elif model_name == "res2net_mish+bi+73":
        model = res2net50_26w_4s_bi(pretrained=False, improve=3, num_classes=50)
    elif model_name == "res2net_bi+73":
        model = res2net50_26w_4s_bi(pretrained=False, improve=4, num_classes=50)
    elif model_name == "res2net_mish+73":
        model = res2net50_26w_4s(pretrained=False, improve=5, num_classes=50)
    net = model
    net =  net.cuda()
    #net =  resnet50(num_classes=50).cuda()
    #et = nn.DataParallel(net).cuda()
    #net.load_state_dict(torch.load('./runs/res2net_/checkpoint.pt'))

    #checkpoint = torch.load('./runs/resnet_/checkpoint.pt', map_location=torch.device('cpu'))
    checkpoint = torch.load('./runs/'+model_name+'_/checkpoint.pth.tar', map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    #net.load_state_dict(checkpoint['state_dict'],False)
    net.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #print("=> loaded checkpoint '{}' (epoch {})"
          #.format(args.resume, checkpoint['epoch']))

    exact_list =["conv1","layer1","layer2","layer3","layer4"]
    dst = './feautures/'+model_name
    therd_size = 256

    myexactor = FeatureExtractor(net, exact_list)
    outs = myexactor(img)
    #print(outs)
    for k, v in outs.items():
        features = v[0]
        #print(features.shape[0])
        iter_range = features.shape[0]
        for i in range(iter_range):
            # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            if 'fc' in k:
                continue
            #print(features.data.cpu())
            feature = features.data.cpu().numpy()
            #print(feature)
            #feature = feature.reshape((iter_range,1,1))
            feature_img = feature[i, :, :]
            #print(feature)
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

            dst_path = os.path.join(dst, k)

            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)

            dst_file = os.path.join(dst_path, str(i) + '.png')
            cv2.imwrite(dst_file, feature_img)


if __name__ == '__main__':
    get_feature()