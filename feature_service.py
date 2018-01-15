# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net 
from PIL import Image
import io
import cv2


class FeatureService(object):

    def __init__(self, opt):
        self.opt = opt
        self.use_gpu = False
        self.config_cuda()
        self.name = self.opt.name
        self.which_epoch = self.opt.which_epoch
        self.data_transforms = transforms.Compose([
            transforms.Resize((288,144), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = self.initialize_network()

    def config_cuda(self):
        str_ids = self.opt.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >=0:
                gpu_ids.append(id)
        # set gpu ids
        if len(gpu_ids)>0:
            torch.cuda.set_device(gpu_ids[0])
        self.use_gpu = torch.cuda.is_available()

    def load_network(self, network):
        save_path = os.path.join('./model', self.name, 'net_%s.pth' % self.which_epoch)
        network.load_state_dict(torch.load(save_path))
        return network

    def initialize_network(self):
        model_structure = ft_net(751)
        model = self.load_network(model_structure)
        # Remove the final fc layer and classifier layer
        model.model.fc = nn.Sequential()
        model.classifier = nn.Sequential()
        # Change to test mode
        model = model.eval()
        if self.use_gpu:
            model = model.cuda()
        return model

    def fliplr(self, img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def extract_single_feature(self, model, img):
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, 2048).zero_()
        for i in range(2):
            if i == 1:
                img = self.fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff + f
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        return ff

    def feature(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tensor = self.data_transforms(img)
        img_tensor = img_tensor.expand(1, 3, 288, 144)
        output_feature = self.extract_single_feature(self.model, img_tensor)
        return output_feature

if __name__ == '__main__':
    ######################################################################
    # Options
    # --------
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids',default='2', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    parser.add_argument('--img_dir', default='test.jpg', type=str, help='path of input image')
    opt = parser.parse_args()

   
    feature_service = FeatureService(opt)

    img_path = '/home/chenkai/project/Person_reID_baseline_pytorch/query_images/subfolder/train.jpg'
    img_pil = Image.open(img_path)
    print("----------------------------")
    output_feature = feature_service.feature(img_pil)
    print(output_feature)


