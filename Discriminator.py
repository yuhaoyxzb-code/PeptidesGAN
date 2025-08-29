#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:29:16 2025

@author: yuhao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from features import feature

# random.seed(42)
# np.random.seed(42) 

class Discriminator(nn.Module):
    def __init__(self, out_channels, con_kernel_sizes, pl_kernel_sizes, pd_sizes, l_in, drop_rate, f_model, device, batch_size, max_len):
        
        super(Discriminator, self).__init__()
        
        self.f = f_model
        
        self.u_device = device
        
        self.out_channels = out_channels
        self.covs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_c, out_c, con_ks,padding=pd),
                          nn.ReLU(),
                          nn.MaxPool1d(pl_ks))
            for in_c,out_c,con_ks,pl_ks,pd in zip(
                    [1] + out_channels[:-1],
                    out_channels,
                    con_kernel_sizes,
                    pl_kernel_sizes,
                    pd_sizes)])
        
        self.linear1 = nn.Linear(l_in, 128)
        
        self.linear2 = nn.Linear(128, 2)
        
        self.re = nn.ReLU()
        
        self.logs = nn.LogSoftmax(dim=1)
        
        self.drog = nn.Dropout(drop_rate)
        
        self.res = nn.Conv1d(1, out_channels[-1], kernel_size=1)
        # self.res = nn.ModuleList([
        #     nn.Conv1d(1, re_s, 1)
        #     for re_s in out_channels])
        
        # self.res_size = res_size
        
        self.batch_size = batch_size
        
        self.max_len = max_len
        
    #     self._init_weights()
        
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight)
    #             nn.init.constant_(m.bias, 0)
        
    def forward(self,x):
        
        # x = x.to(self.u_device)
        
        x = self.f.cal(x,self.batch_size,self.max_len).unsqueeze(1).float().to(self.u_device)
        
        residual = self.res(x)
        
        for con in self.covs:
            
            x = con(x)
        residual = F.adaptive_max_pool1d(residual, x.shape[-1])
        x += residual
        
        # residual_resized = F.interpolate(residual, size=(self.res_size), mode='linear', align_corners=False)

        # x += residual_resized
        # print(x.shape)
        x = x.view(x.size(0),-1)
        # print(x.shape)
        x = self.linear1(x)
        
        x = self.re(x)
        
        x = self.drog(x)
        
        x = self.linear2(x)        
        x = self.logs(x)
        
        return x
    
