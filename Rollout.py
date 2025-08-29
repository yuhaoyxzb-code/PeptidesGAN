# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:58:14 2025

@author: CD
"""
import torch.nn as nn
import torch
import numpy as np
import copy
from generate_gru import gernarate
from dddd import Discriminator
from features import feature



class rollout():
    
    def __init__(self,model,up_date,seq_lens,device,features_num):
        
        self.old_model = model
        self.new_model = copy.deepcopy(model)
        self.update_rate = up_date
        self.seq_lens = seq_lens
        self.f = feature(features_num=features_num)
        self.device = device
        
    def rewards(self,N,x,discriminator):
        x = x.long()
        rewards = []
        # batch_size = x.size(0)
        seq_lens = x.size(1)
        for n in range(N):
            for seq_len in range(1,self.seq_lens):
                input_data = x[:,0:seq_len].to(self.device)
                samples = self.new_model.sample_out(input_data)
                r_temp = discriminator(samples)
                r_temp = r_temp.cpu().data[:,1].numpy()
                # print(r_temp)
                if n == 0:
                    rewards.append(r_temp)
                else:
                    rewards[seq_len-1] = rewards[seq_len-1] + r_temp
            r_temp = discriminator(x)
            r_temp = r_temp.cpu().data[:,1].numpy()
            if n == 0:
                rewards.append(r_temp)
            else:
                rewards[seq_lens-1] = rewards[seq_len-1] + r_temp
        rewards = np.transpose(np.array(rewards)) / (1.0 * N) # batch_size * seq_len
        return rewards
    
    def update_params(self):
        dic = {}
        for name, param in self.old_model.named_parameters():
            dic[name] = param.data
        for name, param in self.new_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]

