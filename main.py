# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:31:08 2025

@author: CD
"""

import argparse
import pickle as pkl
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import numpy as np
import matplotlib.pyplot as plt
from generate import gernarate
from dataload_final1 import GenDataIter,DisDataIter
from dddd import Discriminator
from Rollout import rollout
from features import feature
from tqdm import tqdm
from plt_nature import feature_loss
import pandas as pd
import time

# random.seed(42)
# np.random.seed(42) 

if torch.cuda.is_available():
    print("CUDA is available! GPU will be used.")
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_samples(model, batch_size, generated_num, output_file, device):
    
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample_out().to(device).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as f:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            f.write('{}\n'.format(string))
            
def t_g_oneepoch(dataload,model,vocab_size,criterion,optimizer):
    
    all_loss = 0
    all_wordnums = 0
    for (data,target) in dataload:
        data = data.clone().detach().to(device)
        print(data.shape)
        target = target.clone().detach()
        target = target.contiguous().view(-1).to(device)
        proba = model.forward(data)[0]
        proba = proba.contiguous().view(-1,vocab_size)
        loss = criterion(proba,target)
        all_loss += loss.item()
        all_wordnums += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    dataload.reset()
    return  math.exp(all_loss / all_wordnums)

def t_d_oneepoch(dataload,num_class,model,vocab_size,criterion,optimizer):
        
    all_loss = 0
    all_wordnums = 0
    for (data,target) in dataload:
        data = data.clone().detach().to(device)
        target = target.clone().detach()
        target = target.contiguous().view(-1).to(device)
        proba = model.forward(data)
        proba = proba.contiguous().view(-1,num_class)
        loss = criterion(proba,target)
        all_loss += loss.item()
        all_wordnums += data.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    dataload.reset()
    out = all_loss / all_wordnums
    return  out
    
class ReL(nn.Module):
    
    def __init__(self):
        super(ReL, self).__init__()
        
    def reward_loss(self,proba,reward,vocab_size,target,device):

        proba = proba.view(-1,vocab_size).to(device)
        length = target.size(0)
        hight = proba.size(1)
        one_hot = torch.zeros((length,hight), device=device)
        target = target.view(-1,1)
        one_hot.scatter_(1, target, 1)
        one_hot = one_hot.to(torch.bool).to(device)
        loss = torch.masked_select(proba, one_hot).to(device)
        loss = loss * reward
        
        return -torch.sum(loss)


##main

if __name__ == '__main__':

    random.seed(42)
    np.random.seed(42)
    
    max_len = 35
    emb_dim = 100
    hidden_dim = 128
    vocab_size = 21
    batch_size = 32
    temperature = 1.0
    embedding_size = 100
    num_class = 2
    lr = 0.001
    betas = (0.9,0.999)
    real_data_file = r'C:\Users\CD\Desktop\real_data.txt'
    generated_num = 228
    fake_data_file = r'C:/wsl_data/iFeature/data/fake_data.txt'
    pre_g_num = 250
    pre_d_num = 150
    bias = -2.0
    h_num = 2
    batch_epochs = 10000
    dp_epochs = 1
    update_rate = 0.8
    MC_num = 16
    out_channels = [32,64,128]
    con_kernel_sizes = [3,3,3]
    pl_kernel_sizes = [2,2,2]
    pd_sizes = [1,1,1]
    l_in = 128 * 1
    drop_rate = 0.5
    features_num = 9
    dis_csv_path = r'C:/wsl_data/iFeature/data/dis_loss_single.csv'
    final_generated_num = 2000
    beishu = 6.0
    f_model = feature(features_num=features_num)

    ger = gernarate(max_len, emb_dim, hidden_dim, vocab_size, batch_size, temperature, embedding_size, device).to(device)
    dis = Discriminator(out_channels, con_kernel_sizes, pl_kernel_sizes, pd_sizes, l_in, drop_rate, f_model, device, batch_size, max_len).to(device)
    g_dataload = GenDataIter(real_data_file, batch_size)
    feadef = feature_loss(features_num)
    
    g_loss = nn.NLLLoss(reduction='sum').to(device)
    g_optimizer = optim.Adam(ger.parameters(),lr=lr,betas=betas)
    
    pre_g_epochs = []
    pre_g_loss = []
    pre_d_epochs = []
    pre_d_loss = []
        
    # Adversarial training 
    
    print('Adversarial training is enabled')
    
    g_rloss_adam = optim.Adam(ger.parameters(),lr=lr,betas=betas)
    rloss = ReL()     
    ro = rollout(model=ger, up_date=update_rate, seq_lens=max_len,device=device,features_num=features_num)
    adver_loss = nn.NLLLoss(reduction='sum').to(device)
    d_rloss_adam = optim.Adam(dis.parameters(),lr=lr,betas=betas)
    
    for bp in range(batch_epochs):
        # print('---')
        samples = ger.sample_out().to(device)
        inputs = torch.cat([torch.ones(batch_size, 1, dtype=torch.long).to(device), samples], dim=1)[:, :-1].to(device).contiguous()
        targets = samples.view(-1).to(device).contiguous()
        rewards = torch.tensor(ro.rewards(N=MC_num, x=samples, discriminator=dis)).reshape(-1).contiguous()
        rewards = torch.exp(rewards).to(device)
        # print(rewards)
        # print('------')
        proba = ger.forward(inputs)[0]
        # print(proba[0,4,:])
        g_loss = rloss.reward_loss(proba=proba, reward=rewards, vocab_size=vocab_size, target=targets, device=device)
        # print('g_loss----',g_loss)
        g_rloss_adam.zero_grad()
        g_loss.backward()
        g_rloss_adam.step()
        
        
        generate_samples(model=ger, batch_size=batch_size, generated_num=generated_num, output_file=fake_data_file, device=device)
        dis_load = DisDataIter(real_data_file, fake_data_file, batch_size)
        # ro.update_params()
        
        ###gen_model save
        
        gen_save_path = f"D:/ML1/gen_model_dict/gen_model_epoch_{bp + 1}.pth"
        torch.save(ger.state_dict(), gen_save_path)
        
        for dp in range(dp_epochs):

            dis_train = t_d_oneepoch(dataload=dis_load, num_class=num_class, model=dis, vocab_size=vocab_size, criterion=adver_loss, optimizer=d_rloss_adam)
            d_loss = dis_train
            current_data = pd.DataFrame({'Epoch': [(dp + 1) + (bp * dp_epochs)], 'D_loss': [d_loss]})

            if (dp + 1) + (bp * dp_epochs) == 1:
                current_data.to_csv(dis_csv_path, mode='w', header=True, index=False)
            else:
                current_data.to_csv(dis_csv_path, mode='a', header=False, index=False)
            
            ###dis_model_save
            dis_save_path = f"D:/ML1/dis_model_dict/dis_model_epoch_{bp * dp_epochs + (dp + 1)}.pth"
            torch.save(dis.state_dict(), dis_save_path)
            
        fake_data_file_plt = f"D:/ML1/final_samples_out/final_samples_fake_epoch{bp + 1}.txt"
        generate_samples(model=ger, batch_size=batch_size, generated_num=final_generated_num, output_file=fake_data_file_plt, device=device)
        feadef.f_r_ht(real_data_file, fake_data_file_plt, bp, g_loss, d_loss)
        
        print(f'Training Progress |^_^| Gen Epoch={bp + 1}/{batch_epochs} |^_^| Gen Loss={g_loss:.4f} |^_^| Dis Epoch={bp * dp_epochs + (dp + 1)}/{batch_epochs * dp_epochs} |^_^| Dis Loss={d_loss:.4f}')
