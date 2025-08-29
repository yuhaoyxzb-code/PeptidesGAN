# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:32:22 2025

@author: CD
"""
import torch.nn as nn
import torch
import random
import numpy as np
###
# random.seed(42)
# np.random.seed(42) 

class gernarate(nn.Module):
    
    def __init__(self,max_len,emb_dim,hidden_dim,vocab_size,batch_size,temperature,
                 embedding_size,device):
        super (gernarate, self).__init__()
        
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.embedding_size = embedding_size
        self.u_device = device
        
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        
        self.init_params()

        
    def init_params(self):
        
        for param in self.parameters():
            param.data.uniform_(-0.05,0.05)
        

    def forward(self, x, hidden=None):
        
        # self.lstm.flatten_parameters()
        
        # ##cell state
        if hidden is None:
            h = torch.zeros(1,x.size(0),self.hidden_dim).to(self.u_device)
            c = torch.zeros(1,x.size(0),self.hidden_dim).to(self.u_device)
            hidden = (h,c)
        x = x.to(self.u_device)
        x = self.emb(x)
        out_lstm,new_hidden = self.lstm(x,hidden)  # LSTM
        out = self.fc(out_lstm)
        
        out = out / self.temperature
        log_out = self.logsoftmax(out)# linear 
    
        return log_out, new_hidden
         
    def sample_out(self,x=None): 
        
        allsamples = []
        samples_temp = []
        
        hidden = None
        
        if x is None:
            
            samples = torch.zeros(self.batch_size,self.max_len,dtype=torch.long,device=self.u_device)
            token_first = torch.ones(self.batch_size,1,dtype=torch.long)
            
            for i in range(self.max_len):
                num_out,hidden = self.forward(token_first,hidden)
                # num_out /= self.temperature
                num_out = num_out.squeeze(1)
                num_out = torch.exp(num_out)
                # num_softmax = torch.softmax(num_out,dim=-1)
                cy = torch.multinomial(num_out, 1).squeeze(1)
                samples[:,i] = cy
                token_first = samples[:,i].unsqueeze(1)


            allsamples.append(samples)
            
        else:
            
            for i in range(x.size(1)):
                temp_list = x.chunk(x.size(1),dim=1)
                samples_temp.append(temp_list[i])

            cy = temp_list[-1]
            for i in range(self.max_len-x.size(1)):
                token_first = cy
                num_out,hidden = self.forward(token_first,hidden)
                # num_out /= self.temperature
                num_out = num_out.squeeze(1)
                num_out = torch.exp(num_out)
                cy = torch.multinomial(num_out, 1)
                samples_temp.append(cy)
            samples_temp = torch.cat(samples_temp,dim=1)

            allsamples.append(samples_temp)
        return torch.cat(allsamples,dim=0)


