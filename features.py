# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:31:48 2025

@author: CD
"""

from Bio.SeqUtils import molecular_weight
import numpy as np
import torch
import os
from modlamp.descriptors import *
import torch
from sklearn.preprocessing import MinMaxScaler
from PyBioMed.PyProtein import AAComposition
from PyBioMed.PyProtein import CTD
from sklearn.preprocessing import StandardScaler
# Define feature extraction class
# random.seed(42)
# np.random.seed(42) 

class feature:
    def __init__(self,features_num):
        
        self.wordvector = {'0': '0', '1': 'A', '2': 'R', '3': 'N', '4': 'D', '5': 'C', '6': 'Q', '7': 'E',
                      '8': 'G', '9': 'H', '10': 'I', '11': 'L', '12': 'K', '13': 'M', '14': 'F',
                      '15': 'P', '16': 'S', '17': 'T', '18': 'W', '19': 'Y', '20': 'V'}
        
        self.features_num  = features_num
        
    def cal(self, x, batch_size, max_len):
    
        x = x.cpu()
        x = np.array(x)
        wordvector_list = []
        p = []
        
        for i in range(x.shape[0]):
            A = [item for item in x[i,:] ]
            A = [item for item in A if item != 0]
            p.append(A)
        for i in range(len(p)):
            wordvector_re = [self.wordvector[str(item)] for item in p[i]]
            wordvector_re = ''.join(wordvector_re)
            wordvector_list.append(wordvector_re)
        feature_np = np.zeros((batch_size,self.features_num))
        for i in range(len(wordvector_list)):
            if len(wordvector_list[i]) == 1:
                wordvector_list[i] = ''
        wordvector_list_index_nozeor = [index for index,item in enumerate(wordvector_list) if item != '']

        wordvector_list = [item for item in wordvector_list if item != '']

        #####nine_features
        
        desc=GlobalDescriptor(wordvector_list)
        desc.calculate_all()
        a = desc.descriptor
        nine_features = np.concatenate((a[:,:-2],a[:,-1].reshape((-1,1))),axis=1)
        

        
        all_features = nine_features
        for i in range(all_features.shape[0]):
            feature_np[wordvector_list_index_nozeor[i],:] = all_features[i,:] 
        sc = StandardScaler()
        all_features = sc.fit_transform(feature_np)

        return torch.tensor(all_features) ####


