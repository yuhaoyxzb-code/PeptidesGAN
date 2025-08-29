# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 02:56:19 2025

@author: CD
"""
import torch.nn as nn
import torch
import numpy as np
import copy
import torch.optim as optim
import pandas as pd

####data read

data_file = r'C:\Users\CD\Desktop\实验数据\实验数据\实验数据\2025-3-28\pep_fix_real.csv'
data = np.array(pd.read_csv(data_file))
allpep = []
for i in range(data.shape[0]):
    a = data[i][0]
    allpep.append(a)
print(allpep)

####worddix

vaule_num = []
for i in range(21):
    vaule_num.append(str(i))
all_amino_acids = ['0','A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
wididx = dict(zip(all_amino_acids,vaule_num))
print(wididx)

#####pad0

def pad_sequence():
    pep_list = []
    max_len = 35
    for i in range(len(allpep)):
        padsequence = '0'
        if len(allpep[i]) >= max_len:
            pep = allpep[i][:max_len]
        elif len(allpep[i]) < max_len:
            pep = allpep[i] + padsequence * (max_len - len(allpep[i]))
        pep_list.append(pep)    
    return pep_list
data_peptides = pad_sequence()
print(data_peptides)

a = set(data_peptides)
a = list(a)

# for i in range(len(a)):
#     wordvector = [int(wididx[f'{item}']) for item in a[i]]
#     print(wordvector)
#     print(i)
# print(a[107])
####data to list

wordvector_list = []
for i in range(len(a)):
    wordvector = [int(wididx[f'{item}']) for item in a[i]]
    wordvector_list.append(wordvector)
print(wordvector_list)
# word_np = np.array(wordvector_list)
# word_np = torch.tensor(word_np)

###write realdata

output_path = r'C:\Users\CD\Desktop\实验数据\实验数据\实验数据\2025-3-28\real_data.txt'  # 输出文件路径
with open(output_path, 'w') as f:
    for seq in wordvector_list:
        a = ' '.join(str(i) for i in seq)
        f.write('{}\n'.format(a))  # 每个肽序列后加一个换行符
print(f"肽序列已保存到 {output_path}")

