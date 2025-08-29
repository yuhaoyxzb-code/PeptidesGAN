# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 01:16:26 2025

@author: CD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from PyBioMed.PyProtein import AAComposition
from PyBioMed.PyProtein import CTD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from modlamp.descriptors import *
import torch
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

class feature_loss:
    def __init__(self,features_num):
        
        self.wordvector = {'0': '0', '1': 'A', '2': 'R', '3': 'N', '4': 'D', '5': 'C', '6': 'Q', '7': 'E',
                      '8': 'G', '9': 'H', '10': 'I', '11': 'L', '12': 'K', '13': 'M', '14': 'F',
                      '15': 'P', '16': 'S', '17': 'T', '18': 'W', '19': 'Y', '20': 'V'}
        
        self.features_num  = features_num
        
    def calculate_percentage(self,sequence, category_dict):

        seq_counter = Counter(sequence)
        total_amino_acids = len(sequence)

        category_percentages = []
        for category, amino_acids in category_dict.items():
            count = sum(seq_counter[aa] for aa in amino_acids)
            percentage = (count / total_amino_acids) if total_amino_acids > 0 else 0
            category_percentages.append(percentage)

        return category_percentages
    
    def calculate_all_sequences(self,sequences, category_dict):

        feature_matrix = []
        for seq in sequences:
            percentages = self.calculate_percentage(seq, category_dict)
            feature_matrix.append(percentages)
        
        return np.array(feature_matrix)
    
    def fea_cal(self,file_path):
    
        peptides = []
        
        with open(file_path,'r') as f:
            for line in f:
                peptide = line.strip().split()
                peptide = [int(item) for item in peptide]
                peptides.append(peptide)
        x = np.array(peptides)
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
        for i in range(len(wordvector_list)):
            if len(wordvector_list[i]) == 1:
                wordvector_list[i] = ''
        wordvector_list = [item for item in wordvector_list if item != '']
        #####ten_features
        
        desc=GlobalDescriptor(wordvector_list)
        desc.calculate_all()
        a = desc.descriptor
        b = desc.featurenames
        nine_features = np.concatenate((a[:,:-2],a[:,-1].reshape((-1,1))),axis=1)
        
        all_features = nine_features
        m  = MinMaxScaler()
        all_features = m.fit_transform(all_features)
        all_features_a = all_features
        
        return all_features_a, all_features, wordvector_list
    
    def f_r_ht(self,real_filepath,fake_filepath,bp,g_loss,d_loss):
        
        g_loss = g_loss.cpu().item()
        # d_loss = d_loss.cpu().item()
        ###11bar

        real_data = self.fea_cal(real_filepath)[0]
        generated_data = self.fea_cal(fake_filepath)[0]
        
        feature_names = [
            'Length',
            'MW',              
            'Charge',          
            'ChargeDensity',              
            'pI',  
            'InstabilityInd',     
            'Aromaticity',    
            'AliphaticInd',
            'HydrophRatio'
        ]

        data = []

        for feature_idx in range(len(feature_names)): 

            real_values = real_data[:, feature_idx]
            generated_values = generated_data[:, feature_idx]
            
            real_norm = real_values
            generated_norm = generated_values
            
            feature_name = feature_names[feature_idx]
            
            for value in real_norm:
                data.append({
                    "Feature": feature_name,
                    "Group": "Real",
                    "Value": value
                })
                
            for value in generated_norm:
                data.append({
                    "Feature": feature_name,
                    "Group": "Generated",
                    "Value": value
                })

        df = pd.DataFrame(data)

        plt.figure(figsize=(18, 7)) 
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)
        plt.rcParams['font.family'] = 'DejaVu Sans'

        palette = {
            "Real": "#2c7bb6",    
            "Generated": "#d7191c"
        }

        box = sns.boxplot(
            x="Feature",
            y="Value",
            hue="Group",
            data=df,
            palette=palette,
            width=0.7,
            linewidth=1.2,
            showfliers=False,     
            showmeans=True,     
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "5"
            }
        )

        plt.title("Protein Structural Feature Distribution Comparison\n(11 Features, Without Outliers)", 
                 fontsize=16, pad=20, fontweight='bold')
        plt.xlabel("Feature Name", fontsize=13, labelpad=12)
        plt.ylabel("Normalized Value Range", fontsize=13, labelpad=12)
        plt.xticks(
            rotation=35,        
            ha='right', 
            fontsize=11,
            fontstyle='italic'
        )

        plt.axhline(0.5, color='grey', linestyle='--', linewidth=1, alpha=0.6)

        legend = plt.legend(
            title="Data Type",
            title_fontsize=12,
            fontsize=11,
            loc='upper right',
            frameon=True,
            shadow=True,
            edgecolor='black',
            facecolor='#f7f7f7'
        )

        plt.ylim(-0.1, 1.1)

        plt.show()
        
        ####AApinlv
        real_features = self.fea_cal(real_filepath)[2]
        fake_features = self.fea_cal(fake_filepath)[2]

        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

        def calculate_aa_frequencies(sequences, amino_acids):
            data = []
            for seq in sequences:
                seq_count = Counter(seq)
                total_aa = len(seq)
                frequencies = {aa: seq_count.get(aa, 0) / total_aa * 100 for aa in amino_acids}
                data.append(frequencies)
            return data

        generated_frequencies = calculate_aa_frequencies(fake_features, amino_acids)
        real_frequencies = calculate_aa_frequencies(real_features, amino_acids)

        generated_df = pd.DataFrame(generated_frequencies)
        real_df = pd.DataFrame(real_frequencies)

        generated_df['Group'] = 'Generated'
        real_df['Group'] = 'Real'

        df = pd.concat([generated_df, real_df], ignore_index=True)

        df_melted = df.melt(id_vars=["Group"], var_name="Amino Acid", value_name="Frequency")

        palette = {
            "Real": "#3A5F90",    
            "Generated": "#D9472B"
        }

        plt.figure(figsize=(20, 8))
        sns.set_style("whitegrid", {'grid.linestyle': ':'})
        plt.rcParams['font.family'] = 'DejaVu Sans' 

        box = sns.boxplot(
            x="Amino Acid",
            y="Frequency",
            hue="Group",
            data=df_melted,
            palette=palette,
            width=0.7,
            linewidth=1.2,
            showfliers=False,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 6
            }
        )

        plt.title("Amino Acid Frequency Distribution Comparison\n(Real vs Generated)", 
                 fontsize=18, pad=25, fontweight='bold')
        plt.xlabel("Amino Acid", fontsize=14, labelpad=15)
        plt.ylabel("Frequency (%)", fontsize=14, labelpad=15)

        plt.xticks(
            rotation=45,
            ha='right',
            fontsize=12,
            fontstyle='italic'
        )

        plt.yticks(np.arange(0, 101, 10)) 
        plt.ylim(-0.1, 100) 


        box.legend_.remove()

        legend_elements = [
            plt.Line2D([0], [0], 
                       marker='s', 
                       color='w',
                       markerfacecolor=palette["Real"],
                       markersize=12,
                       label='Real Sequences'),
            plt.Line2D([0], [0],
                       marker='s',
                       color='w',
                       markerfacecolor=palette["Generated"],
                       markersize=12,
                       label='Generated Sequences')
        ]

        plt.legend(
            handles=legend_elements,
            title="Data Category",
            title_fontsize=13,
            fontsize=12,
            loc='upper right',
            frameon=True,
            shadow=True,
            edgecolor='black',
            facecolor='#f7f7f7',
            bbox_to_anchor=(1.02, 1)
        )

        for y in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            plt.axhline(y, color='grey', linestyle=':', linewidth=0.8, alpha=0.6)
        plt.tight_layout()

        plt.show()

    ####g_loss,d_loss save
    
        csv_path = f'D:/ML1/epoch_loss/epoch_loss.csv'
        current_data = pd.DataFrame({'Epoch': [bp + 1], 'G_Loss': [g_loss], 'D_loss': [d_loss]})
        
        
        if (bp + 1) == 1:
            current_data.to_csv(csv_path, mode='w', header=True, index=False)
        else:
            current_data.to_csv(csv_path, mode='a', header=False, index=False)

