#!/usr/bin/env python

import os
import sys

import pandas as pd
import numpy as np
import scipy.optimize as optim
from BetaDist import BetaDist
import math
import scipy.special as sp

ub, lb = 6.0,-3.0

data_directory = '/Users/sns9/Research/IMS_project/LimitingLandscapes/BElacXstrong'
data_fractions = [1,2,5,10,20]
n_samples = list(range(1,6))

def create_radial_law(MI_matrix,H_matrix):
    max_H = math.log10(np.max(H_matrix)*1.05)
    min_H = -2#math.log10(np.min(H_matrix)*0.95)
    bin_size = 10#int(self.resolution/2)
    H_bins = np.linspace(min_H,max_H,bin_size+1)
    d_H = H_bins[1]-H_bins[0]
    MI_samples = {}

    for k in range(0,bin_size):
        MI_samples[k] = []

    rs, cs = MI_matrix.shape[0],MI_matrix.shape[1]

    for i in range(0,rs):
        for j in range(0,cs):
            H = H_matrix[i,j]
            MI = MI_matrix[i,j]

            if H>0.0:
                bin_loc = max(int((math.log10(H)-min_H)/d_H),0)
                MI_samples[bin_loc].append(MI_matrix[i,j])

    f = open('MI_rate.csv','w')
    print('H,I,+,-',file=f)
    f.close()

    for k in range(0,bin_size):
        if len(MI_samples[k])>0:
            outstring = str(10**(0.5*(H_bins[k]+H_bins[k+1])))
            this_mean = np.mean(MI_samples[k])
            outstring += ','+str(this_mean)
            outstring += ','+str(max(MI_samples[k])-this_mean)
            outstring += ','+str(this_mean-min(MI_samples[k]))

            print(outstring,file=open('MI_rate.csv','a'))
        else:
            print('0,0,0,0',file=open('MI_rate.csv','a'))

def linear_fit(x,a,b):
    return a*x+b

os.chdir(data_directory)

trial_frame = pd.read_csv('trial_MI_landscapeT.csv')

dataframes = []
xlist = []

data_size = 0

for df in data_fractions:
    if df>=1:
        df_str = str(int(df))
    else:
        df_s = str(df).split('.')
        df_str = df_s[0]+'p'+df_s[1]

    for n in n_samples:
        xlist.append(float(df))
        filename = 'trial'+df_str+'_'+str(n)+'_MI_landscape.csv'
        dataframes.append(pd.read_csv(filename))
        data_size += 1
        dataframes[-1]['Unnamed: 0'] = trial_frame['Unnamed: 0']
        dataframes[-1].rename(columns={'Unnamed: 0': ''},inplace=True)

        dataframes[-1].to_csv(filename,index=False)
