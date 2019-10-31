#!/usr/bin/env python

import os
import sys

import pandas as pd
import numpy as np
import scipy.optimize as optim
from BetaDist import BetaDist
import math
import scipy.special as sp
from scipy.stats import beta

ub, lb = 6.0,-3.0

data_directory = '/Users/sns9/Research/IMS_project/LimitingLandscapes/BElacYstrong'
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

x = np.array(xlist)


# data_size = 0
# for filename in file_list:
#     dataframes.append(pd.read_csv(filename))
#     data_size += 1
dataframes[-1].rename(columns={'Unnamed: 0': ''},inplace=True)

data_shape = dataframes[-1].shape
index_set = dataframes[-1].index
columns_set = list(dataframes[-1].columns)
rows_set = list(dataframes[-1][columns_set[0]])

column_range = range(0,len(columns_set))

output_frame = dataframes[-1].copy()
output_pcov = dataframes[-1].copy()

max_MI = 0
MI_var = 0
max_coord = [0,0]

for i in index_set[1:]:
    for jj in column_range[1:]:
        y = []

        for k in range(0,data_size):
            j = dataframes[k].columns[jj]
            y.append(dataframes[k].at[i,j])

        popt, pcov = optim.curve_fit(linear_fit,x,np.array(y))

        if popt[1]>max_MI:
            max_MI = popt[1]
            max_coord = [float(rows_set[i]),float(j)]
            MI_var = pcov[1,1]

        j = output_frame.columns[jj]
        output_frame.at[i,j] = popt[1]
        output_pcov.at[i,j] = pcov[1,1]

#output_transpose = output_frame.T

#output_pcov.to_csv('LimitingPcov.csv',header=False,index=False)

output_frame.T.to_csv('LimitingMI_surface.csv',header=False)
#output_frame.to_csv('LimitingMI_surface.csv',index=False)

f = open('C-coord.txt','w')
print('C = ',max_MI,file=f)
print('C_var = ',MI_var,file=f)
print('Mean = ',max_coord[0],file=f)
print('Std = ',max_coord[1],file=f)

# Creating relative entropy landscape
rel_ent_frame = dataframes[-1].copy()
#C_beta = BetaDist(lb,ub,math.log10(max_coord[0]),max_coord[1])
C_beta = BetaDist(lb,ub,math.log10(max_coord[1]),max_coord[0])

B_C = sp.gamma(C_beta.p_beta)*sp.gamma(C_beta.q_beta)/sp.gamma(C_beta.p_beta + C_beta.q_beta)

for i in index_set[1:]:
    for j in columns_set[1:]:
        #beta_obj = BetaDist(lb,ub,math.log10(float(rows_set[i])),float(j))
        beta_obj = BetaDist(lb,ub,math.log10(float(rows_set[i])),float(j))

        this_B = sp.gamma(beta_obj.p_beta)*sp.gamma(beta_obj.q_beta)/sp.gamma(beta_obj.p_beta + beta_obj.q_beta)
        this_di_p = sp.digamma(beta_obj.p_beta)
        this_di_q = sp.digamma(beta_obj.q_beta)
        this_di_pq = sp.digamma(beta_obj.p_beta + beta_obj.q_beta)

        try:
            v = math.log(B_C) - math.log(this_B) + (beta_obj.p_beta - C_beta.p_beta)*this_di_p + (beta_obj.q_beta - C_beta.q_beta)*this_di_q
            v += (C_beta.p_beta - beta_obj.p_beta + C_beta.q_beta - beta_obj.q_beta)*this_di_pq
            v *= 1.0/math.log(2.0)
        except ValueError:
            print(math.log10(float(rows_set[i])),float(j))
            sys.stdout.flush()
            sys.exit()

        rel_ent_frame.at[i,j] = v

rel_ent_frame.T.to_csv('Hdrop.csv',header=False)
#rel_ent_frame.to_csv('Hdrop.csv',index=False)

MI_matrix = output_frame.to_numpy()[:,1:]
H_matrix = rel_ent_frame.to_numpy()[:,1:]
create_radial_law(MI_matrix,H_matrix)

# Write CC pdf
m, s = math.log10(max_coord[1]), max_coord[0]

s_m = (m-lb)/(ub-lb)
s_v = (s**2)/((ub-lb)**2)

p = s_m*(s_m*(1-s_m)/s_v - 1)
q = (s_m*(1-s_m)/s_v - 1) - p

xx = np.linspace(0.0,1.0,1000)
sx = xx*(ub-lb) + lb
xpf = beta.cdf(xx,p,q,0,1)
c_pdf = beta.pdf(xx,p,q,0,1)/(ub-lb)

ff = open('cc_pdf.csv','w')

for k in range(0,1000):
    print(str(10**sx[k])+','+str(xpf[k]),file=ff)

ff.close()

ff = open('p_c.csv','w')

for k in range(0,1000):
    print(str(10**sx[k])+','+str(c_pdf[k]),file=ff)

ff.close()
