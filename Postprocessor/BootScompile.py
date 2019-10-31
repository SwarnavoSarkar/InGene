#!/usr/bin/env python

import os
import sys

import pandas as pd
import numpy as np
import scipy.optimize as optim
from BetaDist import BetaDist
import math
import scipy.special as sp
import scipy.stats as stats

ub, lb = 0.4,0.0
m_max, m_min = 0.35, 0.05
s_max, s_min = 0.125, 0.01
box_size = 400

data_directory = '/Users/sns9/Research/SaraWalker_collaboration/Master/5_BT'
n_samples = list(range(1,1001))

os.chdir(data_directory)

dataframes = []

coord_dict = {}
coord_dict['Mean'] = []
coord_dict['Std'] = []

data_size = 0

sum_m, sum_s = 0.0, 0.0

C_values = []

for n in n_samples:
    #filename = 'trial_'+str(n)+'_MI_landscape.csv'
    filename = 'trial_'+str(n)+'_C.csv'
    dataframes.append(pd.read_csv(filename))
    dataframes[-1].rename(columns={'Unnamed: 0': ''},inplace=True)

    sum_m += dataframes[-1]['Mean'][0]
    sum_s += dataframes[-1]['Std'][0]

    coord_dict['Mean'].append(dataframes[-1]['Mean'])
    coord_dict['Std'].append(dataframes[-1]['Std'])

    #dataframes[-1]['Mean'] = 10**dataframes[-1]['Mean']
    data_size += 1

    C_values.append(dataframes[-1]['C'][0])

cf = open('c_average.txt','w')
print(np.mean(np.array(C_values)),(sum_m/data_size),sum_s/data_size,file=cf)
cf.close()

m_c = sum_m/data_size
s_c = sum_m/data_size

of = open('C_summary.csv','w')
ks = list(dataframes[-1])

print(ks)

out_string = ks[0]+','+ks[1]+','+ks[2]
print(out_string,file=of)

for n in range(0,data_size):
    outstring = ''
    for k in ks:
        outstring += str(dataframes[n].at[0,k])+','

    outstring = outstring.rstrip(',')

    print(outstring,file=of)

of.close()

# Compute Kernel Density Estimate
m_array = np.transpose(np.array(coord_dict['Mean']))
s_array = np.transpose(np.array(coord_dict['Std']))

xmin = m_array.min()
xmax = m_array.max()
ymin = s_array.min()
ymax = s_array.max()

#X, Y = np.mgrid[xmin:xmax:50j,ymin:ymax:50j]
X, Y = np.mgrid[m_min:m_max:400j,s_min:s_max:400j]
positions = np.vstack([X.ravel(),Y.ravel()])
values = np.vstack([m_array,s_array])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,Y.shape)

Z_factor = sum(sum(Z))#*(m_max-m_min)*(s_max-s_min)/(float(box_size**2))
Z = Z/Z_factor
print(sum(sum(Z)))

c_max, c_min = max(C_values), min(C_values)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
#CS = plt.contour(X,Y,Z)
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent=[m_min, m_max, s_min, s_max])
#ax.plot(m1, m2, 'k.', markersize=2)
#ax.clabel(CS,inline=1)
ax.set_xlim([m_min, m_max])
ax.set_ylim([s_min, s_max])
#ax.set_xlim([-1.0, 3.0])
#ax.set_ylim([0.5, 3.5])
#plt.show()

n = 1000
t = np.linspace(0, Z.max(), n)
integral = ((Z >= t[:, None, None]) * Z).sum(axis=(1,2))
#print(t)

from scipy import interpolate
f = interpolate.interp1d(integral, t)
t_contours = f(np.array([0.9]))
#plt.imshow(Z.T, origin='lower', extent=[m_min,m_max,s_min,s_max], cmap="gray")
plt.contour(Z.T, t_contours, extent=[m_min,m_max,s_min,s_max])
plt.show()

#print(t_contours)

of = open('contour.txt','w')
print('PDF level for a 90% contour: '+str(t_contours[0]),file=of)
of.close()

m_range = np.linspace(m_min,m_max,box_size)
s_range = np.linspace(s_min,s_max,box_size)

head_string = ''

for m in m_range:
    head_string += ','+str(m)

of = open('C_probs.csv','w')
print(head_string,file=of)

for i in range(0,len(s_range)):
    out_string = str(s_range[i])
    for j in range(0,len(m_range)):
        out_string += ','+str(Z[j,i])

    print(out_string,file=of)

of.close()

c_pdf_file = open('cpdf.csv','w')

c_x = np.linspace(0.0,0.4,1000)

beta_p = BetaDist(lb,ub,0.135,0.096)
c_pdf = stats.beta.pdf(c_x,beta_p.p_beta,beta_p.q_beta,0,scale=0.4)
c_cdf = stats.beta.cdf(c_x,beta_p.p_beta,beta_p.q_beta,0,scale=0.4)

for k in range(0,len(list(c_x))):
    print(str(c_x[k])+','+str(c_pdf[k])+','+str(c_cdf[k]),file=c_pdf_file)

c_pdf_file.close()
