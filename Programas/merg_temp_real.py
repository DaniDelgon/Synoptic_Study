import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from prueb_nombr1 import *

path='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Datos_Reales/'
name1='tipos_tiempo_mod_era5_1980_2009.txt'
name2='precxreg_med_SPREAD_1980_2009.txt'
type=np.loadtxt('%s%s'%(path,name1),delimiter=' , ')
precxreg=np.loadtxt('%s%s'%(path,name2),delimiter=' , ')

n_comps=6

prectot1=np.sum((type[:,3]!=4)*1,axis=0)
prectot2=np.sum((type[:,3]==0)*1,axis=0)+np.sum((type[:,3]==1)*1,axis=0)
threshold=0.5

v=np.array([0,1,2,3,4,5,6,7])
tiemp=np.array([0,1,2,3])
G=np.zeros((6,11))
#print(sum(type[:,3]==4))
for i in range(0,n_comps):
    for j in range(0,len(tiemp)):
        a=type[:,3]==j
        #print(sum(a*1),'a')
        a1=precxreg[:,i+3][a]>=threshold
        #print(sum(a1*1),'a1')
        G[i,7+j]=round((sum(a1*1)/prectot1)*365.,2)
    G[i,8]=G[i,7]+G[i,8]
    for k in range(0,len(v)):
        b=((type[:,3]==0)&(type[:,5]==k))*1
        c=((type[:,3]==1)&(type[:,5]==k))*1
        d=b+c==1
        a2=precxreg[:,i+3][d]>=threshold
        G[i,k]=round((sum(a2*1)/prectot1)*365.,2)

np.savetxt('/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Tablas_de_Calor/Datos_reales/Tablas_Temp/Tabla_Temp_de_Calor_%smm_%s.txt'%(str(threshold),ind_nombr2(name2,2,3,4)),G,delimiter=' , ',fmt='%.2f')
#print(ind_nombr2(name2,2,3,4))
#print(str(threshold))