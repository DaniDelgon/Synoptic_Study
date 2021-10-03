import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from prueb_nombr1 import *
from os import listdir
from os.path import isfile, join

def merg(name1,name2,path1,path2,path3):
    type=np.loadtxt('%s%s'%(path1,name1),delimiter=' , ')
    precxreg=np.loadtxt('%s%s'%(path2,name2),delimiter=' , ')
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
    print(name2)

    np.savetxt('%sTabla_Temp_de_Calor_%smm_%s.txt'%(path3,str(threshold),ind_nombr(name2,2,3,4,5)),G,delimiter=' , ',fmt='%.2f')

path1='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Modelos_Climaticos/Futuro/Presion_superficial/'
path2='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Modelos_Climaticos/Futuro/Lluvia/'
path3='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Tablas_de_Calor/Futuro/Tablas_Temp/'
type= [f for f in listdir(path1) if isfile(join(path1, f))]
rain= [f for f in listdir(path2) if isfile(join(path2, f))]
rain=rain[:12]
#print(rain)
#print(type)

[merg(x[0],x[1],path1,path2,path3) for x in zip(type,rain)]

#type=np.loadtxt('%s%s'%(path1,name1),delimiter=' , ')
#precxreg=np.loadtxt('%s%s'%(path2,name2),delimiter=' , ')
