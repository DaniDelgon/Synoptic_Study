import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from prueb_nombr1 import *
from os import listdir
from os.path import isfile, join

def descrip(l):
    A=np.zeros((4,8))
    return A

#path='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Datos_Reales/'
#name1='tipos_tiempo_mod_era5_1980_2009.txt'
#name2='precxreg_tot_SPREAD_1980_2009.txt'


def merger(name1,name2,path1,path2,path3):
    n_comps=6
    type=np.loadtxt('%s%s'%(path1,name1),delimiter=' , ')
    precxreg=np.loadtxt('%s%s'%(path2,name2),delimiter=' , ')
    prectot1=np.sum((precxreg[:,3:][type[:,3]!=4]),axis=0)
    prectot2=np.sum((precxreg[:,3:][type[:,3]==0]),axis=0)+np.sum((precxreg[:,3:][type[:,3]==1]),axis=0)

    v=np.array([0,1,2,3,4,5,6,7])
    tiemp=np.array([0,1,2,3])
    G=np.zeros((6,11))
    #print(sum(np.sum((precxreg[:,3:][type[:,3]==4]),axis=0)/sum(prectot1)))
    for i in range(0,n_comps):
        for j in range(0,len(tiemp)):
            a=type[:,3]==j
            G[i,7+j]=round((sum(precxreg[:,i+3][a])/prectot1[i])*100,2)
        G[i,8]=G[i,7]+G[i,8]
        for k in range(0,len(v)):
            b=((type[:,3]==0)&(type[:,5]==k))*1
            c=((type[:,3]==1)&(type[:,5]==k))*1
            d=b+c==1
            G[i,k]=round((sum(precxreg[:,i+3][d])/prectot2[i])*100,2)
    np.savetxt('%sTabla_de_Calor_%s.txt'%(path3,ind_nombr(name2,2,3,4,5)),G,delimiter=' , ',fmt='%.2f')

path3='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Tablas_de_Calor/Futuro/'
path1='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Modelos_Climaticos/Futuro/Presion_superficial/'
path2='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Modelos_Climaticos/Futuro/Lluvia/'
type= [f for f in listdir(path1) if isfile(join(path1, f))]
rain= [f for f in listdir(path2) if isfile(join(path2, f))]
rain=rain[12:]

print(ind_nombr(rain[0],2,3,4,5))
#[merger(x[0],x[1],path1,path2,path3) for x in zip(type,rain)]