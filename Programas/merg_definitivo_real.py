import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from prueb_nombr1 import *

def descrip(l):
    A=np.zeros((4,8))
    return A

'''
def heatmap(su):
    su3=np.array(['S','SW','W','NW','N','NE','E','SE','Hibrido (0)/FlujoLineal (1)','Ciclon (2)','Anticiclon (3)'])
    su4=np.array(['Region 1','Region 2','Region 3','Region 4','Region 5','Region 6'])
    fig, ax = plt.subplots()
    im = ax.imshow(su[0],cmap="coolwarm_r")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    cbar = ax.figure.colorbar(im, cax=cax, cmap="YlGn")
    cbar.ax.set_ylabel('suuuu', rotation=-90, va="bottom")
    # hide axes
    ax.set_xticks(np.arange(len(su3)))
    ax.set_yticks(np.arange(len(su4)))
    ax.set_xticklabels(su3)
    ax.set_yticklabels(su4)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    
    for k in range(len(su4)):
        for l in range(len(su3)):
            text = ax.text(l, k,'{:.2f}'.format(su[0][k, l]),ha="center", va="center", color="w")
    ax.set_title('Error Relativo %s'%(new_name(su[1]).split('_')[0]))
    fig.tight_layout()
    plt.savefig('/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Datos_reales/Error_Relativo_%s_Analysis.png'%new_name(su[1]))
    plt.clf()
'''
'''
def merger(names):
    rain=clas_rain(names[0],names[2])    #presion
    precxreg=counter_rain(names[1],names[4],names[3])   #lluvia
    n_comps=6
    
    prectot=np.sum((precxreg[:,3:]),axis=0)

    v=np.array([0,1,2,3,4,5,6,7])
    tiemp=np.array([0,1,2,3])

    G=np.zeros((6,11))

    for i in range(0,n_comps):
        D=descrip(i+1)
        for j in range(0,len(tiemp)):
            for k in range(0,len(v)):
                a=(rain[:,3]==j)*1
                b=(rain[:,5]==k)*1
                c=(a+b)==2
                D[j,k]=round((sum(precxreg[:,i+3][c])/prectot[i])*100,2)
        G[i,0:8]=D.sum(axis=0)
        #print(D[0,:]+D[1,:])
        G[i,8]=(D[0,:]+D[1,:]).sum(axis=0)
        G[i,9:]=D[2:,:].sum(axis=1)
    #G=np.around(G,decimals=1)
    #heatmap(G,names[1])
    return G
'''

path='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Datos_Reales/'
name1='tipos_tiempo_mod_era5_1980_2009.txt'
name2='precxreg_tot_SPREAD_1980_2009.txt'
type=np.loadtxt('%s%s'%(path,name1),delimiter=' , ')
precxreg=np.loadtxt('%s%s'%(path,name2),delimiter=' , ')

n_comps=6

prectot1=np.sum((precxreg[:,3:][type[:,3]!=4]),axis=0)
prectot2=np.sum((precxreg[:,3:][type[:,3]==0]),axis=0)+np.sum((precxreg[:,3:][type[:,3]==1]),axis=0)

v=np.array([0,1,2,3,4,5,6,7])
tiemp=np.array([0,1,2,3])
G=np.zeros((6,11))
print(sum(type[:,3]==4))
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
np.savetxt('/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Tablas_de_Calor/Datos_reales/tipos_tiempo_mod_%s.txt'%(ind_nombr2(name2,2,3,4)),G,delimiter=' , ',fmt='%.2f')
print(ind_nombr2(name2,2,3,4))
#print(G)
'''
a=list(zip(presion,lluvia,pres,tiemp,lluv))
new=[merger(i)for i in a]
#print(new[0])
#heatmap((new[0],lluvia[0]))
new1=[(new[1]-new[0])/new[0],(new[2]-new[0])/new[0],(new[3]-new[0])/new[0]]
b=list(zip(new1,lluvia[1:4]))
#print(b)ax = sns.heatmap(df, vmin=-90, vmax=-50, cmap="viridis")
new2=[heatmap(i) for i in b]
'''