import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

precxreg=np.loadtxt('/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Datos_Reales/precxreg_med_SPREAD_1980_2009.txt',delimiter=' , ')
coord=np.loadtxt('/Users/fredm/Desktop/TFG_DEFINITIVO/6_comps/KMEANS/regiones.txt',delimiter=' ')
print(np.shape(precxreg))
n_comps=6
ag=np.unique(precxreg[:,2])
rain=np.zeros((len(ag)*12,n_comps))
j=0

for k in range(0,len(ag)):
    for i in range(0,12):
        ind=np.where((precxreg[:,2]==ag[k])&(precxreg[:,1]==i+1))
        rain[j,:]=precxreg[np.min(ind):np.max(ind),3:].mean(axis=0)#/sum(coord[:,2*j]!=0)
        j=j+1


fig, axes = plt.subplots(nrows=3, ncols=2)
for l in range(0,n_comps):
    plt.subplot(3,2,l+1)
    plt.plot(rain[:,l],label='Region %d'%(l+1))
    ticks=np.linspace(0,360,3)
    labels=['1980','1995','2009']
    plt.title('Region %d'%(l+1))
    plt.xticks(ticks, labels)
    plt.axis([0,360,0,12])
    plt.grid()

fig.tight_layout()
plt.savefig('/Users/fredm/Desktop/TFG_DEFINITIVO/Resultados/Otros/Precipitacion_anual.png')
plt.show()