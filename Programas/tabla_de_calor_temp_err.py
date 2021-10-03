from prueb_nombr1 import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from mpl_toolkits.axes_grid1 import make_axes_locatable

def err_heatmap(path1,path2,name,max):
    array=np.loadtxt('%s%s'%(path1,name),delimiter=' , ',dtype = np.float64)
    fig, ax = plt.subplots()
    array=np.around(array,1)
    im = ax.imshow(array,cmap="coolwarm_r",vmin=-max,vmax=max)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.25)
    #cbar = ax.figure.colorbar(im, cax=cax)
    #cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    sux=['S','SW','W','NW','N','NE','E','SE','Hibrido (0)/FlujoLineal (1)','Ciclon (2)','Anticiclon (3)']
    suy=['Region 1','Region 2','Region 3','Region 4','Region 5','Region 6']

    ax.set_xticks(np.arange(len(sux)))
    ax.set_yticks(np.arange(len(suy)))
    ax.set_xticklabels(sux)
    ax.set_yticklabels(suy)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    for i in range(len(suy)):
        for j in range(len(sux)):
            text = ax.text(j, i, array[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Cambio Temporal en Precipitaciones %s"%(ind_nombr1(1,name,3,4,5,6,7)))
    #print(ind_nombr(0,name,2,3,4,5))
    fig.tight_layout()
    plt.savefig('%sError_relativo_Heatmap_%s.png'%(path2,ind_nombr1(1,name,3,4,5,6,7)))
    plt.clf()

path1='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Tablas_de_Calor/Futuro/Error_rel_Temp/'
path2='/Users/fredm/Desktop/TFG_DEFINITIVO/Resultados/Tablas_de_Calor/Futuro/Error_rel_Temp/'
name= [f for f in listdir(path1) if isfile(join(path1, f))]
array=[np.loadtxt('%s%s'%(path1,i),delimiter=' , ') for i in name]
max=10
#print(np.max(max))
print(name)
[err_heatmap(path1,path2,i,max)for i in name]