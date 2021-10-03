import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import sys
from netCDF4 import Dataset,num2date
from prueb_nombr1 import *
from os import listdir
from os.path import isfile, join
#from datetime import datetime, date, time, timedelta
#from pylab import *

def rain(path1,path2,name):
    varsp = 'pr'
    tname='Times'

    ncinsp = Dataset('%s%s'%(path1,name), 'r')
    prsp = ncinsp.variables[varsp][:]
    lonssp = ncinsp.variables['lon'][:]
    latssp = ncinsp.variables['lat'][:]
    nctime=ncinsp.variables[tname][:]
    t_unit = ncinsp.variables[tname].units 
    t_cal = ncinsp.variables[tname].calendar

    tvalue = num2date(nctime,units = t_unit,calendar = t_cal)
    ncinsp.close()

    agnos = [i.year for i in tvalue]
    meses = [i.month for i in tvalue]
    dias = [i.day for i in tvalue]

    coord=np.loadtxt('/Users/fredm/Desktop/TFG_DEFINITIVO/6_comps/KMEANS/regiones.txt',dtype=int)
    #print(np.shape(coord))
    n_class=int(len(coord[0,:])/2)
    prsp=prsp[:,:,:]
    prsp=prsp.transpose(0,2,1)
    rain=np.ones((np.size(prsp,axis=0),n_class))

    for i in range(0,n_class):
        y=coord[:,2*i]!=0
        prz1=prsp[:,coord[:,2*i+1][y],coord[:,2*i][y]]
        rain[:,i]=np.sum((prz1),axis=1)/sum(y)

    ARR=np.zeros((len(rain[:,0]),n_class+3))

    #print(rain[-2])
    ARR[:,3:]=rain[:]
    ARR[:,0]=dias[:]
    ARR[:,1]=meses[:]
    ARR[:,2]=agnos[:]
    #print(ARR[0:2,:])
    np.savetxt('%sprecxreg_med_%s.txt'%(path2,ind_nombr(name,0,3,4,5)),ARR,delimiter=' , ',fmt='%.1f')

path1='/Users/fredm/Desktop/TFG_DEFINITIVO/Bases_de_datos/Modelos_Climaticos/Futuro/Lluvia/'
path2='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Modelos_Climaticos/Futuro/Lluvia/'
lluvia = [f for f in listdir(path1) if isfile(join(path1, f))]
[rain(path1,path2,i)for i in lluvia]
#print([ind_nombr(i,0,3,4,5)for i in lluvia])