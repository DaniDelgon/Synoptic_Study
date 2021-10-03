import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import sys
from netCDF4 import Dataset,num2date
from prueb_nombr1 import *
#from datetime import datetime, date, time, timedelta
#from pylab import *

name='SPREAD_1980_2009_pleased03.nc'
varsp = 'pcp'
tname='time'

ncinsp = Dataset('/Users/fredm/Desktop/TFG_DEFINITIVO/Bases_de_datos/Datos_Reales/%s'%(name), 'r')
prsp = ncinsp.variables[varsp][:]
lonssp = ncinsp.variables['lon'][:]
latssp = ncinsp.variables['lat'][:]
nctime=ncinsp.variables['time'][:]
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
print(np.shape(prsp))
rain=np.ones((np.size(prsp,axis=0),n_class))

for i in range(0,n_class):
    y=coord[:,2*i]!=0
    prz1=prsp[:,coord[:,2*i+1][y],coord[:,2*i][y]]
    rain[:,i]=np.sum((prz1),axis=1)/sum(y)

ARR=np.zeros((len(rain[1:,0]),n_class+3))

#print(rain[-2])
ARR[:,3:]=rain[1:]
ARR[:,0]=dias[:-1]
ARR[:,1]=meses[:-1]
ARR[:,2]=agnos[:-1]
#print(ARR[0:2,:])
np.savetxt('/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Datos_Reales/precxreg_med_%s.txt'%(new_name(name)),ARR,delimiter=' , ',fmt='%.1f')