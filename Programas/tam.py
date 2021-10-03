from os import listdir
from os.path import isfile, join
import numpy as np
from netCDF4 import Dataset,num2date

path1='/Users/fredm/Desktop/TFG_DEFINITIVO/Bases_de_datos/Modelos_Climaticos/Futuro/Lluvia/'
path2='/Users/fredm/Desktop/TFG_DEFINITIVO/Bases_de_datos/Modelos_Climaticos/Futuro/Presion_superficial/'
path3='/Users/fredm/Desktop/TFG_DEFINITIVO/Bases_de_datos/Modelos_Climaticos/Pasado/Lluvia/'
path4='/Users/fredm/Desktop/TFG_DEFINITIVO/Bases_de_datos/Modelos_Climaticos/Pasado/Presion_superficial/'

lluvia_fut=[f for f in listdir(path1) if isfile(join(path1, f))]
presion_fut=[f for f in listdir(path2) if isfile(join(path2, f))]
lluvia_past=[f for f in listdir(path3) if isfile(join(path3, f))]
presion_past=[f for f in listdir(path4) if isfile(join(path4, f))]

lluvia_tot=lluvia_past+lluvia_fut
presion_tot=presion_past+presion_fut

#print(lluvia_tot[3])

ncinsp = Dataset('%s%s'%(path1,lluvia_tot[3]), 'r')
prsp = ncinsp.variables['pr'][:]
prsp=prsp[:,:,:]
print(lluvia_tot[3], 'LLUVIA')
print(np.shape(prsp))

'''
for i in range(0,len(lluvia_tot)):
    if i<=2:
        ncinsp = Dataset('%s%s'%(path3,lluvia_tot[i]), 'r')
        prsp = ncinsp.variables['pr'][:]
        prsp=prsp[:,:,:]
        print(lluvia_tot[i], 'LLUVIA')
        print(np.shape(prsp))
        ncin=Dataset('%s%s'%(path4,presion_tot[i]),'r')
        slp=ncin.variables['SLP'][:]
        slp=slp[:,:,:]
        print(presion_tot[i], 'PRESION')
        print(np.shape(slp))
    if i>>2:
        ncinsp = Dataset('%s%s'%(path1,lluvia_tot[i]), 'r')
        prsp = ncinsp.variables['pr'][:]
        prsp=prsp[:,:,:]
        print(lluvia_tot[i], 'LLUVIA')
        print(np.shape(prsp))
        ncin=Dataset('%s%s'%(path2,presion_tot[i]),'r')
        slp=ncin.variables['SLP'][:]
        slp=slp[:,:,:]
        print(presion_tot[i], 'PRESION')
        print(np.shape(slp))
'''
