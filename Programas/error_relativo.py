import numpy as np
from prueb_nombr1 import *
from os import listdir
from os.path import isfile, join

path1='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Tablas_de_Calor/Pasado/'
tabla1= [f for f in listdir(path1) if isfile(join(path1, f))]

#ps=np.loadtxt('%s%s'%(path1,tabla1[0]),delimiter=' , ')
ps=[np.loadtxt('%s%s'%(path1,i),delimiter=' , ') for i in tabla1]

path2='/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Tablas_de_Calor/Futuro/'
tabla2= [f for f in listdir(path2) if isfile(join(path2, f))]
#print(tabla2)
pm=[np.loadtxt('%s%s'%(path2,i),delimiter=' , ') for i in tabla2]

for i in range(0,len(tabla1)):
    err_rel=(pm[4*(i+1)-4:4*(i+1)]-ps[i])
    print(len(err_rel))
    print([ind_nombr(1,x,3,4,5,6)for x in tabla2[4*(i+1)-4:4*(i+1)]])
    [np.savetxt('/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Tablas_de_Calor/Futuro/Error_rel/err_rel_%s'%(ind_nombr(1,x[0],3,4,5,6)),x[1],delimiter=' , ',fmt='%.2f') for x in zip(tabla2[4*(i+1)-4:4*(i+1)],err_rel)]

#[np.savetxt('/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Tablas_de_Calor/Futuro/Error_rel/err_rel_%s'%(ind_nombr(x[0],3,4,5,6)),x[1],delimiter=' , ',fmt='%.2f') for x in zip(tabla2,err_rel)]
