import numpy as np
from netCDF4 import Dataset,num2date
from datetime import datetime, date, time, timedelta
import sys
import os
#from windrose import WindroseAxes
#import matplotlib.cm as cm
sys.path.append('C:\\Users\\fredm\\Desktop\\TFG_DEFINITIVO\\Programas')
from prueb_nombr1 import *
import pylab as py

'''
if len(sys.argv)!=2:
	print ("usage: %s <year>" %(sys.argv[0]))
	sys.exit(-1)

y=int(sys.argv[1])
'''

name='era5_slp_1980_2009_llbox.nc' #% y

# Definimos las variables

ncin=Dataset('/Users/fredm/Desktop/TFG_DEFINITIVO/Bases_de_datos/Datos_Reales/%s'%(name),'r')
#print(ncin)
slp=ncin.variables['msl'][:]    #CAMBIO PARA ALIGERAR TIEMPO DE COMPUTACION
lon=ncin.variables['longitude'][:]
lat=ncin.variables['latitude'][:]
time=ncin.variables['time'][:]
t_unit = ncin.variables['time'].units 
t_cal = ncin.variables['time'].calendar

tvalue = num2date(time,units = t_unit,calendar = t_cal)
ncin.close()

print('SL Pressure: ',np.shape(slp))
print('Time:',np.shape(time))
print('Lat: ',np.shape(lat))
print('Lon: ',np.shape(lon))

# Booleanos
"""
a1=lat==20
a2=lat==25
a3=lat==30
a4=lat==35
a5=lat==40
"""
a1=lat==18.5
a2=lat==23.5
a3=lat==28.5
a4=lat==33.5
a5=lat==38.5

blat=a1+a2+a3+a4+a5 # Latitud
b1=lon==-25
b2=lon==-20
b3=lon==-15
b4=lon==-10
blon=b1+b2+b3+b4 # Longitud

# Cogemos los valores que nos interesan

SLP=slp[:,blat,:][:,:,blon]
LAT=lat[blat]
LON=lon[blon]
print()
print('SLP bool: ',np.shape(SLP))

# Definimos las presiones de la rejilla

P1=SLP[:,0,1]
P2=SLP[:,0,2]
P3=SLP[:,1,0]
P4=SLP[:,1,1]
P5=SLP[:,1,2]
P6=SLP[:,1,3]
P7=SLP[:,2,0]
P8=SLP[:,2,1]
P9=SLP[:,2,2]
P10=SLP[:,2,3]
P11=SLP[:,3,0]
P12=SLP[:,3,1]
P13=SLP[:,3,2]
P14=SLP[:,3,3]
P15=SLP[:,4,1]
P16=SLP[:,4,2]

# Definimos las funciones

lam=30.

W=1/2.*(P12+P13)-1/2.*(P4+P5)
S=1/(np.cos(np.radians(lam)))*(1/4.*(P5+2.*P9+P13)-1/4.*(P4+2.*P8+P12))
F=np.sqrt(S**2.+W**2.)

theta=np.arctan2(W,S)

ZW=np.sin(np.radians(lam))/(np.sin(np.radians(lam-5.)))*(1/2.*(P15+P16)-1/2.*(P8+P9))-np.sin(np.radians(lam))/(np.sin(np.radians(lam+5.)))*(1/2.*(P8+P9)-1/2.*(P1+P2))
ZS=1/(2.*(np.cos(np.radians(lam)))**2.)*(1/4.*(P6+2.*P10+P14)-1/4.*(P5+2.*P9+P13)-1/4.*(P4+2.*P8+P12)+1/4*(P3+2.*P7+P11))
Z=ZW+ZS

G=np.sqrt(F**2.+(0.5*Z)**2.) # Se supone que es un indicador de temporales. 1-13 enero de 1997 (preguntar a Albano)

'''
Aplicamos las condiciones para tipo de tiempo:

Tipo 0: Híbridos (F<|Z|<2F) e indeterminados (F<3 & |Z|<3) nkhkuh
Tipo 1: Flujo lineal (|Z|<F)
Tipo 2: Anticiclónico (|Z|>2F & Z<0)
Tipo 3: Ciclónico (|Z|>2F & Z>0)

Dirección del viento:

S  (tipo 0): ß<22.5º & ß>337.5º
SW (tipo 1): 67.5º>=ß>=22.5º
W  (tipo 2): 112.5º>ß>67.5º
NW (tipo 3): 157.5º>=ß>=112.5º
N  (tipo 4): 202.5º>ß>157.5º
NE (tipo 5): 247.5º>=ß>=202.5º
E  (tipo 6): 292.5º>ß>247.5º
SE (tipo 7): 337.5º>=ß>=292.5º

'''

i=np.arange(0,len(F))
cond=np.zeros((len(F),))
direc=np.zeros((len(F),))
v=np.zeros((len(F),))

# Condiciones de la dirección del viento

for m in i:
	direc[m]=np.rad2deg(theta[m])%360
	"""
	if W[m]>=0 and S[m]>=0:
		direc[m]=np.degrees(theta[m])
	if W[m]<0 and S[m]>0:
		direc[m]=np.degrees(theta[m]+2*np.pi)
	if W[m]>0 and S[m]<0:
		direc[m]=np.degrees(theta[m]+np.pi)
	if W[m]<0 and S[m]<0:
		direc[m]=np.degrees(theta[m]+np.pi)
	"""

for n in i:
	if direc[n]<22.5 and direc[n]>337.5:
		v[n]=0.
	if 67.5>=direc[n]>=22.5:
		v[n]=1.
	if 112.5>direc[n]>67.5:
		v[n]=2.
	if 157.5>=direc[n]>=112.5:
		v[n]=3.
	if 202.5>direc[n]>157.5:
		v[n]=4.
	if 247.5>=direc[n]>=202.5:
		v[n]=5.
	if 292.5>direc[n]>247.5:
		v[n]=6.
	if 337.5>=direc[n]>=292.5:
		v[n]=7.

# Condiciones del tipo de tiempo

for k in i:
	if np.abs(Z[k])<F[k]:
		cond[k]=1
	if np.abs(Z[k])>2*F[k] and Z[k]<0:
		cond[k]=2
	if np.abs(Z[k])>2*F[k] and Z[k]>0:
		cond[k]=3
	if F[k]<np.abs(Z[k])<2*F[k]:
		cond[k]=0
	if F[k]<3 and np.abs(Z[k])<3:
		cond[k]=4

dd=[]
mm=[]
yy=[]

#print(time)

#print(tvalue)

#str_time = [i.strftime("%Y-%m-%d") for i in tvalue]
agnos = [i.year for i in tvalue]
meses = [i.month for i in tvalue]
dias = [i.day for i in tvalue]
#print(str_time)
tt=np.zeros((len(cond[1:]),6))

tt[:,0]=dias[:-1]
tt[:,1]=meses[:-1]
tt[:,2]=agnos[:-1]
tt[:,3]=cond[:-1]
tt[:,4]=direc[:-1]
tt[:,5]=v[:-1]

np.savetxt('/Users/fredm/Desktop/TFG_DEFINITIVO/Ficheros/Datos_reales/tipos_tiempo_mod_%s.txt'%(new_name(name)),tt,delimiter=' , ',fmt='%.1f')

'''
for j in np.arange(0,len(time)):
	f="%.d %.d" % (time[j]-time[0]+1.,y)
	f=datetime.strptime(f,"%j %Y")
	dd.insert(j,int(datetime.strftime(f,"%d")))
	mm.insert(j,int(datetime.strftime(f,"%m")))
	yy.insert(j,int(datetime.strftime(f,"%Y")))

dd=np.array(dd)
mm=np.array(mm)
yy=np.array(yy)

tt=np.zeros((len(cond),6))
tt[:,0]=dd
tt[:,1]=mm
tt[:,2]=yy
tt[:,4]=cond
tt[:,3]=direc
tt[:,5]=v

np.savetxt('tipos_tiempo_%d.txt' % y,tt,delimiter=',',fmt='%.1f')
'''
# Se representa una gráfica en polares la dirección del viento para cada tipo de tiempo. Hice una rotación de las direcciones porque empieza toma el norte como ángulo 0 y es horario (yo tomé el este como ángulo 0 y dirección antihoraria)

# La rosa de los vientos nos da hacia donde va el viento
"""
for b in np.arange(0,4):
	py.figure("%d" % b)
	a=cond==b
	A=direc[a]
	ax=WindroseAxes.from_ax()
	ax.contour(A,cond[a],bins=np.arange(0,1,1),nsector=64,cmap=cm.cool)
	py.savefig(u'rosa_%d_tipo%d.png' % (y,b))
"""