from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import time

import cartopy
import cartopy.feature as cpf
import cartopy.crs as ccrs
import sys

import rpy2
from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

import sklearn.cluster as cl
from sklearn.preprocessing import StandardScaler

psych = importr('psych')

start_time = time.time()

filenamesp='/Users/fredm/Desktop/tfg/Otros/SPREAD_1980_2009_pleased03.nc'
varsp = 'pcp'

#os.environ['R_HOME'] = r'C:/Users/fredm/anaconda3/lib/'
ncinsp = Dataset(filenamesp, 'r')
prsp = ncinsp.variables[varsp][:]
lonssp = ncinsp.variables['lon'][:]
latssp = ncinsp.variables['lat'][:]
ncinsp.close()

h=1

#para probar cosas mas rapido (menos instantes temporales)
prsp=prsp[:,:,:]

"""
if h==1:
    prsp=prsp[:,:,:]
else:
    prsp=prsp[:400,:,:]
"""

Xsp = np.reshape(prsp, (prsp.shape[0], len(latssp) * len(lonssp)), order='F')

oceansp = Xsp.sum(0).mask
landsp = ~oceansp

Xsp = Xsp[:,landsp]
print("Xsp",Xsp.shape)

Xsp_mean = Xsp.mean(axis=0)
Xmsp = Xsp - Xsp_mean

rsp = robjects.r
numpy2ri.activate()
msp = rsp.matrix(Xmsp)

ncomps=np.array([6])

for l in range(0,len(ncomps)):
    #CREAMOS LOS DIRECTORIOS DONDE GUARDAR LOS GRAFICOS ORDENADAMENTE
    n_comps=ncomps[l]
    print(n_comps)
    dir = '/Users/fredm/Desktop/tfg'
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = '/Users/fredm/Desktop/tfg/%.d_compstratra'%(n_comps)
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = '/Users/fredm/Desktop/tfg/%.d_compstratra/KMEANS'%(n_comps)
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = '/Users/fredm/Desktop/tfg/%.d_compstratra/MEANSHIFT'%(n_comps)
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = '/Users/fredm/Desktop/tfg/%.d_compstratra/AFFINITY_PROPAGATION'%(n_comps)
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = '/Users/fredm/Desktop/tfg/%.d_compstratra/DBS_SCAN'%(n_comps)
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = '/Users/fredm/Desktop/tfg/%.d_compstratra/PCA'%(n_comps)
    if not os.path.exists(dir):
        os.makedirs(dir)
    f=open('/Users/fredm/Desktop/tfg/%.d_compstratra/PCA/%.d_compstratra.txt'%(n_comps,n_comps),'w')
    f=open('/Users/fredm/Desktop/tfg/%.d_compstratra/PCA/%.d_compstratra.txt'%(n_comps,n_comps),'a')
    psp = psych.principal(Xmsp, rotate="varimax", nfactors=n_comps)
    loadingssp = np.array(psp.rx2('loadings'))
    
    print("loadsp",loadingssp.shape)
    lamsp = np.array(psp.rx2('Vaccounted'))[1,:]

    f2, ax = plt.subplots(figsize=(5,5))
    ax.plot(lamsp[0:20]*100)
    ax.plot(lamsp[0:20]*100,'ro')
    ax.set_ylabel(u'Percent of variance explained')
    ax.set_xlabel(u'Principal component')
    ax.set_ylim(0,30)
    ax.grid()
    plt.savefig('/Users/fredm/Desktop/tfg/%.d_compstratra/02SPREAD_varianza_%.d_compstratra.png'%(n_comps,n_comps))

    EOFssp = loadingssp.T
    expl_varsp = lamsp
    maxsp=np.argmax(EOFssp,axis=0)

    #Recostrucción de la matriz de pcs y de unos conteniendo las zonas con mas pc
    EOF_reconssp = np.ones((n_comps, len(latssp) * len(lonssp))) * -999.
    ones=np.ones((n_comps, len(latssp) * len(lonssp))) * -999.
    #print(EOF_reconssp.shape)

    for i in range(n_comps):
        isp=np.where(maxsp==i)
        y=np.ones(np.sum(landsp))
        y[isp[0]]=0
        y=y*-999.
        y[isp[0]]=EOFssp[i,isp[0]]
        EOF_reconssp[i,landsp] = y #EOFssp[i,isp[0][k]]
        y[isp[0]]=1
        ones[i,landsp] = y
    
    EOF_reconssp = np.flip(np.ma.masked_values(np.reshape(EOF_reconssp, (n_comps, len(latssp), len(lonssp)), order='C'), -999.),axis=1)
    ones = np.flip(np.ma.masked_values(np.reshape(ones, (n_comps, len(latssp), len(lonssp)), order='C'), -999.),axis=1)

    #IMPRIMIMOS LAS COORDENADAS DE CADA PCA
    coord=np.zeros((1310,2*n_comps))

    for i in range(0,n_comps):
        ysp=np.where(EOF_reconssp[i,:]>0)
        coord[:len(ysp[0]),2*i]=ysp[0]
        coord[:len(ysp[1]),2*i+1]=ysp[1]

    eof_sp = EOF_reconssp

    Usp = np.array(psp.rx2('weights'))
    Asp = np.dot(Xmsp,Usp)

    #...

    #...
    #MATRICES DE PC MAYOR POR SEPARADO
    for j in range(n_comps):
        fig3 = plt.figure()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        ax.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax.imshow(eof_sp[j,:,:], extent=img_extent, cmap=plt.cm.inferno, norm=colors.Normalize(vmin=0.0, 	vmax=1.0))
        ax.set_title("PC%.d (%3.2f)" %(j+1,lamsp[j]*100),fontsize='9')
        cbi = plt.colorbar(im, shrink=0.7, format='%.2f',ax=ax,orientation='horizontal')
        ocean = cpf.NaturalEarthFeature(category='physical',name='ocean',scale='10m',facecolor='white')
        if h==1:
            ax.add_feature(ocean, zorder=100, edgecolor='k')
        plt.savefig('/Users/fredm/Desktop/tfg/%.d_compstratra/PCA/02SPREAD%.d_PC.png' % (n_comps,j+1))

    #GRAFICOS DE TODOS LOS PC DE MAYOR PESO POR DIFERENTE COLOR
    fig3 = plt.figure()
    
    for j in range(n_comps):
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        ax.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax.imshow(np.trunc(j*ones[j,:,:]), extent=img_extent, cmap=plt.cm.tab20, norm=colors.Normalize(vmin=0.0, 	vmax=n_comps))#,label='PC%.d'%(j+1))
        ax.set_title("PCA (%3.2f)" %(lamsp[j]*100),fontsize='9')
        ocean = cpf.NaturalEarthFeature(
	            category='physical',
	            name='ocean',
	            scale='10m',
	            facecolor='white')
        if h==1:
            ax.add_feature(ocean, zorder=100, edgecolor='k')

    plt.savefig('/Users/fredm/Desktop/tfg/%.d_compstratra/PCA/PCA.png'%(n_comps))
    np.savetxt('/Users/fredm/Desktop/tfg/%.d_compstratra/PCA/%.d_compstratra.txt'%(n_comps,n_comps),coord,fmt='%.2f')
    f.close()
    
    EOF_reconssp = np.ones((n_comps, len(latssp) * len(lonssp))) * -999.

    for i in range(n_comps): 
	    EOF_reconssp[i,landsp] = EOFssp[i,:]

    EOF_reconssp = np.flip(np.ma.masked_values(np.reshape(EOF_reconssp, (n_comps, len(latssp), len(lonssp)), order='C'), -999.),axis=1)
    eof_sp = EOF_reconssp

    Usp = np.array(psp.rx2('weights'))
    Asp = np.dot(Xmsp,Usp)
    
    maxsp=np.argmax(loadingssp,axis=1)

    #iwrf0=np.where(maxwrf==0)
    #iwrf1=np.where(maxwrf==1)
    #iwrf2=np.where(maxwrf==2)
    #iwrf3=np.where(maxwrf==3)
    isp0=np.where(maxsp==0)
    isp1=np.where(maxsp==1)
    isp2=np.where(maxsp==2)
    isp3=np.where(maxsp==3)

    #mwrf0=np.mean(Xwrf[:,iwrf0],axis=1)
    #mwrf1=np.mean(Xwrf[:,iwrf1],axis=1)
    #mwrf2=np.mean(Xwrf[:,iwrf2],axis=1)
    #mwrf3=np.mean(Xwrf[:,iwrf3],axis=1)
    msp0=np.mean(Xsp[:,isp0],axis=0)
    msp1=np.mean(Xsp[:,isp1],axis=0)
    msp2=np.mean(Xsp[:,isp2],axis=0)
    msp3=np.mean(Xsp[:,isp3],axis=0)

    msp=np.ones((n_comps,Xsp.shape[-1]))
    ssp=np.ones((n_comps,Xsp.shape[0]))*-999.

    for k in range(n_comps):
	    ind = np.where(maxsp==k)
	    msp[k,ind] = np.mean(Xsp[:,ind], axis=0)
	    aux = Xsp[:,ind][:,0,:]
	    ssp[k,:] = np.mean(aux, axis=1)

    PCsp_r = np.ones((n_comps, len(latssp) * len(lonssp))) * -999.

    for i in range(n_comps): 
	    PCsp_r[i,landsp] = msp[i,:]

    PCsp_r =np.flip(np.ma.masked_values(np.reshape(PCsp_r, (n_comps, len(latssp), len(lonssp)), order='C'), -999.),axis=1)

    #GRAFICAS DE LLUVIA
    for j in range(n_comps):
        fig3 = plt.figure()
        ax = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        ax.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax.imshow(eof_sp[j,:,:], extent=img_extent, cmap=plt.cm.inferno, norm=colors.Normalize(vmin=0.0, 	vmax=1.0))
        ax.set_title("PC%.d (%3.2f)" %(j+1,lamsp[j]*100),fontsize='9')
        cbi = plt.colorbar(im, shrink=0.7, format='%.2f',ax=ax,orientation='horizontal')
        ocean = cpf.NaturalEarthFeature(category='physical',name='ocean',scale='10m',facecolor='white')
        if h==1 :
            ax.add_feature(ocean, zorder=100, edgecolor='k')
        #--------------
        ax1 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax1.coastlines(resolution='10m', color='black', linewidth=1)
        ax1.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax1.imshow(PCsp_r[j,:,:], extent=img_extent, cmap=plt.cm.Blues, norm=colors.Normalize(vmin=0.0, 	vmax=int(np.max(msp))+1))
        ax1.set_title("Average rainfall PC%.d" % (j+1),fontsize='9')
        cbi = plt.colorbar(im, format='%.2f',ax=ax1,orientation='horizontal')
        ocean = cpf.NaturalEarthFeature(
	            category='physical',
	            name='ocean',
	            scale='10m',
	            facecolor='white')
        if h==1:
            ax.add_feature(ocean, zorder=100, edgecolor='k')
	
        plt.savefig('/Users/fredm/Desktop/tfg/%.d_compstratra/rain_02SPREAD%.d_PC.png' % (n_comps,j+1))

    #KMEANS
    f=open('/Users/fredm/Desktop/tfg/%.d_compstratra/KMEANS/regiones.txt'%(n_comps),'w')
    f=open('/Users/fredm/Desktop/tfg/%.d_compstratra/KMEANS/regiones.txt'%(n_comps),'a')

    n_class=n_comps

    y_pred = cl.KMeans(n_clusters=n_class, random_state=170).fit_predict(loadingssp)

    EOF_reconssp = np.ones((n_class, len(latssp) * len(lonssp))) * -999.
    ones=np.ones((n_class, len(latssp) * len(lonssp))) * -999.

    for i in range(n_class):
        isp=np.where(y_pred==i)
        y=np.ones(np.sum(landsp))
        y[isp[0]]=0
        y=y*-999.
        y[isp[0]]=EOFssp[i,isp[0]]
        EOF_reconssp[i,landsp] = y
        y[isp[0]]=1
        #ones=np.flip(ones,axis=0)
        ones[i,landsp] = y

    EOF_reconssp = np.flip(np.ma.masked_values(np.reshape(EOF_reconssp, (n_class, len(latssp), len(lonssp)), order='C'), -999.),axis=1)
    ones = np.flip(np.ma.masked_values(np.reshape(ones, (n_class, len(latssp), len(lonssp)), order='C'), -999.),axis=1)

    coord=np.zeros((1310,2*n_class))

    for i in range(0,n_class):
        ysp=np.where(EOF_reconssp[i,:]>0)
        coord[:len(ysp[0]),2*i]=ysp[0]
        coord[:len(ysp[1]),2*i+1]=ysp[1]

    eof_sp = EOF_reconssp

    Usp = np.array(psp.rx2('weights'))
    Asp = np.dot(Xmsp,Usp)

    #...

    #...

    for j in range(n_class):
        fig3 = plt.figure()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        ax.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax.imshow(eof_sp[j,:,:], extent=img_extent, cmap=plt.cm.inferno, norm=colors.Normalize(vmin=0.0, 	vmax=1.0))
        ax.set_title("KMEANS_CLASS_%.d" %(j+1),fontsize='9')
        cbi = plt.colorbar(im, shrink=0.7, format='%.2f',ax=ax,orientation='horizontal')
        ocean = cpf.NaturalEarthFeature(
	            category='physical',
	            name='ocean',
	            scale='10m',
                facecolor='white')
        if h==1:
            ax.add_feature(ocean, zorder=100, edgecolor='k')
	
        plt.savefig('/Users/fredm/Desktop/tfg/%.d_compstratra/KMEANS/02SPREAD%.d_Class_KMEANS.png' % (n_comps,j+1))


    fig3 = plt.figure()

    for j in range(n_class):
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        ax.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax.imshow(np.trunc(j*ones[j,:,:]), extent=img_extent, cmap=plt.cm.tab20, norm=colors.Normalize(vmin=0.0, 	vmax=n_comps))
        ax.set_title("KMEANS" ,fontsize='9')
        
        ocean = cpf.NaturalEarthFeature(
                category='physical',
                name='ocean',
                scale='10m',
                facecolor='white')
        if h==1:
            ax.add_feature(ocean, zorder=100, edgecolor='k')
        	
    plt.savefig('/Users/fredm/Desktop/tfg/%.d_compstratra/KMEANS/02SPREAD_KMEANS.png'%(n_comps))

    np.savetxt('/Users/fredm/Desktop/tfg/%.d_compstratra/KMEANS/regiones.txt'%(n_comps),coord,fmt='%.2f')
    f.close()

    #MEANSHIFTS

    f=open('/Users/fredm/Desktop/tfg/%.d_compstratra/MEANSHIFT/regiones.txt'%(n_comps),'w')
    f=open('/Users/fredm/Desktop/tfg/%.d_compstratra/MEANSHIFT/regiones.txt'%(n_comps),'a')

    bandwidth = cl.estimate_bandwidth(loadingssp, quantile=0.2, n_samples=len(loadingssp[:,0]))
    ms = cl.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(loadingssp)
    y_pred = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(y_pred)
    n_class = len(labels_unique)

    EOF_reconssp = np.ones((n_class, len(latssp) * len(lonssp))) * -999.
    ones=np.ones((n_class, len(latssp) * len(lonssp))) * -999.

    for i in range(n_class):
        isp=np.where(y_pred==i)
        y=np.ones(np.sum(landsp))
        y[isp[0]]=0
        y=y*-999.
        y[isp[0]]=EOFssp[i,isp[0]]
        EOF_reconssp[i,landsp] = y
        y[isp[0]]=1
        ones[i,landsp] = y

    EOF_reconssp = np.ma.masked_values(np.reshape(EOF_reconssp, (n_class, len(latssp), len(lonssp)), order='C'), -999.)
    ones = np.ma.masked_values(np.reshape(ones, (n_class, len(latssp), len(lonssp)), order='C'), -999.)

    coord=np.zeros((1310,2*n_class))

    for i in range(0,n_class):
        ysp=np.where(EOF_reconssp[i,:]>0)
        coord[:len(ysp[0]),2*i]=ysp[0]
        coord[:len(ysp[1]),2*i+1]=ysp[1]

    eof_sp = EOF_reconssp

    Usp = np.array(psp.rx2('weights'))
    Asp = np.dot(Xmsp,Usp)

    #...

    #...

    for j in range(n_class):
        fig3 = plt.figure()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        ax.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax.imshow(eof_sp[j,:,:], extent=img_extent, cmap=plt.cm.inferno, norm=colors.Normalize(vmin=0.0, 	vmax=1.0))
        ax.set_title("MEANSHIFT_CLASS_%.d" %(j+1),fontsize='9')
        cbi = plt.colorbar(im, shrink=0.7, format='%.2f',ax=ax,orientation='horizontal')
        ocean = cpf.NaturalEarthFeature(
	            category='physical',
	            name='ocean',
	            scale='10m',
                facecolor='white')
        if h==1:
            ax.add_feature(ocean, zorder=100, edgecolor='k')
	
        plt.savefig('/Users/fredm/Desktop/tfg/%.d_compstratra/MEANSHIFT/02SPREAD%.d_Class_MEANSHIFT.png' % (n_comps,j+1))


    fig3 = plt.figure()

    for j in range(n_class):
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        ax.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax.imshow(np.trunc(j*ones[j,:,:]), extent=img_extent, cmap=plt.cm.tab20, norm=colors.Normalize(vmin=0.0, 	vmax=n_comps))
        ax.set_title('MEANSHIFT',fontsize='9')
        
        ocean = cpf.NaturalEarthFeature(
                category='physical',
                name='ocean',
                scale='10m',
                facecolor='white')
        if h==1:
            ax.add_feature(ocean, zorder=100, edgecolor='k')
        	
    plt.savefig('/Users/fredm/Desktop/tfg/%.d_compstratra/MEANSHIFT/02SPREAD_MEANSHIFT.png'%(n_comps))
    np.savetxt('/Users/fredm/Desktop/tfg/%.d_compstratra/MEANSHIFT/regiones.txt'%(n_comps),coord,fmt='%.2f')
    f.close()

    """
    #AFFINITY_PROPAGATION

    f=open('/Users/Asus/Desktop/tfg/%.d_comps/AFFINITY_PROPAGATION/regiones.txt'%(n_comps),'w')
    f=open('/Users/Asus/Desktop/tfg/%.d_comps/AFFINITY_PROPAGATION/regiones.txt'%(n_comps),'a')

    af = cl.AffinityPropagation(preference=None).fit(loadingssp)
    cluster_centers_indices = af.cluster_centers_indices_
    y_pred = af.labels_

    n_class = len(cluster_centers_indices)
    print(n_class)
    EOF_reconssp = np.ones((n_class, len(latssp) * len(lonssp))) * -999.
    ones=np.ones((n_class, len(latssp) * len(lonssp))) * -999.

    for i in range(n_class):
        isp=np.where(y_pred==i)
        y=np.ones(np.sum(landsp))
        y[isp[0]]=0
        y=y*-999.
        y[isp[0]]=EOFssp[i,isp[0]]
        EOF_reconssp[i,landsp] = y
        y[isp[0]]=1
        ones[i,landsp] = y

    EOF_reconssp = np.ma.masked_values(np.reshape(EOF_reconssp, (n_class, len(latssp), len(lonssp)), order='C'), -999.)
    ones = np.ma.masked_values(np.reshape(ones, (n_class, len(latssp), len(lonssp)), order='C'), -999.)

    coord=np.zeros((1310,2*n_class))

    for i in range(0,n_class):
        ysp=np.where(EOF_reconssp[i,:]>0)
        coord[:len(ysp[0]),2*i]=ysp[0]
        coord[:len(ysp[1]),2*i+1]=ysp[1]

    eof_sp = EOF_reconssp

    Usp = np.array(psp.rx2('weights'))
    Asp = np.dot(Xmsp,Usp)

    #...

    #...

    for j in range(n_class):
        fig3 = plt.figure()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        ax.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax.imshow(eof_sp[j,:,:], extent=img_extent, cmap=plt.cm.inferno, norm=colors.Normalize(vmin=0.0, 	vmax=1.0))
        ax.set_title("AFFINITY_PROPAGATION_CLASS_%.d" %(j+1),fontsize='9')
        cbi = plt.colorbar(im, shrink=0.7, format='%.2f',ax=ax,orientation='horizontal')
        ocean = cpf.NaturalEarthFeature(
	            category='physical',
	            name='ocean',
	            scale='10m',
                facecolor='white')
        if h==1:
            ax.add_feature(ocean, zorder=100, edgecolor='k')
	
        plt.savefig('/Users/Asus/Desktop/tfg/%.d_comps/AFFINITY_PROPAGATION/02SPREAD%.d_Class_AFFINITY_PROPAGATION.png' % (n_comps,j+1))


    fig3 = plt.figure()

    for j in range(n_class):
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        ax.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax.imshow(np.trunc(j*ones[j,:,:]), extent=img_extent, cmap=plt.cm.tab20, norm=colors.Normalize(vmin=0.0, 	vmax=n_comps))
        ax.set_title('AFFINITY_PROPAGATION',fontsize='9')
        
        ocean = cpf.NaturalEarthFeature(
                category='physical',
                name='ocean',
                scale='10m',
                facecolor='white')
        if h==1:
            ax.add_feature(ocean, zorder=100, edgecolor='k')
        	
    plt.savefig('/Users/Asus/Desktop/tfg/%.d_comps/AFFINITY_PROPAGATION/02SPREAD_AFFINITY_PROPAGATION.png'%(n_comps))
    np.savetxt('/Users/Asus/Desktop/tfg/%.d_comps/AFFINITY_PROPAGATION/regiones.txt'%(n_comps),coord,fmt='%.2f')
    f.close()
    """

    #DBS_SCAN

    f=open('/Users/fredm/Desktop/tfg/%.d_compstratra/DBS_SCAN/regiones.txt'%(n_comps),'w')
    f=open('/Users/fredm/Desktop/tfg/%.d_compstratra/DBS_SCAN/regiones.txt'%(n_comps),'a')

    #loadingssp1 = StandardScaler().fit_transform(loadingssp)
    db = cl.DBSCAN(eps=0.1, min_samples=80).fit(loadingssp)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    y_pred = db.labels_
    labels=y_pred
    y_pred=y_pred+1

    # Number of clusters in labels, ignoring noise if present.
    n_class = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(n_class)

    EOF_reconssp = np.ones((n_class, len(latssp) * len(lonssp))) * -999.
    ones=np.ones((n_class, len(latssp) * len(lonssp))) * -999.

    for i in range(n_class):
        isp=np.where(y_pred==i)
        y=np.ones(np.sum(landsp))
        y[isp[0]]=0
        y=y*-999.
        y[isp[0]]=EOFssp[i,isp[0]]
        EOF_reconssp[i,landsp] = y
        y[isp[0]]=1
        ones[i,landsp] = y

    EOF_reconssp = np.ma.masked_values(np.reshape(EOF_reconssp, (n_class, len(latssp), len(lonssp)), order='C'), -999.)
    ones = np.ma.masked_values(np.reshape(ones, (n_class, len(latssp), len(lonssp)), order='C'), -999.)

    coord=np.zeros((1310,2*n_class))

    for i in range(0,n_class):
        ysp=np.where(EOF_reconssp[i,:]>0)
        coord[:len(ysp[0]),2*i]=ysp[0]
        coord[:len(ysp[1]),2*i+1]=ysp[1]

    eof_sp = EOF_reconssp

    Usp = np.array(psp.rx2('weights'))
    Asp = np.dot(Xmsp,Usp)

    #...

    #...

    for j in range(n_class):
        fig3 = plt.figure()
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        ax.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax.imshow(eof_sp[j,:,:], extent=img_extent, cmap=plt.cm.inferno, norm=colors.Normalize(vmin=0.0, 	vmax=1.0))
        ax.set_title("DBS_SCAN_CLASS_%.d" %(j+1),fontsize='9')
        cbi = plt.colorbar(im, shrink=0.7, format='%.2f',ax=ax,orientation='horizontal')
        ocean = cpf.NaturalEarthFeature(
	            category='physical',
	            name='ocean',
	            scale='10m',
                facecolor='white')
        if h==1:
            ax.add_feature(ocean, zorder=100, edgecolor='k')
	
        plt.savefig('/Users/fredm/Desktop/tfg/%.d_compstratra/DBS_SCAN/02SPREAD%.d_Class_DBS_SCAN.png' % (n_comps,j+1))


    fig3 = plt.figure()

    for j in range(n_class):
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        ax.set_extent([-18.3, -13.3, 27.5, 29.4])
        img_extent = [np.min(lonssp), np.max(lonssp), np.min(latssp), np.max(latssp)]
        im = ax.imshow(np.trunc(j*ones[j,:,:]), extent=img_extent, cmap=plt.cm.tab20, norm=colors.Normalize(vmin=0.0, 	vmax=n_comps))
        ax.set_title('DBS_SCAN',fontsize='9')
        
        ocean = cpf.NaturalEarthFeature(
                category='physical',
                name='ocean',
                scale='10m',
                facecolor='white')
        if h==1:
            ax.add_feature(ocean, zorder=100, edgecolor='k')
        	
    plt.savefig('/Users/fredm/Desktop/tfg/%.d_compstratra/DBS_SCAN/02SPREAD_DBS_SCAN.png'%(n_comps))
    np.savetxt('/Users/fredm/Desktop/tfg/%.d_compstratra/DBS_SCAN/regiones.txt'%(n_comps),y_pred,fmt='%.2f')
    f.close()
    



elapsed_time = time.time() - start_time
print('%.3f horas de ejecución del programa.'%(elapsed_time/3600.))
