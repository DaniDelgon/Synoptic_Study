import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''Dias no coinciden, un día menos
Anticiclon por ciclon
Unir hibridos y lineales
Cambiar gráfica'''


def descrip(l):
    A=np.zeros((4,8))#.astype(object)
    #A[0,:]=np.array([' ','S','SW','W','NW','N','NE','E','SE'])
    #A[:,0]=np.array(['Region %.d'%(l),'Hibrido (0)','FlujoLineal (1)','Anticiclon (2)','Ciclon (3)'])
    return A

su2=np.array(['S','SW','W','NW','N','NE','E','SE'])
su1=np.array(['Hibrido (0)','FlujoLineal (1)','Anticiclon (2)','Ciclon (3)'])
su3=np.array(['S','SW','W','NW','N','NE','E','SE','Hibrido (0)/FlujoLineal (1)','Anticiclon (2)','Ciclon (3)'])
su4=np.array(['Region 1','Region 2','Region 3','Region 4','Region 5','Region 6'])

rain=np.loadtxt('/Users/fredm/Desktop/tfg/tipos_tiempo_mod.txt',delimiter=' , ')
precxreg=np.loadtxt('/Users/fredm/Desktop/tfg/precxreg.txt',delimiter=' , ')

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
    np.savetxt('/Users/fredm/Desktop/tfg/SPREAD_ANALYSIS/Region %s'%(i+1),D,delimiter='  ',fmt='%s')
    fig, ax = plt.subplots()
    im = ax.imshow(D)

    # hide axes
    ax.set_xticks(np.arange(len(su2)))
    ax.set_yticks(np.arange(len(su1)))
    ax.set_xticklabels(su2)
    ax.set_yticklabels(su1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    
    for k in range(len(su1)):
        for l in range(len(su2)):
            text = ax.text(l, k, D[k, l],ha="center", va="center", color="w")

    ax.set_title('Region %.d'%(i+1))
    fig.tight_layout()
    plt.savefig('/Users/fredm/Desktop/tfg/SPREAD_ANALYSIS/Region_%d_ca2.png'%(i+1))
    plt.clf()

G=np.around(G,decimals=2)
fig, ax = plt.subplots()
im = ax.imshow(G)

# hide axes
ax.set_xticks(np.arange(len(su3)))
ax.set_yticks(np.arange(len(su4)))
ax.set_xticklabels(su3)
ax.set_yticklabels(su4)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    
for k in range(len(su4)):
    for l in range(len(su3)):
        text = ax.text(l, k, G[k, l],ha="center", va="center", color="w")

ax.set_title('SPREAD')
fig.tight_layout()
plt.savefig('/Users/fredm/Desktop/tfg/SPREAD_ANALYSIS/SPREAD_Analysis.png')
plt.clf()
