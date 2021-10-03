import numpy as np

nombr=['gfdl_d01_1980_2009_slp_grEra5','ipsl_d01_1980_2009_slp_grEra5','miroc_d01_1980_2009_slp_grEra5','SPREAD_1980_2009_pleased03']
#print(nombr[0].split('_'))
#print(int(nombr[0].split('_')[2]))

def new_name(name):
    namesplit=name.split('_')
    phrase=[namesplit[0]]
    for i in range(0,len(namesplit)):
        if namesplit[i].isdigit():
            phrase.append(namesplit[i])
    name2='_'.join(phrase)
    return name2

def nombre_comp(name):
    new1=[new_name(i) for i in name]
    return new1

def nombre_simpl(name):
    new2=[new_name(i).split('_')[0] for i in name]
    return new2

def nom_fut(name1):
    namesplit=name1.split('_')
    phrase=[namesplit[0],namesplit[2],namesplit[3],namesplit[2]]
    name2='_'.join(phrase)
    return name2

def ind_nombr(val,name1,i,j,k,l):
    namesplit=name1.split('_')
    if val==1:
        phrase=[namesplit[i],namesplit[j],namesplit[k],namesplit[l][:-4]]
    if val==0:
        phrase=[namesplit[i],namesplit[j],namesplit[k],namesplit[l]]#[:-4]]
    name2='_'.join(phrase)
    return name2

def ind_nombr2(name1,i,j,k):
    namesplit=name1.split('_')
    if val==1:
        phrase=[namesplit[i],namesplit[j],namesplit[k][:-4]]
    if val==0:
        phrase=[namesplit[i],namesplit[j],namesplit[k]]#[:-4]]
    name2='_'.join(phrase)
    return name2

def ind_nombr1(val,name1,i,j,k,l,m):
    namesplit=name1.split('_')
    if val==1:
        phrase=[namesplit[i],namesplit[j],namesplit[k],namesplit[l],namesplit[m][:-4]]
    if val==0:
        phrase=[namesplit[i],namesplit[j],namesplit[k],namesplit[l],namesplit[m]]#[:-4]]
    name2='_'.join(phrase)
    return name2




