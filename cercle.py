import sys
import numpy as np  # importation du package numérique
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.impute import SimpleImputer

# euclidienne distance between 2 vectors
def euclidienne(l1, l2):
    # représentera la distance
    dist = 0.0
    """on va boucle pour qu'il aille dans toutes les colonnes"""
    for i in range(21):
        # def de la dist euclidienne
        dist = dist + pow((l1[i] - l2[i]), 2)

    # on fait la racine carré de la distance
    dist = pow(dist, 0.5)
    return dist

def voisin_proche_cercle(X,vecteur,k,indexlist):
    # 'indexlist' est une liste de 4000 entiers (un pour chaque donnée)
    # Lors de l'appel de cette fonction, tous ces entiers sont initialisés à -1
    # A la fin de cette fonction récursive, les k plus proches voisins auront pour valeur d'entier 1, et les autres -1
    # Pendant la récursivité, les données qui seront exclues via la méthode des cercles auront temporairement un entier de valeur 0
    
    # version avec la méthode des cercles :
    global matricedistance
    global voisin
    dist = []
    index_voisin = -1
    dist_voisin = -1
    d = -1
    for X_row in X.transpose():
        if (indexlist[X_row] == 0):
            indexlist[X_row] = -1
        else :
            if (indexlist[X_row] == -1) :
                dist_temp = euclidienne(vecteur, X.iloc[X_row])
                if ( (dist_voisin == -1) or (dist_temp <= dist_voisin)) :
                    index_voisin = X_row
                    dist_voisin = dist_temp
                    j = 0
                else :
                    if (X_row != 3999) :
                        # on applique la méthode
                        r = dist_temp - dist_voisin
                        R = dist_temp + dist_voisin
                        # Pour chaque donnée restante
                        for i in range(X_row+1, len(matricedistance)) :
                            if (matricedistance[i][X_row] == -1) :
                                d = euclidienne(X.iloc[i],X.iloc[X_row])
                                matricedistance[i][X_row] = d
                                matricedistance[X_row][i] = d
                            else :
                                d = matricedistance[i][X_row]
                            # Si la donnée n'est pas comprise entre les deux cercles
                            if ( (d>R) or (d<r) ):
                                # La valeur ne sera pas le plus proche voisin
                                indexlist[i] = 0
    # A la fin on a notre plus proche voisin
    indexlist[index_voisin] = 1
    if (k != 0):
        voisin_proche_cercle(X,vecteur,k-1,indexlist)
                    
    

                             

def predict(X, vecteur, k):
    indexlist = [-1] * 4000
    voisin_proche_cercle(X, vecteur, k, indexlist)
    voisin = []
    for i in range(4000) :
        if (indexlist[i] == 1):
            voisin.append(i)
    # fin de la méthode
    
    print(voisin)
    output_values = [ytrain[row] for row in voisin]
    #print(output_values)
    #prendra le nombre qui apparait le plus de fois
    prediction = max(set(tuple(output_values)), key=output_values.count)
    #print(prediction)
    return prediction

                             
wave = 'waveform.data'
# data prends la data et la sépare a chaque fois présence d'un espace
af = np.loadtxt(wave, delimiter=',', skiprows=0, dtype=str, usecols=range(21))
#on utilise panda
X = pd.DataFrame(af)
ay = np.loadtxt(wave, delimiter=',', skiprows=0, dtype=str, usecols=[21])
y = pd.DataFrame(ay)
ay=ay.astype(int)
# Convert string column to float
for i in range(21):
    X[i] = X[i].astype(float)
#y=y.astype(float)
#print(X)
#création et remplissage de la matrice des distances
matricedistance = []
for i in range(4000):
    matricedistance.append([])
    for j in range(4000):
        matricedistance[i].append(-1)
#calcul distance euclidienne entre deux lignes
euclidienne(X.loc[29], X.loc[30])
#Liste des k plus proches voisins
voisin = []
#affiche tableau du plus proche voisins
voisin=voisin_proche(X,X.loc[1],5)
# create the 2 subsets
Xtrain = X[0:4000]
Xtest =X[4000:5000]
ytrain =ay[0:4000]
ytest =ay[4000:4999]
resul=predict(Xtrain,Xtest.iloc[2],1)
print(resul)
print(ytest[2])
print(ytrain[308])
