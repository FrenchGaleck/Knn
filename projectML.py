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


# trois param : data, le vecteur
def voisin_proche(X, vecteur, k):
    # stockera les 5000 distances
    dist = []
    j = 0
    for X_row in X.transpose():
        # on va comparer avec tout les autres vecteurs
        dist_temp = euclidienne(vecteur,X.iloc[X_row])
        # liste qui contiens des couples d'elem, avec comme premiere valeur le voisin, et seconde valeur la dist
        dist.append((j, X_row, dist_temp))
        j += 1
    # on trie les distances par ordre croissant
    dist.sort(key=lambda tri: tri[2])
    #print(dist)
    voisin = []
    # on va prendre les k plus proches voisins
    for i in range(k):
        # on va rajouter dans notre vecteur voisin le numero du voisin le i plus proche
        voisin.append(dist[i][0])
        print(voisin)
    return voisin


def predict(X, vecteur, k):
    voisin = voisin_proche(X, vecteur, k)
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
print(X)
#calcul distance euclidienne entre deux lignes
euclidienne(X.loc[29], X.loc[30])
#affiche tableau du plus proche voisins
#voisin=voisin_proche(X,X.loc[1],10)
# create the 2 subsets
Xtrain = X[0:4000]
Xtest =X[4000:5000]
ytrain =ay[0:4000]
ytest =ay[4000:4999]
resul=predict(Xtrain,Xtest.iloc[2],1)
print(resul)
print(ytest[2])
print(ytrain[308])

