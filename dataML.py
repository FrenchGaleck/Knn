import sys
import numpy as np  # importation du package numérique
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.impute import SimpleImputer
from projectML import euclidienne,predict,voisin_proche

wave = 'waveform.data'
# data prends la data et la sépare a chaque fois présence d'un espace
af = np.loadtxt(wave, delimiter=',', skiprows=0, dtype=str, usecols=range(21))

X = pd.DataFrame(af)
"""print(af)
print(X)#contient les 21 premiers colonnes
print(X.shape)
print(X.describe)"""
ay = np.loadtxt(wave, delimiter=',', skiprows=0, dtype=str, usecols=[21])
y = pd.DataFrame(ay)
ay=ay.astype(int)
print(ay)
#print(y)  # vecteur des résultats
"""
#on tranforme en PCA la data pour avoir que deux params
pca=PCA(n_components=2)
x_pca=pca.fit_transform(X)#entraine model pca avec données, transforme ensuite x en x_pca
print(np.shape(x_pca))
plt.figure()
plt.scatter(x_pca[:,0],x_pca[:,1],c=y)#les deux colonnes car pca les as réduits en deux colonnes
plt.show()

"""
# Convert string column to float
for i in range(21):
    X[i] = X[i].astype(float)

# y=y.astype(float)
#print(X)
# create the 2 subsets
Xtrain = X[0:4000]
Xtest =X[4000:4100]
ytrain =ay[0:4000]
ytest =ay[4000:4100]
def bruteforce (Xtrain,Xtest,k):
    resul=[]
    bonresul=0
    for i in range(0,100):
        resul.append(predict(Xtrain,Xtest.iloc[i],k))
    for i in range (0,100):
        if ay[i]==resul[i] :
            bonresul=bonresul + 1

    print(bonresul)


bruteforce(Xtrain,Xtest,1)