import sys
import numpy as np  # importation du package numérique
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from projectML import euclidienne,predict,voisin_proche
import matplotlib.pyplot as plt

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
Xtest =X[4000:5000]
ytrain =ay[0:4000]
ytest =ay[4000:5000]
def bruteforce (Xtrain,Xtest,k):
    resul=[]
    bonresul=0
    for i in range(0,100):
        resul.append(predict(Xtrain,Xtest.iloc[i],k))
    print(resul)
    for i in range (0,100):
        if ytest[i]==resul[i] :
            bonresul=bonresul + 1
    print(bonresul)
#print(ytest[0])
#bruteforce(Xtrain,Xtest,1)
#pour regarder avec le classifier scikit learn


#donnera les differentes accuracy en fonction de k, et trouvera le meilleur k, utilisation de scikit learn
def learn(Xtrain,ytrain,Xtest,ytest):
    accuracy = []
    for k in range (1,1000):
        #objectif c'est d'obtenir l'accuracy en fonction de k
        ne=KNeighborsClassifier(n_neighbors=k)
        ne=ne.fit(Xtrain,ytrain)
        pre=ne.predict(Xtest)
        #print(pre)
        bonnerep=0
        for i in range (len(ytest)):
            if ytest[i]==pre[i] :
                bonnerep=bonnerep+1
        acc=bonnerep/len(ytest)
        #contiendra toutes les valeurs de précision
        accuracy.append(acc)
    plt.figure()
    plt.plot(np.arange(1,1000,1,dtype=int),accuracy)
    plt.show()
    #sera le max temp de accuracy
    max = 0
    #c'est l'indice qui représentera val max
    indice = -1
    for i in range(0,999) :
        if accuracy[i] > max :
            max=accuracy[i]
            indice=i
    #LE MEILLEUR K EST DONC 134
    print(indice+1)
learn(Xtrain,ytrain,Xtest,ytest)