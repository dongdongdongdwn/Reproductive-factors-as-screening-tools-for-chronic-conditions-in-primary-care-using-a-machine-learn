import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import choice
import pandas as pd
from pandas import DataFrame
from pandas import Series,DataFrame
import os
import xlrd


data=pd.read_excel("C:\\Users\\siat\\desktop\\dwn\\stroke\\strokeAlbert\\Ablert19\\positive.xlsx")
X=np.array(data)



neigh = NearestNeighbors(n_neighbors = 5)
neigh.fit(X)
N=10

S = np.zeros(shape=(X.shape[0]*(N-1), X.shape[1]))
S = np.vstack((X, S))
print (S)
for i in range(X.shape[0]):
    nn = neigh.kneighbors(X[i].reshape(1, -1), return_distance=False)
    for n in range(N-1):
        nn_index = choice(nn[0])
        #NOTE: nn includes T[i], we don't want to select it
        while nn_index == i:
            nn_index = choice(nn[0])
        dif = X[nn_index] - X[i]
        # print dif
        gap = np.random.random()
        index = n + i * (N-1)+X.shape[0]
        print (index)
        S[index, :] = X[i,:] + gap * dif[:]

print (S)
df=pd.DataFrame(S)
df.to_csv('SMOTEpositve.csv')
