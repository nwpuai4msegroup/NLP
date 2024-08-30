import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import random


Alpha = ["Al","O","C","N"]
Eutectic_Beta = ["Fe","Mn","Ni","Co","Cu","Si"]
Eutectoid_Beta= ["Mo","V","Nb","Ta","W","Cr"]
Neutral = ["Sn","Zr","Hf"]
file = "/home/wangping/kmeans/19.json"

lines = []
with open(file, "r", encoding="utf-8") as f:
    for line in f.readlines():
        line = eval(line)
        lines.append(line)
    f.close()
CLSS = []
for i in range(19):
    content = lines[i]
    CLS = content['features'][1]["layers"][0]["values"]    
    CLSS.append(CLS)
CLSS = pd.DataFrame(CLSS)

labels = []
labels += (["Alpha"] * len(Alpha))  #"Alpha"
labels += (["Eutectic_Beta"] * len(Eutectic_Beta))  #"Eutectic_Beta"
labels += (["Eutectoid_Beta"] * len(Eutectoid_Beta))  #"Eutectoid_Beta"
labels += (["Neutral"] * len(Neutral))  #"Neutral"


for i in range(1000000):

    selected_columns = list(random.sample(range(768), 100))
    X_embedded = CLSS.iloc[:, selected_columns]
    X_embedded = pd.DataFrame(X_embedded)
    #Using K-means clustering
    kmeans = KMeans(n_clusters=4, n_init=10,random_state=42)  # ,init='k-means++'
    kmeans.fit(X_embedded)
 
    labels2 = kmeans.labels_
    #Calculate and adjust the Rand index
    ari = adjusted_rand_score(labels, labels2.tolist())
    if ari > 0.7:
        ari_df = pd.DataFrame({'ARI': [ari]})
        ari_df.to_csv('ari_values0.7_50.csv', mode='a', header=False, index=False)
        selected_columns = pd.DataFrame({'weidu': [selected_columns]})
        selected_columns.to_csv('weidu0.7_50.csv', mode='a', header=False, index=False)
    if ari > 0.8:
        ari_df = pd.DataFrame({'ARI': [ari]})
        ari_df.to_csv('ari_values0.8_50.csv', mode='a', header=False, index=False)
        selected_columns = pd.DataFrame({'weidu': [selected_columns]})
        selected_columns.to_csv('weidu0.8_50.csv', mode='a', header=False, index=False)
    if ari > 0.9:
        ari_df = pd.DataFrame({'ARI': [ari]})
        ari_df.to_csv('ari_values0.9_50.csv', mode='a', header=False, index=False)
        selected_columns = pd.DataFrame({'weidu': [selected_columns]})
        selected_columns.to_csv('weidu0.9_50.csv', mode='a', header=False, index=False)