import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


Alpha = ["Al","O","C","N"]
Eutectic_Beta = ["Fe","Mn","Ni","Co","Cu","Si"]
Eutectoid_Beta= ["Mo","V","Nb","Ta","W","Cr"]
Neutral = ["Sn","Zr","Hf"]
file = "/Users/wangping/Desktop/pythonProject/extract_features/19_ele.json"

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
print(labels)

#0-767
col = [34, 443, 472]
X_embedded = CLSS.iloc[:, col]
X_embedded = X_embedded.rename(columns={col[0]: 'Dimension 1',col[1]: 'Dimension 2',col[2]:"Dimension 3"})
print(X_embedded)

kmeans = KMeans(n_clusters=4, n_init=10,random_state=42)  # ,init='k-means++'
kmeans.fit(X_embedded)

labels2 = kmeans.labels_
print(labels2)
#Calculate and adjust the Rand index
ari = adjusted_rand_score(labels, labels2.tolist())
Print (f "Adjust Rand Index (ARI): {ari}")

#Calculate the contour coefficient between -1 and 1
silhouette_avg = silhouette_score(X_embedded, labels2) 
print("Average Silhouette Score:", silhouette_avg)

