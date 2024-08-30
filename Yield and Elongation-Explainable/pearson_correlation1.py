import pandas as pd
from itertools import combinations
from scipy import stats


Alpha = ["Al","O","C","N"]
Eutectic_Beta = ["Fe","Mn","Ni","Co","Cu","Si"]
Eutectoid_Beta= ["Mo","V","Nb","Ta","W","Cr"]
Neutral = ["Sn","Zr","Hf"]

nature = pd.read_csv("/Users/wangping/Desktop/pythonProject/extract_features/Elemental properties.csv")
xingzhi = pd.DataFrame(nature.iloc[10,1:].T,dtype=float)  #First line property
print(xingzhi)

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



combinations_list = list(combinations(range(768), 1))
for i in range(len(combinations_list)):
    X_embedded = CLSS.iloc[:, list(combinations_list[i])]
    # Calculate Spearman correlation coefficient
    #spearman_corr, _ = spearmanr(xingzhi, X_embedded)
    correlation, p_value = stats.pearsonr(xingzhi.iloc[:,0], X_embedded.iloc[:,0])
    #print(i,spearman_corr)
    print(i, correlation)


