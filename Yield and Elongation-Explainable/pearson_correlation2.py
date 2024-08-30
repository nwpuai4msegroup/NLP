import pandas as pd
from scipy.stats import spearmanr


Alpha = ["Al","O","C","N"]
Eutectic_Beta = ["Fe","Mn","Ni","Co","Cu","Si"]
Eutectoid_Beta= ["Mo","V","Nb","Ta","W","Cr"]
Neutral = ["Sn","Zr","Hf"]

nature = pd.read_csv("/Users/wangping/Desktop/pythonProject/extract_features/Elemental properties.csv")
xingzhi = pd.DataFrame(nature.iloc[2, 1:].T,dtype=float)  
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


# 0-767
col = [663]
X_embedded = CLSS.iloc[:, col]

pd.set_option('display.max_rows', None)    
pd.set_option('display.max_columns', None) 
print(X_embedded)
spearman_corr, _ = spearmanr(xingzhi, X_embedded)

