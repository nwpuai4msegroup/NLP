import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

num_heads = 12
num_layers = 12
file = '/Users/wangping/Desktop/pythonProject/extract_features/Alloy Attention.json'
with open(file, "r", encoding="utf-8") as f:
    lines = json.load(f)   
f.close()


heads = {}
for i in range(num_heads):
    head = lines["attention_ws"][0]["attention_ws"]["head_%s" % i]
    head = pd.DataFrame(head)
    head = head.loc[~(head == 0).all(axis=0)]  
    head = head.loc[:, list(head.index)]
    #print(head)  

    for j in range(head.shape[1]):
        #max = head[j].max()
        #min = head[j].min()
        #head[j] = ((head[j] - min) / (max - min))
        head[j].loc[head[j] <= 0.1] = 0  
    heads["head_%s" % i] = head

pd.DataFrame(heads["head_8"]).to_excel("/Users/wangping/Desktop/pythonProject/extract_features/attention.xlsx")
#pd.DataFrame(heads["head_10"]).to_excel("/Users/wangping/Desktop/pythonProject/extract_features/attention2.xlsx")

for i in range(0, 12):
    sns.heatmap(data=heads['head_%s'% i])
    plt.show()
