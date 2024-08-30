import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Y = pd.read_excel('/Users/wangping/Desktop/pythonProject/downstream task/data4.0/data4.0.xlsx')
y1 = Y["Yield"]
non_zero_indices1 = y1[y1 != 0].index
y2 = Y["Elongation"]
non_zero_indices2 = y2[y2 != 0].index

#Intersection
intersection_indices = set(non_zero_indices1).intersection(set(non_zero_indices2))
intersection_list = list(intersection_indices)

x = y1.loc[intersection_list]
y = y2.loc[intersection_list]
merged_df = pd.concat([x, y], axis=1)
merged_df.to_excel('/Users/wangping/Desktop/pythonProject/D-electron/merged_data.xlsx')




