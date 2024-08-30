import numpy as np
import pandas as pd


df = pd.read_excel('/Users/wangping/Desktop/NLP-data/217samples-217alloys.xlsx')
data768 = df.iloc[:, :768]

new_column_names = ['X' + str(i) for i in range(1, 769)]
data768.columns = new_column_names

#Calculate the correlation coefficient matrix between features
correlation_matrix = data768.corr()
#Create a collection to store the features to be deleted
features_to_remove = set()


for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.75:
            colname = correlation_matrix.columns[i]
            features_to_remove.add(colname)


data768_filtered = data768.drop(columns=features_to_remove)


print("The number of filtered features：", data768_filtered.shape[1])

data768_filtered.to_csv("/Users/wangping/Desktop/NLP-data/217samples-217alloys(0.75).csv")  