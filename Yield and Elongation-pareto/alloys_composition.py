import pandas as pd

df1 = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/front_candidates.xlsx", index_col=0,dtype = "float64")
df2 = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/y_pred_candiates_yield.xlsx", sheet_name= None,dtype = "float64")
df2 = [df2[sheet_name] for sheet_name in df2.keys()]
df2 = pd.concat(df2, ignore_index=True)
df3 = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/y_pred_candiates_elongation.xlsx", sheet_name= None,dtype = "float64")
df3 = [df3[sheet_name] for sheet_name in df3.keys()]
#Merge all DataFrames and reset the index to ensure index alignment
df3 = pd.concat(df3, ignore_index=True)
df4 = pd.read_csv("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/candidates.csv")


matching_indices1 = []
for i in range(df1.shape[0]):
    
    matching = df2[df2.iloc[:, 0] == df1['YS_order'][i]].index
    matching_indices1.extend(matching.tolist())
print(matching_indices1)

yield_ = 0
index_remove = []
#Print the value of df2.iloc [:, 0] under the sorted_matching_indices index
for index in matching_indices1:
    if df2.iloc[index, 0] == yield_:
        index_remove.append(index)
    yield_ = df2.iloc[index, 0]
matching_indices11 = [item for item in matching_indices1 if item not in index_remove]
print(matching_indices11)


"""
matching_indices2 = []
for i in range(df1.shape[0]):
    #Step 1: Obtain the value of the first column in df2 that matches df1 ['YS']
    matching = df3[df3.iloc[:, 0] == df1['elongation_order'][i]].index
    matching_indices1.extend(matching.tolist())
print(matching_indices2)
elongation = 0
index_remove = []
#Print the value of df2.iloc [:, 0] under the sorted_matching_indices index
for index in matching_indices2:
    if df3.iloc[index, 0] == elongation:
        index_remove.append(index)
    elongation = df2.iloc[index, 0]
matching_indices2 = [item for item in matching_indices2 if item not in index_remove]
print(matching_indices2)
"""



#Read the file and create a DataFrame, assuming the delimiter is a tab (the delimiter can be changed according to the actual situation)
file_path = '/Users/wangping/Desktop/pythonProject/D-electron/Result/candidates_all.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

df = pd.DataFrame(lines, columns=['Value'])
# Remove any leading or trailing whitespace characters (like '\n') from each value
df['Value'] = df['Value'].str.strip()
df = df.loc[df.index.repeat(6)]
#Ensure that the index starts from 0
df.reset_index(drop=True, inplace=True)
intersection_values = pd.concat([df.iloc[matching_indices11], df4.iloc[matching_indices11]], axis=1)
intersection_values.to_excel("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/front_alloys.xlsx")
print(intersection_values)





