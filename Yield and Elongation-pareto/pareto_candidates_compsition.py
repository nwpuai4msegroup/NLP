import pandas as pd

df1 = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/front_candidates.xlsx", index_col=0,dtype = "float64")
df2 = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/y_pred_candiates_yield.xlsx", sheet_name= None,dtype = "float64")
df2 = [df2[sheet_name] for sheet_name in df2.keys()]
df3 = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/y_pred_candiates_elongation.xlsx", sheet_name= None,dtype = "float64")
df3 = [df3[sheet_name] for sheet_name in df3.keys()]

df2 = pd.concat(df2, ignore_index=True)
df3 = pd.concat(df3, ignore_index=True)
#print(df2)


#Step 1: Obtain the value of the first column in df2 that matches df1 ['YS']
matching_indices = df2[df2.iloc[:, 0].isin(df1['YS'])].index

#Step 2: Create a mapping dictionary to map the values in df1 ['YS'] to their indexes
index_mapping = {value: idx for idx, value in enumerate(df1['YS'])}
#Step 3: Sort matching_indices based on the index order of each value in df2.iloc [:, 0] in df1 ['YS']
sorted_matching_indices1 = sorted(matching_indices, key=lambda x: index_mapping[df2.iloc[x, 0]])
print(sorted_matching_indices1)



#0 corresponds to the first alloy
#Print the value of df2.iloc [:, 0] under the sorted_matching_indices index
for index in sorted_matching_indices1:
    print(df2.iloc[index, 0])





matching_indices = df3[df3.iloc[:, 0].isin(df1['elongation'])].index
index_mapping = {value: idx for idx, value in enumerate(df1['elongation'])}
sorted_matching_indices2 = sorted(matching_indices, key=lambda x: index_mapping[df3.iloc[x, 0]])
print(sorted_matching_indices2)
for index in sorted_matching_indices2:
    print(df3.iloc[index, 0])

#intersection
intersection_indices = list(set(sorted_matching_indices1) & set(sorted_matching_indices2))
print(intersection_indices)









# 读取文件并创建 DataFrame，假设分隔符为制表符（可以根据实际情况更改分隔符）
file_path = '/Users/wangping/Desktop/pythonProject/D-electron/Result/candidates_all.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

df = pd.DataFrame(lines, columns=['Value'])
# Remove any leading or trailing whitespace characters (like '\n') from each value
df['Value'] = df['Value'].str.strip()

df = df.loc[df.index.repeat(6)].reset_index(drop=True)

# 确保索引从 0 开始
df.reset_index(drop=True, inplace=True)
# 查看 DataFrame
print(df)
intersection_values = df.iloc[intersection_indices]
print(intersection_values)






