import pandas as pd


df1 = pd.read_excel("/Users/wangping/Desktop/NLP/NLP-yield and modulus/pareto/front.xlsx", index_col=0,dtype = "float64")
df2 = pd.read_excel("/Users/wangping/Desktop/NLP/NLP-yield and modulus/y_pred_candiates_yield.xlsx", sheet_name= None,dtype = "float64")
df2 = [df2[sheet_name] for sheet_name in df2.keys()]
df3 = pd.read_excel("/Users/wangping/Desktop/NLP/NLP-yield and modulus/y_pred_candiates_E.xlsx", sheet_name= None,dtype = "float64")
df3 = [df3[sheet_name] for sheet_name in df3.keys()]

df2 = pd.concat(df2, ignore_index=True)
df3 = pd.concat(df3, ignore_index=True)
df4 = pd.read_csv("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/candidates.csv")



matching_indices1 = []
for i in range(df1.shape[0]):
    matching = df2[df2.iloc[:, 0] == df1['YS_order'][i]].index
    matching_indices1.extend(matching.tolist())
print(matching_indices1)


matching_indices2 = df3[df3.iloc[:, 0].isin(df1['E_order'])].index
print(matching_indices1)

if len(matching_indices1) == len(matching_indices2):
    print("correct")
else:
    print("error")




file_path = '/Users/wangping/Desktop/pythonProject/D-electron/Result/candidates_all.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()
df = pd.DataFrame(lines, columns=['Value'])
# Remove any leading or trailing whitespace characters (like '\n') from each value
df['Value'] = df['Value'].str.strip()
df = df.loc[df.index.repeat(6)]

df.reset_index(drop=True, inplace=True)
intersection_values = pd.concat([df.iloc[matching_indices1], df4.iloc[matching_indices1]], axis=1)
intersection_values.to_excel("/Users/wangping/Desktop/NLP/NLP-yield and modulus/Pareto/front_alloys.xlsx")
print(intersection_values)






