import pandas as pd
import  numpy as np

Elements = pd.read_csv('/Users/wangping/Desktop/pythonProject/D-electron/elemental_properties.csv', index_col=0)
Elements.fillna(0, inplace=True)


def expand_features(data, n):
    samples = []
    moeq = []
    Bo_list = []
    Md_list = []
    new_data = pd.DataFrame(None, columns=data.columns)
    for index, row in data.iterrows():

        moeq += [sum(Elements.loc['Mo/Al equivalent(wt%)', row.index] * row.values)]
  
        Atom = {}
        total_atom = 0
        Bo = 0
        Md = 0
        for j in range(len(row)):
            atom = (float(row[j])) / Elements.loc['Relative atomic mass', row.index[j]]  
            Atom[row.index[j]] = atom  
            total_atom += atom
        for k, v in Atom.items():
            Atom[k] = v / total_atom 

            Bo += Elements.loc['Bo value(bcc)', k] * float(v / total_atom)
            Md += Elements.loc['Md(eV)(bcc)', k] * float(v / total_atom)
        Bo_list += [Bo]
        Md_list += [Md]
        samples += [Atom]
    new_data = pd.DataFrame(samples)
    new_data['Moeq'] = moeq
    new_data['Bo'] = Bo_list
    new_data['Md'] = Md_list
    new_data = new_data.set_index(data.index)
    return new_data


def filter(n_df):

    bound = pd.read_excel('/Users/wangping/Desktop/pythonProject/D-electron/Md-Bo.xls')
    x1 = bound['eMfY'].values
    y1 = bound['eMfX'].values
    x2 = bound['slip/twinY'].values
    y2 = bound['slip/twinX'].values
    min_x = max(min(x1), min(x2))
    max_x = min(max(x1), max(x2))

    fliter = []
    # x=Bo, y=Md
    for i in n_df.index:
        #     x = float(n_df['Bo'][i])
        x = n_df.loc[i, 'Bo']
        if min_x <= x <= max_x:
            y = float(n_df['Md'][i])
            fliter += [fun(x, y, x1, y1, x2, y2)]
        else:
            fliter += [False]

    n_df['fliter'] = fliter
    inter = n_df[n_df['fliter'] == True]
    return inter



# d电子理论筛选
def fun(x, y, x1, y1, x2, y2):

    i = False
    j = False
    a = x2 - x
    if 0 in a:
        if y > y2[np.where(a == 0)]:
            i = True
    else:
        c = np.where(a > 0)[0]
        min_y = min(y2[c[0]], y2[c[0] - 1])
        if y >= min_y:
            i = True

    # if lower than f1
    b = x1 - x
    if 0 in b:
        if y < y1[np.where(b == 0)]:
            j = True
    else:
        c = np.where(b > 0)[0]
        max_y = max(y1[c[0]], y1[c[0] - 1])
        if y <= max_y:
            j = True

    if i and j is True:
        # plt.scatter(x, y)
        # print(x, y, 'is inter')
        return True
    else:
        return False

data = pd.read_excel("/Users/wangping/Desktop/NLP-data/data2.xlsx")
y1 = data["YS"]
non_zero_indices1 = y1[y1 != 0].index
y2 = data["E"]
non_zero_indices2 = y2[y2 != 0].index

intersection_indices = set(non_zero_indices1).intersection(set(non_zero_indices2))
intersection_list = list(intersection_indices)
df = data.loc[intersection_list, :].iloc[:, 0:15]
df = expand_features(df, 15)#15元
#df = df.drop_duplicates()

inter = filter(df)
inter_diffcom = inter.drop_duplicates()

print("含不同成分", inter_diffcom.shape[0])

performance_and_process = data.iloc[inter.index]
performance = performance_and_process[["YS", "E"]]
process = data.iloc[inter.index, 15:16]
performance_and_process = pd.concat([performance, process], axis=1)
inter = pd.concat([inter, performance_and_process], axis=1)
inter.to_csv('/Users/wangping/Desktop/NLP-data/pareto/selected_train_points.csv')
inter_diffcom.to_csv("/Users/wangping/Desktop/NLP-data/pareto/selected_train_points_diffcom.csv")





