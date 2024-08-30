import pandas as pd

def expand_features(data):
    samples = []
    moeq = []
    Bo_list = []
    Md_list = []
    new_data = pd.DataFrame(None, columns=data.columns)
    for index, row in data.iterrows():
        #Calculate Mo equivalent
        moeq += [sum(Elements.loc['Mo/Al equivalent(wt%)', row.index] * row.values)]
        #Calculate atomic percentage
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
        #Calculate the electronic parameters of d
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
    new_data.to_csv('/Users/wangping/Desktop/pythonProject/D-electron/recommand/reported alloy_Bo.csv')
    return new_data

Elements = pd.read_csv('/Users/wangping/Desktop/pythonProject/D-electron/elemental_properties.csv', index_col=0)
Elements.fillna(0, inplace=True)
df = pd.read_csv('/Users/wangping/Desktop/pythonProject/D-electron/recommand/reported alloy.csv',index_col=0)
expand_features(df)
