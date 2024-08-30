import pandas as pd
import numpy as np

E_sheets = pd.read_excel("/Users/wangping/Desktop/NLP/NLP-yield and modulus/y_pred_candiates_E.xlsx", sheet_name=None)
yield_sheets = pd.read_excel("/Users/wangping/Desktop/NLP/NLP-yield and modulus/y_pred_candiates_yield.xlsx", sheet_name=None)

e
E = pd.concat(E_sheets.values(), ignore_index=True)
yield_ = pd.concat(yield_sheets.values(), ignore_index=True)
print(E.shape)

def Pareto_Front(points):
    """
    points: [(x1,y1),(x2,y2),...,(xn,yn)]
    """
    pareto_front = [points[0]]
    for i in points:
        dominate_x = []
        dominate_y = []
        for j in range(len(pareto_front)):
            dominate_x += [i[0] < pareto_front[j][0]]
            dominate_y += [i[1] < pareto_front[j][1]]
        if all([any([n, m]) for n, m in zip(dominate_x, dominate_y)]):
            bools = [all([n, m]) for n, m in zip(dominate_x, dominate_y)]
            pareto_front = [points for points, b in zip(pareto_front, bools) if b is not True]
            pareto_front.append(i)
    pareto_front.sort()
    return np.array(pareto_front)

objects = [(x, y) for x, y in zip(1 * E.iloc[:,0], -1 * yield_.iloc[:,0])]
front = Pareto_Front(objects)
front = pd.DataFrame(front)

front.columns = ["E","YS"]
front['YS'] = -1*front['YS']


mapping_dict1 = dict(zip(E.iloc[:, 0], E.iloc[:, 1]))
mapping_dict2 = dict(zip(yield_.iloc[:, 0], yield_.iloc[:, 1]))
front['E_std'] = front['E'].map(mapping_dict1)
front['YS_std'] = front['YS'].map(mapping_dict2)
print(front)


front.to_excel("/Users/wangping/Desktop/NLP/NLP-yield and modulus/pareto/front.xlsx")

