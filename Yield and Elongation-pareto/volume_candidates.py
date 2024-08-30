import pandas as pd
import numpy as np

elongation_sheets = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/y_pred_candiates_elongation.xlsx", sheet_name=None)
yield_sheets = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/y_pred_candiates_yield.xlsx", sheet_name=None)


elongation = pd.concat(elongation_sheets.values(), ignore_index=True)
yield_ = pd.concat(yield_sheets.values(), ignore_index=True)
print(elongation.shape)

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

objects = [(x, y) for x, y in zip(-1 * elongation.iloc[:,0], -1 * yield_.iloc[:,0])]
front = Pareto_Front(objects)
front = pd.DataFrame(-1*front)
front.columns = ["elongation","YS"]

mapping_dict1 = dict(zip(elongation.iloc[:, 0], elongation.iloc[:, 1]))
mapping_dict2 = dict(zip(yield_.iloc[:, 0], yield_.iloc[:, 1]))
front['elongation_s'] = front['elongation'].map(mapping_dict1)
front['YS_s'] = front['YS'].map(mapping_dict2)
print(front)


front.to_excel("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/front.xlsx")

