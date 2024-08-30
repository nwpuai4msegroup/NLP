import pandas as pd


a_df = pd.read_excel('/Users/wangping/Desktop/pythonProject/downstream task/data3.0.xlsx')
a_df = a_df.iloc[:, 13:]

b_df = pd.read_excel('/Users/wangping/Desktop/pythonProject/extract_features/50_comp.xlsx')
cls = []
processing = []

repetitions = [9, 9, 9, 5, 4, 4, 10, 3, 3, 3, 27, 28, 28, 6, 13, 4, 14, 12, 30, 9, 1, 25, 33, 9, 6, 22, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 10, 10, 25, 16, 32, 15, 9, 1, 1, 1, 2, 8, 3, 3]

for i in range(len(b_df)):

    rowb = b_df.iloc[i].to_frame().T
    cls.extend([rowb]*repetitions[i])


cls = pd.concat(cls, ignore_index=True)
data_cls = pd.concat([cls, a_df], axis=1)

data_cls.to_excel('/Users/wangping/Desktop/pythonProject/downstream task/data4.0/data4.0.xlsx', index=False)