import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor



#for train
df = pd.read_excel('/Users/wangping/Desktop/pythonProject/downstream task/data3.0.xlsx')
X = df.iloc[:, :13]
Y = pd.DataFrame(df["transition temperature"])
#Retrieve row indices with transition temperatures not equal to 0
non_zero_indices = Y[Y["transition temperature"] != 0].index
X = X.loc[non_zero_indices]
X = X.drop_duplicates()
index = X.index
Y = Y.loc[index]
print(Y)
#Do not normalize, Ta is all 0
print(X)





#candidates
data2 = pd.read_csv("/Users/wangping/Desktop/pythonProject/D-electron/Result/selected_samples_wt_2.csv", index_col=0)
data3 = pd.read_csv("/Users/wangping/Desktop/pythonProject/D-electron/Result/selected_samples_wt_3.csv", index_col=0)
data4 = pd.read_csv("/Users/wangping/Desktop/pythonProject/D-electron/Result/selected_samples_wt_4.csv", index_col=0)
data5 = pd.read_csv("/Users/wangping/Desktop/pythonProject/D-electron/Result/selected_samples_wt_5.csv", index_col=0)
data6 = pd.read_csv("/Users/wangping/Desktop/pythonProject/D-electron/Result/selected_samples_wt_6.csv", index_col=0)
X2 = pd.concat([data2, data3, data4, data5, data6], axis=0)
#Rearrange the features in the order of the training set, add a column of oxygen, all zero
X2 = X2[["Ti", "Mo", "Al", "Sn", "V", "Zr", "Cr", "Nb", "Ta", "Fe", "W", "Si"]]
X2['O'] = 0
#print(X2)


def plt_true_vs_pred(model):

    model.fit(X,Y)
    y_pred = model.predict(X)
    y_pred = pd.DataFrame(y_pred)
    true_and_train = pd.concat([Y, y_pred], axis=1)
    true_and_train.to_excel("/Users/wangping/Desktop/pythonProject/D-electron/transition_temperature_pre/train_transition_temperature.xlsx")
    r2 = r2_score(Y, y_pred)
    y_pred_test = model.predict(X2)
    y_pred_test = pd.DataFrame(y_pred_test)

    #y_pred_test.to_excel('/Users/wangping/Desktop/pythonProject/D-electron/transition_temperature_pre/transition_temperature_pre.xlsx', index=False)
    return r2


def train_and_chose_model(key, value, X, Y, X2):

        model = value
        if key == 'GBR':
            r2_train_transtem = plt_true_vs_pred(model)
            print(r2_train_transtem)


if __name__ == '__main__':

    all_model = {
        'GBR': GradientBoostingRegressor(n_estimators=500,
                                                         learning_rate=0.01,
                                                         max_depth=5,
                                                         min_samples_split=5)
    }

    for key, value in all_model.items():
        train_and_chose_model(key, value, X, Y, X2)
