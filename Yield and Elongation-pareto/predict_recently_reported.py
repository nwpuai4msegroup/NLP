import random
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample


X = pd.read_csv('/Users/wangping/Desktop/pythonProject/features_filt/50_comp_emb(0.75).csv', index_col=0)
Y = pd.read_excel('/Users/wangping/Desktop/pythonProject/downstream task/data4.0/data4.0.xlsx')
processsing_condition = Y.iloc[:, 768:790]
y1 = Y["Yield"]
y2 = Y["Elongation"]

non_zero_indices1 = y1[y1 != 0].index
X_filtered1 = X.loc[non_zero_indices1]
selected_columns1 = ["X23", "X54", "X99", "X108", "X111", "X148", "X159", "X230", "X235", "X319", "X336", "X406",
                     "X705"]
X_filtered1 = X_filtered1[selected_columns1]
y_filtered1 = y1.loc[non_zero_indices1]
processsing_condition_filtered1 = processsing_condition.loc[non_zero_indices1]
#processsing_condition_filtered1 = (processsing_condition_filtered1 - processsing_condition_filtered1.min()) / (processsing_condition_filtered1.max() - processsing_condition_filtered1.min())
#X_filtered1 = (X_filtered1 - X_filtered1.min()) / (X_filtered1.max() - X_filtered1.min())
y1min = y_filtered1.min()
y1max = y_filtered1.max()
#y_filtered1 = (y_filtered1 - y_filtered1.min()) / (y_filtered1.max() - y_filtered1.min())

non_zero_indices2 = y2[y2 != 0].index
X_filtered2 = X.loc[non_zero_indices2]
selected_columns2 = ["X58", "X88", "X214", "X255", "X301", "X343", "X356", "X535", "X609", "X647", "X677", "X688",
                     "X759"]
X_filtered2 = X_filtered2[selected_columns2]
y_filtered2 = y2.loc[non_zero_indices2]
processsing_condition_filtered2 = processsing_condition.loc[non_zero_indices2]
#processsing_condition_filtered2 = (processsing_condition_filtered2 - processsing_condition_filtered2.min()) / (processsing_condition_filtered2.max() - processsing_condition_filtered2.min())
y2min = y_filtered2.min()
y2max = y_filtered2.max()
#X_filtered2 = (X_filtered2 - X_filtered2.min()) / (X_filtered2.max() - X_filtered2.min())
#y_filtered2 = (y_filtered2 - y_filtered2.min()) / (y_filtered2.max() - y_filtered2.min())




all_process = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/recently_reported.xlsx")
all_process = all_process.iloc[:,5:]
#print(all_process)



# print(all_process)
#all_process.to_csv("/home/wangping/exfeatures/result/candidates.csv")
#all_process.info()

X1 = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/yield_GBR_450_zhang.xlsx", header=0)
X1.columns = selected_columns1  
X1 = (X1 - X1.min()) / (X1.max() - X1.min())
X1 = pd.DataFrame(np.repeat(X1, 1, axis=0), columns=X1.columns)

#all_process = (all_process - all_process.min()) / (all_process.max() - all_process.min())
all_process = all_process.fillna(0)
X1 = pd.concat([X1, all_process], axis=1)

X2 = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/elongation_GBR_450_zhang.xlsx", header=0)
X2.columns = selected_columns2
#X2 = (X2 - X2.min()) / (X2.max() - X2.min())
X2 = pd.DataFrame(np.repeat(X2, 1, axis=0), columns=X2.columns)
X2 = pd.concat([X2, all_process], axis=1)


def plt_true_vs_pred1(model, i):
    all_predictions = []
    X_filcomposition_and_process1 = pd.concat([X_filtered1, processsing_condition_filtered1], axis=1)
    for j in range(10):

        X_resampled, y_resampled = resample(X_filcomposition_and_process1, y_filtered1)
        model.fit(X_resampled, y_resampled)
        y_pred1 = model.predict(X_resampled)
        r2 = r2_score(y_resampled, y_pred1)

        prediction = model.predict(X1)
        all_predictions.append(prediction)

    all_predictions = np.array(all_predictions)

    #all_predictions = all_predictions * (y1max - y1min) + y1min

    prediction_mean = all_predictions.mean(axis=0)
    prediction_mean = pd.DataFrame(prediction_mean)
    prediction_std = all_predictions.std(axis=0)
    prediction_std = pd.DataFrame(prediction_std)
    prediction = pd.concat([prediction_mean, prediction_std], axis=1)

    with pd.ExcelWriter('/Users/wangping/Desktop/pythonProject/D-electron/Pareto/y_pred_candiates_yield_zhang_%s.xlsx'%i) as writer:
        prediction.to_excel(writer, index=False)

    return r2


def plt_true_vs_pred2(model, i):

    all_predictions = []
    X_filcomposition_and_process2 = pd.concat([X_filtered2, processsing_condition_filtered2], axis=1)
    for j in range(10):

        X_resampled, y_resampled = resample(X_filcomposition_and_process2, y_filtered2)

        model.fit(X_resampled, y_resampled)
        y_pred1 = model.predict(X_resampled)
        r2 = r2_score(y_resampled, y_pred1)

        prediction = model.predict(X2)
        all_predictions.append(prediction)

    all_predictions = np.array(all_predictions)


    #all_predictions = all_predictions * (y2max - y2min) + y2min

    prediction_mean = all_predictions.mean(axis=0)
    prediction_mean = pd.DataFrame(prediction_mean)
    print(prediction_mean)
    prediction_std = all_predictions.std(axis=0)
    prediction_std = pd.DataFrame(prediction_std)
    prediction = pd.concat([prediction_mean, prediction_std], axis=1)
    # prediction.to_excel('/Users/wangping/Desktop/pythonProject/D-electron/Pareto/y_pred_candiates_elongation.xlsx',index=False)



    with pd.ExcelWriter('/Users/wangping/Desktop/pythonProject/D-electron/Pareto/y_pred_candiates_elongation_zhang_%s.xlsx'%i) as writer:
        prediction.to_excel(writer,index=False)

    return r2


def train_and_chose_model(key, value,i):
    model = value

    if key == 'GBR':
        r2_train_yield = plt_true_vs_pred1(model,i)
        r2_train_elonation = plt_true_vs_pred2(model,i)
        #print(r2_train_yield, r2_train_elonation)


if __name__ == '__main__':

    all_model = {
        'GBR': GradientBoostingRegressor(n_estimators=500,
                                                learning_rate=0.01,
                                                max_depth=5,
                                                min_samples_split=5)
    }

    random_list = [0]
    for i in range(len(random_list)):
        np.random.seed(random_list[i])
        for key, value in all_model.items():
            train_and_chose_model(key, value, i)
