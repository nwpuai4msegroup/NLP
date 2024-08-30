import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


X = pd.read_csv('/home/wangping/paleituo/217samples-217alloys(0.75).csv', index_col=0)
Y = pd.read_excel('/home/wangping/paleituo/data2.xlsx')
processsing_condition = Y.iloc[:, 15:16]
y1 = Y["YS"]
y2 = Y["E"]

non_zero_indices1 = y1[y1 != 0].index

X_filtered1 = X.loc[non_zero_indices1]
selected_columns1 = ["X23", "X54","X99","X108","X111","X148","X159","X230","X235","X319","X336","X406","X705"]
X_filtered1 = X_filtered1[selected_columns1]
y_filtered1 = y1.loc[non_zero_indices1]
#print(y_filtered1)
processsing_condition_filtered1 = processsing_condition.loc[non_zero_indices1]

processsing_condition_filtered1 = (processsing_condition_filtered1 - processsing_condition_filtered1.min()) / (processsing_condition_filtered1.max() - processsing_condition_filtered1.min())
#print(processsing_condition_filtered1)
X_filtered1 = (X_filtered1 - X_filtered1.min()) / (X_filtered1.max() - X_filtered1.min())
y1min = y_filtered1.min()
y1max = y_filtered1.max()
y_filtered1 = (y_filtered1 - y_filtered1.min()) / (y_filtered1.max() - y_filtered1.min())



non_zero_indices2 = y2[y2 != 0].index

X_filtered2 = X.loc[non_zero_indices2]
selected_columns2 = ["X58","X88","X214","X255","X301","X343","X356","X535","X609","X647","X677","X688","X759"]
X_filtered2 = X_filtered2[selected_columns2]
y_filtered2 = y2.loc[non_zero_indices2]
processsing_condition_filtered2 = processsing_condition.loc[non_zero_indices2]

processsing_condition_filtered2 = (processsing_condition_filtered2 - processsing_condition_filtered2.min()) / (processsing_condition_filtered2.max() - processsing_condition_filtered2.min())
y2min = y_filtered2.min()
y2max = y_filtered2.max()
X_filtered2 = (X_filtered2 - X_filtered2.min()) / (X_filtered2.max() - X_filtered2.min())
y_filtered2 = (y_filtered2 - y_filtered2.min()) / (y_filtered2.max() - y_filtered2.min())




X1 = pd.read_excel("/home/wangping/paleituo/yield_MLP_450.xlsx", header=0)

all_process = pd.DataFrame(index=range(X1.shape[0]*6))
print(all_process.shape)
offsets = np.tile([0, 1, 2,3,4,5], X1.shape[0])
offsets = pd.DataFrame(offsets, columns=["condition_type"])
print(offsets.shape)
all_process = pd.concat([all_process, offsets], axis=1)
print(all_process.shape)



#print(all_process)
all_process.to_csv("/home/wangping/paleituo/candidates.csv")
all_process.info()


X1.columns = selected_columns1  
X1 = (X1-X1.min())/(X1.max()-X1.min())
X1 = pd.DataFrame(np.repeat(X1, 6, axis=0), columns=X1.columns)


all_process = (all_process - all_process.min()) / (all_process.max() - all_process.min())
all_process = all_process.fillna(0)
X1 = pd.concat([X1, all_process], axis=1)


X2 = pd.read_excel("/home/wangping/paleituo/E_MLP_450.xlsx", header=0)
X2.columns = selected_columns2
X2 = (X2-X2.min())/(X2.max()-X2.min())
X2 = pd.DataFrame(np.repeat(X2, 6, axis=0), columns=X2.columns)
X2 = pd.concat([X2, all_process], axis=1)







def plt_true_vs_pred1(model):


    all_predictions = []
    X_filcomposition_and_process1 = pd.concat([X_filtered1, processsing_condition_filtered1], axis=1)
    for i in range(500):

        X_resampled, y_resampled = resample(X_filcomposition_and_process1, y_filtered1)
        model.fit(X_resampled, y_resampled)
        y_pred1 = model.predict(X_resampled)
        r2 = r2_score(y_resampled, y_pred1)
        prediction = model.predict(X1)
        all_predictions.append(prediction)
    all_predictions = np.array(all_predictions)


    all_predictions = all_predictions*(y1max-y1min)+y1min
    print(all_predictions.shape)

    prediction_mean = all_predictions.mean(axis=0)
    prediction_mean = pd.DataFrame(prediction_mean)
    prediction_std = all_predictions.std(axis=0)
    prediction_std = pd.DataFrame(prediction_std)

    prediction = pd.concat([prediction_mean, prediction_std], axis=1)
    #prediction.to_excel('/Users/wangping/Desktop/pythonProject/D-electron/Pareto/y_pred_candiates_yield.xlsx', index=False)

    with pd.ExcelWriter('/home/wangping/paleituo/y_pred_candiates_yield.xlsx') as writer:
   
        for i in range(4):
            sheet_name = f'Sheet_{i + 1}'
            start_row = i * 700000
            end_row = min((i + 1) * 700000, len(prediction))
            df_part = prediction.iloc[start_row:end_row]
            df_part.to_excel(writer, sheet_name=sheet_name, index=False)

    return np.mean(r2_scores)


def plt_true_vs_pred2(model):

    all_predictions = []
    X_filcomposition_and_process2 = pd.concat([X_filtered2, processsing_condition_filtered2], axis=1)
    for i in range(500):

        X_resampled, y_resampled = resample(X_filcomposition_and_process2, y_filtered2)
        model.fit(X_resampled, y_resampled)
        y_pred1 = model.predict(X_resampled)
        r2 = r2_score(y_resampled, y_pred1)
        prediction = model.predict(X2)
        all_predictions.append(prediction)
        
    all_predictions = np.array(all_predictions)

    all_predictions = all_predictions*(y2max-y2min)+y2min
    print(all_predictions.shape)  # (500, 5000)500tree
    prediction_mean = all_predictions.mean(axis=0)
    prediction_mean = pd.DataFrame(prediction_mean)
    prediction_std = all_predictions.std(axis=0)
    prediction_std = pd.DataFrame(prediction_std)    
    prediction = pd.concat([prediction_mean, prediction_std], axis=1)
    #prediction.to_excel('/Users/wangping/Desktop/pythonProject/D-electron/Pareto/y_pred_candiates_elongation.xlsx',index=False)


    with pd.ExcelWriter('/home/wangping/paleituo/y_pred_candiates_E.xlsx') as writer:
        for i in range(4):
            sheet_name = f'Sheet_{i + 1}'
            start_row = i * 700000
            end_row = min((i + 1) * 700000, len(prediction))
            df_part = prediction.iloc[start_row:end_row]
            df_part.to_excel(writer, sheet_name=sheet_name, index=False)

    return np.mean(r2_scores)


def train_and_chose_model(key, value):

        model = value
        if key == 'MLP':
            r2_train_yield = plt_true_vs_pred1(model)
            r2_train_E = plt_true_vs_pred2(model)
            print(r2_train_yield, r2_train_E)


if __name__ == '__main__':

    all_model = {
        'MLP': MLPRegressor(hidden_layer_sizes=[128, 256, 128], activation='relu',
                                              alpha=0.05, max_iter=2000, solver='adam', verbose=False,
                                              tol=1e-10),
    }

    for key, value in all_model.items():
        train_and_chose_model(key, value)
