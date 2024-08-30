import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import GPy
from GPy.models import GPRegression
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score,cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_excel('/Users/wangping/Desktop/pythonProject/downstream task/data3.0.xlsx')
X = df.iloc[:,:35]
Y = pd.DataFrame(df["Strength"])  


non_zero_indices = Y[Y["Strength"] != 0].index
X = X.loc[non_zero_indices]
Y = Y.loc[non_zero_indices]

#pd.set_option('display.max_rows', None) 
#print(Y)

X = (X - X.min()) / (X.max() - X.min())
Y_or = Y
#print(Y.min())
#print(Y.max())
Y = (Y - Y.min()) / (Y.max() - Y.min())
perf = pd.DataFrame()

def plt_true_vs_pred(model, true_values, predicted_values, seed):
    kf = KFold(n_splits=4, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X):
        train_index = train_index.tolist()
        test_index = test_index.tolist()
        #print(test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        true_values.extend(y_test.iloc[:, 0].tolist())
        predicted_values.extend([item for item in y_pred.tolist()])


    ## inverse transform
    true_values_arr = np.array(true_values)
    y_true = true_values_arr * (int(Y_or.max()) - int(Y_or.min())) + int(Y_or.min())
    #print(y_true)
    y_true_df = pd.DataFrame(y_true)
    #y_true_df.to_excel("true.xlsx",index=False)

    predicted_value_arr = np.array(predicted_values)
    y_pred = predicted_value_arr * (int(Y_or.max()) - int(Y_or.min())) + int(Y_or.min())
    #print(y_pred)
    y_pred_df = pd.DataFrame(y_pred)
    #y_pred_df.to_excel("pred.xlsx", index=False)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    new = pd.DataFrame({'R2':[r2], 'MAE':[mae], 'RMSE':[mse]})
    return new

def train_and_chose_model(key, value, X_filcomposition_and_process, y_filtered):
    perf = pd.DataFrame() 
    for seed in [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]:

        true_values = []
        predicted_values = []
        model = value
        if key == 'LR':

            #scores = cross_val_score(model, X_filcomposition_and_process, y_filtered, cv=kf, scoring='r2', n_jobs=-1)
            #print(scores.mean())

            perf_one = plt_true_vs_pred(model,
                             true_values,
                             predicted_values,
                             seed)


        if key == 'RF':
            #scores = cross_val_score(model, X_filcomposition_and_process, y_filtered, cv=kf, scoring='r2', n_jobs=-1)
            #print(scores.mean())

            perf_one = plt_true_vs_pred(model,
                             true_values,
                             predicted_values,
                             seed)


        if key == 'GBR':

            perf_one = plt_true_vs_pred(model,
                             true_values,
                             predicted_values,
                             seed)

        if key == 'SVR':
            perf_one = plt_true_vs_pred(model,
                             true_values,
                             predicted_values,
                             seed)

        if key == 'MLP':
            perf_one = plt_true_vs_pred(model,
                             true_values,
                             predicted_values,
                             seed)

        if key == 'GPR':
            perf_one = plt_true_vs_pred(model,
                             true_values,
                             predicted_values,
                             seed)

        if key == 'GBR-Grid Search':

            ## define parameter choices
            param_grid = {
                'n_estimators': list(range(300, 700, 50)),
                'max_depth': list(range(2, 11, 2)),
                'min_samples_split': list(range(2, 11, 2)),
                'learning_rate': [0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
            }

            grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=kf, n_jobs=-1)
            grid_search.fit(X, Y)
            print("Best parameters：", grid_search.best_params_)
            print("Best R2 score：", grid_search.best_score_)

        
            best_model = grid_search.best_estimator_
            perf_one = plt_true_vs_pred(best_model,
                             true_values,
                             predicted_values,
                             seed)

        perf = pd.concat([perf, perf_one], ignore_index=True)
    return perf

if __name__ == '__main__':

    all_model = {
        'LR': LinearRegression(),


    }
    perf = pd.DataFrame()
    for key, value in all_model.items():
        new = train_and_chose_model(key, value, X, Y)
        print(new)
        perf = pd.concat([perf, new], ignore_index=True)
    perf.to_excel("/Users/wangping/Desktop/pythonProject/downstream task/prediction.xlsx", index=False)

    """
                     'LR': LinearRegression(),
                     'RF': RandomForestRegressor(n_estimators=200,
                                              max_depth=10,
                                              min_samples_split=3),
                     'GBR': GradientBoostingRegressor(n_estimators=500,
                                                             learning_rate=0.01,
                                                             max_depth=5,
                                                             min_samples_split=5),
                     'SVR': svm.SVR(C=1.2),
                     'MLP': MLPRegressor(hidden_layer_sizes=[128, 256, 128], activation='relu',
                                              alpha=0.05, max_iter=2000, solver='adam', verbose=False,
                                              tol=1e-10),
                     'GPR': GaussianProcessRegressor(
                         kernel=RBF(length_scale=0.5) + WhiteKernel(noise_level=(0.05) ** 2)),
                    
                    'GBR-Grid Search'： GradientBoostingRegressor()
    """

