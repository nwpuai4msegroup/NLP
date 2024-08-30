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


df = pd.read_csv('/Users/wangping/Desktop/NLP-Creep data/high-temperature titanium alloys(88 samples).csv', index_col=0)
X = df.iloc[:, 0:23]  
Y = pd.DataFrame(df["Creep rupture life (h)"])
X = (X - X.min()) / (X.max() - X.min())
Y_or = Y
Y = (Y - Y.min()) / (Y.max() - Y.min())
perf = pd.DataFrame()


def plt_true_vs_pred(model, true_values, predicted_values, seed):
    kf = KFold(n_splits=4, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X):  
        train_index = train_index.tolist()
        test_index = test_index.tolist()
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        true_values.extend(y_test.iloc[:, 0].tolist())
        predicted_values.extend([item for item in y_pred.tolist()])

    """
    ## inverse transform
    true_values_arr = np.array(true_values)

    y_true = true_values_arr * (int(Y_or.max()) - int(Y_or.min())) + int(Y_or.min())
    predicted_value_arr = np.array(predicted_values)
    y_pred = predicted_value_arr * (int(Y_or.max()) - int(Y_or.min())) + int(Y_or.min())


    r2 = r2_score(y_true, y_pred)

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)
    pridect = pd.DataFrame({'R2': [r2], 'MAE': [mae], 'RMSE': [mse]})
    """

    true_values_arr = np.array(true_values)
    predicted_value_arr = np.array(predicted_values)

    r2 = r2_score(true_values_arr, predicted_value_arr)
 
    mae = mean_absolute_error(true_values_arr, predicted_value_arr)

    mse = mean_squared_error(true_values_arr, predicted_value_arr)
    pridect = pd.DataFrame({'R2': [r2], 'MAE': [mae], 'RMSE': [mse]})
    return pridect


def train_and_chose_model(key, value):
    perf = pd.DataFrame() 
    for seed in [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]:
        true_values = []
        predicted_values = []
        model = value
        if key == 'LR':
            perf_one = plt_true_vs_pred(model,
                                        true_values,
                                        predicted_values,
                                        seed)

        if key == 'RF':
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

        perf = pd.concat([perf, perf_one], ignore_index=True)
    return perf


if __name__ == '__main__':
    all_model = {
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
    }
    all_perf = pd.DataFrame()
    for key, value in all_model.items():
        perf = train_and_chose_model(key, value)
        print(perf)
        all_perf = pd.concat([all_perf, perf], ignore_index=True)
    all_perf.to_excel("/Users/wangping/Desktop/NLP-Creep data/prediction.xlsx", index=False)
