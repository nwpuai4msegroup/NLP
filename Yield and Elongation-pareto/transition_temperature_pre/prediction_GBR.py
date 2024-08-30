import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

#for train
df = pd.read_excel('/Users/wangping/Desktop/NLP/pythonProject/downstream task/data3.0.xlsx')
X = df.iloc[:, :13]
Y = pd.DataFrame(df["transition temperature"])
#Retrieve row indices with transition temperatures not equal to 0
non_zero_indices = Y[Y["transition temperature"] != 0].index
X = X.loc[non_zero_indices]
#Remove duplicates
X = X.drop_duplicates()
index = X.index
Y = Y.loc[index]
print(Y)
#Do not normalize, Ta is all 0
print(X)

mu = Y.mean()
sigma = Y.std()
print(mu)
print(sigma)


r2_scores_train = []
r2_scores_val = []
mae_train_list = []
mae_val_list = []
rmse_train_list = []
rmse_val_list = []

def plt_true_vs_pred(model):
    random_seed = 10
    for i in range(10):
        random_seed += i
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=random_seed)
        model.fit(X_train, Y_train)
        #model.fit(X,Y)


        y_pred1 = model.predict(X_train)
        y_pred2 = model.predict(X_val)
        y_pred1 = pd.DataFrame(y_pred1)
        y_pred2 = pd.DataFrame(y_pred2)
        """
        y_pred1 = (y_pred1 - 910.12963) / 84.608328
        y_pred2 = (y_pred2 - 910.12963) / 84.608328
        Y_train = (Y_train - 910.12963) / 84.608328
        Y_val = (Y_val - 910.12963) / 84.608328
        
        print(y_pred1,y_pred2,Y_train, Y_val)
        """

        #true_and_train = pd.concat([Y_train, y_pred], axis=1)
        #true_and_train = pd.concat([Y_val, y_pred], axis=1)
        #r2 = r2_score(Y_train, y_pred)
        r2_train = r2_score(Y_train, y_pred1)
        r2_val = r2_score(Y_val, y_pred2)
        # MAE
        mae_train = mean_absolute_error(Y_train, y_pred1)
        mae_val = mean_absolute_error(Y_val, y_pred2)

        # RMSE
        rmse_train = np.sqrt(mean_squared_error(Y_train, y_pred1))
        rmse_val = np.sqrt(mean_squared_error(Y_val, y_pred2))

        r2_scores_train.append(r2_train)
        r2_scores_val.append(r2_val)
        mae_train_list.append(mae_train)
        mae_val_list.append(mae_val)
        rmse_train_list.append(rmse_train)
        rmse_val_list.append(rmse_val)

    r2_scores_train_df = pd.DataFrame(r2_scores_train, columns=["R² train"])
    r2_scores_val_df = pd.DataFrame(r2_scores_val, columns=["R² val"])
    mae_train_list_df = pd.DataFrame(mae_train_list, columns=["mae train"])
    mae_val_list_df = pd.DataFrame(mae_val_list, columns=["mae val"])
    rmse_train_list_df = pd.DataFrame(rmse_train_list, columns=["rmse train"])
    rmse_val_list_df = pd.DataFrame(rmse_val_list, columns=["rmse val"])

    r2 = pd.concat([r2_scores_train_df, r2_scores_val_df, mae_train_list_df, mae_val_list_df,rmse_train_list_df, rmse_val_list_df], axis=1)

    r2.to_excel("/Users/wangping/Desktop/train_transition_temperature1.xlsx")
    return r2


def train_and_chose_model(key, value, X, Y):

        model = value
        if key == 'GBR':
            r2_train_transtem = plt_true_vs_pred(model)



if __name__ == '__main__':

    all_model = {
        'GBR': GradientBoostingRegressor(n_estimators=500,
                                                         learning_rate=0.01,
                                                         max_depth=5,
                                                         min_samples_split=5)
    }

    for key, value in all_model.items():
        train_and_chose_model(key, value, X, Y)
