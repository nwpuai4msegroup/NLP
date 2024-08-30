# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict,train_test_split
from joblib import Parallel, delayed
import time
import random
from sklearn.metrics import r2_score


X = pd.read_csv('/Users/wangping/Desktop/NLP-Creep data/88samples-14alloys_em(0.75).csv', index_col=0)
Y = pd.read_csv('/Users/wangping/Desktop/NLP-Creep data/high-temperature titanium alloys(88 samples).csv')
Y = Y.iloc[:, 1:]
processsing_condition = Y.iloc[:, 0:11]
y = Y["Creep rupture life (h)"]
y = pd.DataFrame(y)
print(y)
print(processsing_condition)
non_zero_indices = y[y != 0].index


X_filtered = X.loc[non_zero_indices]
y_filtered = y.loc[non_zero_indices]
processsing_condition_filtered = processsing_condition.loc[non_zero_indices]

processsing_condition_filtered = (processsing_condition_filtered - processsing_condition_filtered.min()) / (processsing_condition_filtered.max() - processsing_condition_filtered.min())
X_filtered = (X_filtered - X_filtered.min()) / (X_filtered.max() - X_filtered.min())
y_filtered = (y_filtered - y_filtered.min()) / (y_filtered.max() - y_filtered.min())

def fitness_function(features):


    model = LinearRegression()
    selected_columns = X_filtered.columns[features.astype(bool)]
    print(selected_columns)
    X_filcomposition_and_process = pd.concat([X_filtered[selected_columns], processsing_condition_filtered], axis=1)
    print(X_filcomposition_and_process)
    X_train, X_test, y_train, y_test = train_test_split(X_filcomposition_and_process, y_filtered, test_size=0.2,
                                                        random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    for true_value, pred_value in zip(y_test, y_pred):
        print(f"Real: {true_value}, Predicted: {pred_value}")
    r2 = r2_score(y_test, y_pred)
    return r2

population_size = 3   
num_generations = 2
num_features = X.shape[1]
fixed_num_features = 13  
crossover_rate = 0.8  
mutation_rate = 0.1   
Model = "LR"


population = np.zeros((population_size, num_features), dtype=int)
for i in range(population_size):
    random_indices = np.random.choice(num_features, fixed_num_features, replace=False)
    population[i, random_indices] = 1

start_time = time.time()
for generation in range(num_generations):

    fitness_scores = Parallel(n_jobs=-1)(delayed(fitness_function)(individual) for individual in population)
    print(fitness_scores)


    min_mae = pd.Series(min(fitness_scores), index=[generation])
    min_mae.to_csv('/Users/wangping/Desktop/NLP-Creep data/max_R2(' + Model + ").csv", mode='a', header=False)
    best_individual = population[np.argmax(fitness_scores)]
    pd.DataFrame([best_individual], index=[generation]).to_csv('/Users/wangping/Desktop/NLP-Creep data/best_individual(' + Model + ").csv", mode='a', header=False)
    selected_features_index = [i for i in range(len(best_individual)) if best_individual[i] == 1]
    selected_features = pd.DataFrame([X_filtered.columns[selected_features_index]], index=[generation])
    selected_features.to_csv('/Users/wangping/Desktop/NLP-Creep data/selected_features(' + Model + ").csv", mode='a', header=False)

    if generation != num_generations-1:
        fitness_scores_0_1 = (fitness_scores - min(fitness_scores)) / (max(fitness_scores) - min(fitness_scores))
        total_fitness = sum([score for score in fitness_scores_0_1])
        selection_probabilities = [score / total_fitness for score in fitness_scores_0_1]
        selected_indices = np.random.choice(len(population), size=len(population), p=selection_probabilities)
        selected_population = [population[i] for i in selected_indices]

        new_population = []
        father_population = selected_population[0:int(population_size / 2)]
        mother_population = selected_population[int(population_size / 2):]
        np.random.shuffle(father_population)
        np.random.shuffle(mother_population)

        for index in range(len(father_population)):
            father = father_population[index]
            mother = mother_population[index]
            if np.random.rand() < crossover_rate:
                crossover_points = list(range(num_features))
                np.random.shuffle(crossover_points)

                for crossover_point in crossover_points:
                    child1 = np.concatenate((father[:crossover_point], mother[crossover_point:]))
                    if np.count_nonzero(child1) != fixed_num_features:
                        continue
                    else:
                        child2 = np.concatenate((mother[:crossover_point], father[crossover_point:]))
                        break
            else:
                child1 = father
                child2 = mother


            for child in [child1, child2]:
                if np.random.uniform(0, 1) <= mutation_rate:
                    zero = []  
                    one = []  
                    for j in range(num_features):
                        if child[j] == 0:
                            zero.append(j)
                        else:
                            one.append(j)
                    random_selection1 = random.sample(zero, 1)
                    random_selection2 = random.sample(one, 1)
                    child[random_selection1] = 1
                    child[random_selection2] = 0
                    new_population.append(child)
                else:
                    new_population.append(child)

        population = np.array(new_population)

end_time = time.time()
run_time = end_time - start_time




