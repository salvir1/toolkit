from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

'''
File that contains functions for scoring and comparing boosting models,
See gradient-boost-regressor-pm805.ipnyb for examples
'''
def less_confusing_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return np.array([[tp, fn],[fp, tn]])

def my_mse(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return mse

def my_rmse(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return np.sqrt(mse)

def stage_score_plot(estimator, X_train, y_train, X_test, y_test, ax):
    '''
    Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                X_train: 2d numpy array
                y_train: 1d numpy array
                X_test: 2d numpy array
                y_test: 1d numpy array

    Returns: A plot of the number of iterations vs the MSE for the model for
    both the training set and test set.
    '''
    estimator.fit(X_train, y_train)
    name = estimator.__class__.__name__
    learn_rate = estimator.learning_rate
    # initialize 
    train_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    test_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    # get train score from each boost and plot
    for i, pred in enumerate(estimator.staged_predict(X_train)):
        train_scores[i] = my_mse(y_train, pred)
    plt.plot(train_scores, label = (f'{name} Train {learn_rate} learning rate'), ls = '--')
    # get test score from each boost and plot
    for i, pred in enumerate(estimator.staged_predict(X_test)):
        test_scores[i] = my_mse(y_test, pred)
    ax.plot(test_scores, label = (f'{name} Test {learn_rate} learning rate'))
        

    ax.set_title(name, fontsize=16, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=14)
    ax.set_xlabel('Iterations', fontsize=14)

def compare_cross_val_scores(models, X_train, y_train, X_holdout, y_holdout, nfolds=10):
    for model in models:
        mse = cross_val_score(model, X_train, y_train, 
                          scoring='neg_mean_squared_error',
                          cv=nfolds, n_jobs=-1) * -1
        # mse multiplied by -1 to make positive
        r2 = cross_val_score(model, X_train, y_train, 
                         scoring='r2', cv=nfolds, n_jobs=-1)
        mean_mse = mse.mean()
        mean_r2 = r2.mean()
        name = model.__class__.__name__

        print(f'{name:<30}Train CV   |   MSE: {mean_mse:.5} \
              | R2 {mean_r2:.5}')
        return mean_mse, mean_r2

def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    '''
    Creates a table of parameters for the best fit model
    Parameters:
    ----------
        estimator: the type of model (e.g. RandomForestRegressor())
        parameter_grid: dict of parameters and values to search
        X_train: np.array of train values
        y_train: np.array of train targets
    Print:
    -----
        table of best fit parameters
    Returns:
    -------
        best fit parameters, best fit model
    '''
    model_gridsearch = GridSearchCV(estimator,
                            parameter_grid,
                            n_jobs=-1,
                            verbose=True,
                            scoring='neg_mean_squared_error')
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_
    model_best = model_gridsearch.best_estimator_
    print("\nResult of gridsearch:\n")
    param_header, opt_header, grid_header = "Parameter", "Optimal", "Gridsearch Values"
    print(f"{param_header:<20s} | {opt_header:<8s} | {grid_header}")
    print("-" * 55)
    for param, vals in parameter_grid.items():
        print("{0:<20s} | {1:<8s} | {2}".format(str(param), 
                                                str(best_params[param]),
                                                str(vals)))
    model_start = estimator
    model_start.fit(X_train, y_train)
    model_start.predict
    return best_params, model_best

def compare_test_models(models, X_test, y_test):
    '''
    Parameters: 
        models: dict of models to compare, nick followed by name
        X_test: 2d numpy array
        y_test: 1d numpy array
    Print: 
        mse and r2 for each of the models 
    '''
    print("\nResult of test comparison:\n")
    model_header, MSE_header, R2_header = "Model Name", "MSE", "R2"
    print(f"{model_header:<15s} | {MSE_header:<6s} | {R2_header}")
    print("-" * 35)

    for name, model in models.items():
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)   
        mean_mse = mse.mean()
        mean_r2 = r2.mean()
        print(f"{name:<15s} | {mean_mse:.5} | {mean_r2:.5}")
    