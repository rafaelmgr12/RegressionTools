import numpy as np
import pandas as pd
from matplotlib.pyplot import rc
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
import sys
import time
from sklearn.metrics import mean_squared_error,make_scorer

msq =make_scorer(mean_squared_error)

class Regression():
    '''
    Executes grid search and cross-validation for many regression models.

    Parameters: 
                models: list of potential regresion models
                grid: grid search parameters

    '''

    def __init__(self, models, grid):
        self.models = models

        # instances only desired models.
        self.grid_of_params = {k: v for k,
                               v in grid.items() if k in self.models}

    def apply_grid_search_pdf(self, X_train, y_train, k=5, n_bins=100):
        self.X_train = X_train
        self.y_train = y_train
        '''
        Parameters: 
                    X_train: 2D ndarray
                    y_train: 1D ndarray                
                    k: cross-validation k-fold. Default: 5.
        '''
        # list of current compatible classifiers
        compatible_classes = [DecisionTreeClassifier(), KNeighborsClassifier(
        ), ExtraTreeClassifier(), RadiusNeighborsClassifier()]

        compatible_classes_str = [str(i) for i in compatible_classes if str(
            i) in self.grid_of_params.keys()]

        self.classificators = [compatible_classes[i].fit(X_train, y_train) for i in range(
            len(compatible_classes)) if str(compatible_classes[i]) in self.grid_of_params.keys()]

        self.model_name = []
        self.accuracies = []
        self.standar_dev = []
        self.best_parameters = []
        self.best_estimators = []
        for i in range(len(self.classificators)):
            start = time.time()
            print("Executing grid search for {}.".format(
                compatible_classes_str[i]))
            grid_search = GridSearchCV(estimator=self.classificators[i],
                                       param_grid=self.grid_of_params[compatible_classes_str[i]],
                                       scoring='accuracy',
                                       cv=k,
                                       n_jobs=-1,
                                       verbose=1)
            grid_search.fit(X_train, y_train)
            self.accuracies.append(grid_search.best_score_)
            self.best_parameters.append(grid_search.best_params_)
            self.best_estimators.append(grid_search.best_estimator_)
            self.standar_dev.append(grid_search.cv_results_[
                                    'std_test_score'][grid_search.best_index_])
            self.model_name.append(compatible_classes_str[i][0:-2])
            end = time.time()
            print("Elapsed time: %.3fs" % (end-start))

    def apply_grid_search(self, X_train, y_train, k=5):
        self.X_train = X_train
        self.y_train = y_train
        '''
        Parameters: 
                    X_train: 2D ndarray
                    y_train: 1D ndarray                
                    k: cross-validation k-fold. Default: 5.
        '''

        # list of current compatible classifiers
        compatible_classes = [DecisionTreeRegressor(), GradientBoostingRegressor(
        ), RandomForestRegressor(), BaggingRegressor(), AdaBoostRegressor()]

        compatible_classes_str = [str(i) for i in compatible_classes if str(
            i) in self.grid_of_params.keys()]

        self.regressors = [compatible_classes[i].fit(X_train, y_train) for i in range(
            len(compatible_classes)) if str(compatible_classes[i]) in self.grid_of_params.keys()]

        self.model_name_reg = []
        self.metrics = []
        self.standar_dev_reg = []
        self.best_parameters_reg = []
        self.best_estimators_reg = []
        for i in range(len(self.regressors)):
            start = time.time()
            print("Executing grid search for {}.".format(
                compatible_classes_str[i]))
            grid_search = GridSearchCV(estimator=self.regressors[i],
                                       param_grid=self.grid_of_params[compatible_classes_str[i]],
                                       scoring=msq,
                                       cv=k,
                                       n_jobs=-1,
                                       verbose=1)
            grid_search.fit(X_train, y_train)
            self.metrics.append(grid_search.best_score_)
            self.best_parameters_reg.append(grid_search.best_params_)
            self.best_estimators_reg.append(grid_search.best_estimator_)
            self.standar_dev_reg.append(grid_search.cv_results_[
                                    'std_test_score'][grid_search.best_index_])
            self.model_name_reg.append(compatible_classes_str[i][0:-2])
            end = time.time()
            print("Elapsed time: %.3fs" % (end-start))

        # XGboost is special...
        if 'XGBRegressor()' in self.grid_of_params.keys():
            start = time.time()
            xgb = XGBRegressor()
            print("Executing grid search for XGBRegressor().")
            grid_search = GridSearchCV(estimator=xgb,
                                       param_grid=self.grid_of_params['XGBRegressor()'],
                                       scoring='neg_mean_squared_error',
                                       cv=k,
                                       n_jobs=-1,
                                       verbose=1)
            grid_search.fit(X_train, y_train)
            self.metrics.append(grid_search.best_score_)
            self.best_parameters_reg.append(grid_search.best_params_)
            self.standar_dev_reg.append(grid_search.cv_results_[
                                    'std_test_score'][grid_search.best_index_])
            self.model_name_reg.append('XGBRegressor')
            end = time.time()
            print("Elapsed time: %.3fs" % (end-start))
            xgb.fit(X_train, y_train)
            self.regressors.append(xgb)
            self.best_estimators_reg.append(grid_search.best_estimator_)

    def show_dataframe_reg(self):
        # zip joins same index tuples of lists
        out = list(zip(self.model_name_reg, self.metrics, self.standar_dev))
        resultsinDataFrame = pd.DataFrame(
            out, columns=['method', 'mean squared error (%)', 'standard deviation (%)'])
        final_df = resultsinDataFrame.sort_values(
            by='mean squared error (%)', ascending=False)
        print(final_df)

    def show_dataframe_pdf(self):
        # zip joins same index tuples of lists
        out = list(zip(self.model_name, self.accuracies, self.standar_dev))
        resultsinDataFrame = pd.DataFrame(
            out, columns=['method', 'mean accuracy (%)', 'standard deviation (%)'])
        final_df = resultsinDataFrame.sort_values(
            by='mean accuracy (%)', ascending=False)
        print(final_df)

    def show_best_parameters(self):
        for i in range(len(self.model_name)):
            print(self.model_name[i], self.best_parameters[i])
        for i in range(len(self.model_name_reg)):
            print(self.model_name_reg[i], self.best_parameters_reg[i])    
    