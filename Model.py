from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import accuracy_score
from joblib import dump

import pandas as pd
from time import time


class TrainInfo:

    def __init__(self, learner, train_time, pred_time, test_score, train_score):
        self.learner = learner
        self.train_time = train_time
        self.pred_time = pred_time
        self.test_score = test_score
        self.train_score = train_score
        self.trained_models = None
        self.training_results = None


class Model:

    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.learners = self.init_classifiers(300)
        self.scorer = accuracy_score

    def init_regressors(self, seed):
        return {
            'AdaBoostRegressor': AdaBoostRegressor(random_state=seed),
            'BaggingRegressor': BaggingRegressor(random_state=seed),
            'ExtraTreesRegressor': ExtraTreesRegressor(random_state=seed),
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=seed),
            'RandomForestRegressor': RandomForestRegressor(random_state=seed),
            'LogisticRegression': LogisticRegression(random_state=seed),
            'PassiveAggressiveRegressor': PassiveAggressiveRegressor(random_state=seed),
            'SGDRegressor': SGDRegressor(random_state=seed),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'MLPRegressor': MLPRegressor(random_state=seed),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=seed),
            'ExtraTreeRegressor': ExtraTreeRegressor(random_state=seed),
        }

    def init_classifiers(self, seed):
        return {
            'AdaBoostClassifier': AdaBoostClassifier(random_state=seed),
            'BaggingClassifier': BaggingClassifier(random_state=seed),
            'ExtraTreesClassifier': ExtraTreesClassifier(random_state=seed),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=seed),
            'RandomForestClassifier': RandomForestClassifier(random_state=seed),
            'LogisticRegression': LogisticRegression(random_state=seed),
            'PassiveAggressiveClassifier': PassiveAggressiveClassifier(random_state=seed),
            'RidgeClassifier': RidgeClassifier(random_state=seed),
            'RidgeClassifierCV': RidgeClassifierCV(),
            'SGDClassifier': SGDClassifier(random_state=seed),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'MLPClassifier': MLPClassifier(random_state=seed),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=seed),
            'ExtraTreeClassifier': ExtraTreeClassifier(random_state=seed),
        }

    ###
    #      This method trains a model.
    #
    #      Args:
    #       learner (classifier / regressor): Model to train.
    #       scorer (function): score function.
    #       X_train: (dataset): Features dataset to train the model.
    #       y_train: (dataset): Targe feature dataset to train the model.
    #       X_test: (dataset): Features dataset to test the model.
    #       y_test: (dataset): Targe feature dataset to test the model.
    #      Returns:
    #       learner (classifier / regressor): trained model.
    #       dfResults (dataset): Dataset with information about the trained model.
    ###
    def train_eval(self, learner):
        start = time()
        learner = learner.fit(self.X_train, self.y_train)
        end = time()

        train_time = end - start

        start = time()
        predictions_test = learner.predict(self.X_test)
        predictions_train = learner.predict(self.X_train)
        end = time()  # Get end time

        pred_time = end - start

        train_score = self.scorer(self.y_train, predictions_train)

        test_score = self.scorer(self.y_test, predictions_test)

        return TrainInfo(learner, train_time, pred_time, test_score, train_score)

    def split_data(self):
        from sklearn.model_selection import train_test_split
        features = self.dataset.drop([self.target], axis=1)
        labels = self.dataset[self.target]
        return train_test_split(features, labels, test_size=0.2, random_state=300, stratify=labels)

    def train_models(self):
        self.trained_models = []

        for learner in list(self.learners.values()):
            train_info = self.train_eval(learner)
            self.trained_models += [train_info]
        self.set_results_df()

    ###
    #      This method use grid search to tune a learner.
    #
    #      Args:
    #       learner (classifier / regressor): learner to tune.
    #       parameters (dict): learner parameters.
    #       X_train: (dataset): Features dataset to train the model.
    #       y_train: (dataset): Targe feature dataset to train the model.
    #       X_test: (dataset): Features dataset to test the model.
    #       y_test: (dataset): Targe feature dataset to test the model.
    #      Returns:
    #       best_learner (classifier / regressor)): Classifier with the best score.
    #       default_score (float): Classifier score before being tuned.
    #       tuned_score (float): Classifier score after being tuned.
    #       cnf_matrix (float): Confusion matrix.
    ###
    def tune_learner(self, learner, parameters):

        labels = self.y_train

        grid_obj = GridSearchCV(learner, param_grid=parameters, scoring=self.scorer, iid=False)
        grid_fit = grid_obj.fit(self.X_train, labels)
        best_learner = grid_fit.best_estimator_
        predictions = (learner.fit(self.X_train, labels)).predict(self.X_test)
        best_predictions = best_learner.predict(self.X_test)

        default_score = self.scorer(self.y_test, predictions)
        tuned_score = self.scorer(self.y_test, best_predictions)

        return best_learner, default_score, tuned_score

    def set_results_df(self):
        self.training_results = pd.DataFrame(columns=['learner', 'train_time', 'pred_time',
                                                      'test_score', 'train_score'])
        for train_info in self.trained_models:
            self.training_results = self.training_results.append(
                {
                    'learner': train_info.learner.__class__.__name__,
                    'train_time': train_info.train_time,
                    'pred_time': train_info.pred_time,
                    'test_score': train_info.test_score,
                    'train_score': train_info.train_score
                }, ignore_index=True)

        self.training_results.sort_values(by='test_score', ascending=False, inplace=True)

    def save_best_model(self, path):
        best_model = self.training_results.head(1)['learner'].values[0]
        best_model = [model for model in self.trained_models if model.learner.__class__.__name__ == best_model]
        best_model = best_model[0]
        dump(best_model.learner, path)
        return best_model

