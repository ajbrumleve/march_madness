import logging
import time

import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, cross_validate


class Classifier:
    def __init__(self, model, xTrain, yTrain, features):
        self.paramGrid = {'bootstrap': [False],
                          'max_depth': [10],
                          'max_features': ['auto'],
                          'min_samples_leaf': [8,9,10,11,12,13,14,15,16],
                          'min_samples_split': [7],
                          'n_estimators': [50]}
        self.model = model
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.features = pd.DataFrame(features)

    def gridSearch(self, paramgrid=None, cv=5, scoring="f1_micro", verbose=2, n_jobs=-1):
        if paramgrid is not None:
            param_grid = paramgrid
        else:
            param_grid = self.paramGrid
        import time
        ts = time.time()
        logging.info("Started grid search")
        # create a grid search object
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring=scoring, verbose=verbose,n_jobs=n_jobs)
        # fit the grid search object to the data
        grid_search.fit(self.xTrain,self.yTrain)
        logging.info(f"Finished grid search in {time.time()-ts}")

        # print the best parameters
        print(grid_search.best_params_)
        # After experimenting, {'bootstrap': False, max_depth': 10, 'max_features': auto,
        # 'min_samples_leaf':13 , 'min_samples_split':7 , 'n_estimators': 50}
        print(grid_search.best_score_)
        self.model = grid_search.best_estimator_

    def RFECVSelect(self, min_features_to_select=5, step=1, n_jobs=-1, scoring="f1_micro", cv=5):
        rfe_selector = RFECV(estimator=self.model, min_features_to_select=min_features_to_select, step=step,
                             n_jobs=n_jobs, scoring=scoring, cv=cv,verbose=2)
        ts = time.time()
        rfe_selector.fit(self.xTrain, self.yTrain)
        self.xTrain = rfe_selector.transform(self.xTrain.copy())
        rfe_selector.get_support() #new_vector = [numWins, avgPointsScored, avgPointsAllowed, checkPower6Conference(team_id),
        #                                         avgAssists, avgTurnovers, tournamentSeed, getTourneyAppearances(team_id),
        #                                         totalPoss, totalfgmPerPoss, totalftmPerPoss, totaldrPerPoss, totalastPerPoss]

        logging.info(f"Finished RFECV in {time.time()-ts}")
        return rfe_selector

    def analyze_model(self):
        accuracy = {}

        _scoring = ['accuracy', 'precision', 'recall', 'f1']

        results = cross_validate(estimator=self.model,
                                   X=self.xTrain,
                                   y=self.yTrain,
                                   cv=5,
                                   scoring=_scoring,
                                   return_train_score=True)
        accuracy[type(self.model).__name__+"trainf1"]=results['train_f1']
        accuracy[type(self.model).__name__+"testf1"]=results['test_f1']
        accuracy[type(self.model).__name__+"train_precision"]=results['train_precision']
        accuracy[type(self.model).__name__+"test_precision"]=results['test_precision']
        accuracy[type(self.model).__name__+"train_recall"]=results['train_recall']
        accuracy[type(self.model).__name__+"test_recall"]=results['test_recall']
        print(f"The difference from train to test f1 is {results['train_f1'].mean()-results['test_f1'].mean()}")
        print(f"The difference from train to test precision is {results['train_precision'].mean()-results['test_precision'].mean()}")
        print(f"The difference from train to test recall is {results['train_recall'].mean()-results['test_recall'].mean()}")
        return accuracy
