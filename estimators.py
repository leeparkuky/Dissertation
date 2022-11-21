from sklearn.base import BaseEstimator, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics      import r2_score

import numpy as np; import pandas as pd
from typing import List, Union

from utils import *
from Dissertation import convert_to_int




class one_way_ANOVA(BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.response = (x for x in y)
        self.segment_means = pd.DataFrame(zip(X, y), columns = ['x','y']).groupby(
            'x')['y'].mean()
        self._rsq = r2_score(y, np.array([self.segment_means.at[x] for x in X]))
        return self
    
    def fit_transform(self, X, y):
        return self.fit(X, y).predict(X)
    
    def predict(self, X):
        if hasattr(self, 'segment_means'):
            if hasattr(self, '_rsq'):
                delattr(self, '_rsq')
            self.y_pred = (self.segment_means.at[x] for x in X)
            return np.array([self.segment_means.at[x] for x in X])
        else:
            raise ValueError('Fit the model first')
        
    @property
    def score(self):
        if hasattr(self, '_rsq'):
            return self._rsq
        else:
            raise AttributeError("fit the estimator first")
#             y = np.array(list(self.response))
#             y_pred = np.array(list(self.y_pred))
#             return r2_score(y, y_pred)
#             ssto = np.square(y - np.mean(y)).sum()
#             sse  = np.square(y - y_pred).sum()
#             rsq  = (ssto-sse)/ssto
#             self._rsq = rsq
#             return rsq
        
        
class ClusteredSegmentation(BaseEstimator):
    def __init__(self, n_clusters):
        self.clusterer = AgglomerativeClustering()
        self.regressor = one_way_ANOVA()
        self.n_clusters = n_clusters
        
    def find_result_from_RG(self, rg, n_clusters:int = None):
        X = rg._X.loc[:,rg._X.columns.str.contains('X')].apply(convert_to_int, axis = 1).to_numpy().reshape(-1,1)
        y = rg()['y'].to_numpy().reshape(-1,1)
        if n_clusters is None:
            self.n_clusters = rg.config.parameter_size
        else:
            self.n_clusters = n_clusters
        self.fit(X, y)
        self.predict(X)
        return self.simulation_result
        
        
    def full_model(self, X: Union[pd.Series, np.ndarray, List], y: Union[pd.Series, np.ndarray, List], return_array = False):
        segment_means = pd.DataFrame(zip(X, y), columns = ['x','y']).groupby('x')['y'].mean()
        self.y_pred_full = (segment_means.at[x] for x in X)
        self.rsq_full = r2_score(y, np.array([segment_means.at[x] for x in X]))
#         ssto = sum((y - np.mean(y))**2)
#         sse  = sum((y - np.array([segment_means.at[x] for x in X]))**2)
#         self.rsq_full = (ssto-sse)/ssto ############################### define rsq_full <<<<<
        if return_array:
            return np.array([segment_means.at[x] for x in X])
    
        
    def fit(self, X: Union[pd.Series, np.ndarray, List], y: Union[pd.Series, np.ndarray, List]):
        if isinstance(X, np.ndarray):
            X = X.reshape(-1)
        if isinstance(y, np.ndarray):
            y = y.reshape(-1)
        self.n = X.shape[0] ########################################### define n <<<<<
        self.p = 0
        max_group_index = max(np.unique(X))
        while 2**self.p < max_group_index:
            self.p += 1
        self.p = 2**self.p ############################################ define p <<<<<

        self.clusterer_ = clone(self.clusterer)
        self.regressor_ = clone(self.regressor)
        self.clusterer_.set_params(n_clusters = self.n_clusters)
        
        group_id_raw = self.clusterer_.fit_predict(self.full_model(X, y, return_array = True).reshape(-1,1))
        group_id_map = pd.DataFrame(zip(group_id_raw, y), columns = ['x','y']).groupby('x')['y'].mean(
        ).sort_values().reset_index().reset_index().set_index('x')['index']
        
        self.clusterer_.labels_ = np.array([group_id_map.at[x] for x in group_id_raw])
        self.full_to_reduced = pd.DataFrame(zip(X, self.clusterer_.labels_), columns = ['full','reduced']).groupby('full').agg(pd.Series.mode)
        self.q = np.unique(self.clusterer_.labels_).shape[0] ########### define q <<<<<
        self.regressor_.fit(self.clusterer_.labels_, y) # now self.regressor_.segment_means is available
        return self
        
    def predict(self, X: Union[pd.Series, np.ndarray, List]):
        if hasattr(self, "full_to_reduced"):
            if isinstance(X, np.ndarray): # if X is ndarray, flatten
                X = X.reshape(-1)
            transformed_X = self.full_to_reduced.loc[X,'reduced'] # get the id from the clustering
            y_pred = self.regressor_.segment_means.loc[transformed_X] # find the corresponding y_hat value for id
            return y_pred
        else:
            raise AttributeError("You must fit the model first")
    
    def set_params(self, **kwargs):
        if 'n_clusters' in kwargs:
            n_clusters = kwargs['n_clusters']
            self.clusterer = AgglomerativeClustering(n_clusters = n_clusters)
            self.n_clusters = n_clusters
        else:
            raise ValueError("The only parameter for this estimator is 'n_clusters'")
            
    
    
    @property
    def score(self):
        if hasattr(self, "regressor_"):
            rsq_reduced = self.regressor_.score
            return rsq_reduced
        else:
            raise AttributeError("Fit-transform the estimator first")
    
    @property
    def tau_metric(self, alpha = .05):
        if hasattr(self, "regressor_"):
            rsq_reduced = self.regressor_.score
            rsq_full    = self.rsq_full
            n           = self.n
            p           = self.p
            q           = self.q
            return tau(n, p, q, rsq_full, rsq_reduced, alpha)
        else:
            raise AttributeError("Fit-transform the estimator first")

