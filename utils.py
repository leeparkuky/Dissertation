# basic imports
from typing import Union, Dict, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
# scipy and sklearn
from scipy.optimize import bisect
from scipy.stats import f, ncf
from sklearn.tree import DecisionTreeRegressor









@dataclass
class test_result():
    ccp_alpha: List[float] = None
    tau_estimates: List[float] = None
    tau_estimates_lowerbound: List[float] = None
    model_parameters: List[int] = None
    
    def append_result(self, **kwargs):
        if all(key in kwargs for key in ['ccp_alpha', 'tau_estimate','tau_lower_bound','parameter_dimension_reduced']):
            if self.tau_estimates == None:
                self.tau_estimates = [kwargs['tau_estimate']]; self.tau_estimates_lowerbound = [kwargs['tau_lower_bound']]
                self.model_parameters = [kwargs['parameter_dimension_reduced']]; self.ccp_alpha = [kwargs['ccp_alpha']]
            else:
                self.tau_estimates.append(kwargs['tau_estimate'])
                self.tau_estimates_lowerbound.append(kwargs['tau_lower_bound'])
                self.model_parameters.append(kwargs['parameter_dimension_reduced'])
                self.ccp_alpha.append(kwargs['ccp_alpha'])
        else:
            raise ValueError("**kwargs must contain all of 'tau_estimate','tau_lower_bound','parameter_dimension_reduced'")

    def __add__(self, result2):
        ccp_alpha = self.ccp_alpha + result2.ccp_alpha
        tau_estimates = self.tau_estimates + result2.tau_estimates
        tau_estimates_lowerbound = self.tau_estimates_lowerbound + result2.tau_estimates_lowerbound
        model_parameters = self.model_parameters + result2.model_parameters
        return test_result(ccp_alpha, tau_estimates, tau_estimates_lowerbound, model_parameters)
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    
    def __len__(self):
        return len(self.tau_estimates)        
        
        
        
@dataclass
class tau:
    n: int
    p: int
    q: int
    r_sqf: float
    r_sqr: float
    alpha: float = 0.05
    

    def find_LB(self):
        tau_est = (self.r_sqf - self.r_sqr)/(1-self.r_sqf)
        dfn = self.p-self.q
        dfd = self.n-self.p
        self.tau_est = tau_est
        self.dfn = dfn
        self.dfd = dfd
        def survival(loc):
            return ncf.sf(tau_est*dfd/dfn, dfn, dfd, loc) - self.alpha
        self.tau_lb = bisect(survival, 0, tau_est*self.n)/self.n
        return self.tau_lb
    
    
@dataclass
class tree_fit_result:
    ccp_alpha: float
    barcode_length: int
    alpha: float
    rsq_full: float = None
    
    @staticmethod
    def gen_all_barcodes(k, barcode_type:str):
        if barcode_type.lower() in ['raw','binary']:
            barcodes = list(product([0, 1], repeat=k))
            return barcodes
        elif barcode_type.lower() in ['decimal','integer']:
            decimal_barcodes = np.arange(2**k).reshape(-1,1)
            return decimal_barcodes
        else:
            raise ValueError("barcode_type can be either binary or decimal")
    
    def full_model(self, X:np.array, y:np.array):
        df = pd.DataFrame(zip(X.reshape(-1), y.reshape(-1)), columns = ['x','y'])
        segment_means = df.groupby('x').mean()
        df['pred'] = df.x.apply(lambda x: segment_means.at[x, 'y'])
        sse = sum((df.y-df.pred)**2)
        ssto = sum((df.y - df.y.mean())**2)
        rsq_full = (ssto - sse)/ssto
        del df, segment_means
        self.rsq_full = rsq_full
    
    def fit_reduced_model(self, X:np.array, y:np.array, return_result = True)->Union[None, tuple[float, int]]:
        self.full_model(X, y)
        self.n = X.shape[0]
        reg = DecisionTreeRegressor(ccp_alpha = self.ccp_alpha)
        reg.fit(X, y)
        r_sq = reg.score(X, y)
        test = self.gen_all_barcodes(self.barcode_length, 'decimal')
        self.p = test.shape[0]
        result = reg.predict(test)
        num_groups = len(np.unique(result))
        self.q = num_groups
        self.rsq_reduced = r_sq
        if return_result:
            return r_sq, num_groups
        else:
            pass
    
    def find_tau_LB(self):
        self.t = tau(self.n, self.p, self.q, self.rsq_full, self.rsq_reduced, self.alpha)
        return self.t.find_LB()
            
    def __call__(self, X:np.array, y:np.array, return_result = True)-> Union[None, Dict[str, Union[int, float]]]:
        self.fit_reduced_model(X, y, return_result = False)
        self.tau_est_lb = self.find_tau_LB()
        self.tau_est = self.t.tau_est
        if return_result:
            return {"parameter_dimension_full": self.p,
                    "parameter_dimension_reduced": self.q,
                   "r_squared_full": self.rsq_full,
                   "r_squared_reduced": self.rsq_reduced,
                   "tau_estimate": self.tau_est,
                   "tau_lower_bound": self.tau_est_lb}
    