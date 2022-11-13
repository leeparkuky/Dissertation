import random
import pandas as pd
import numpy as np
from scipy.stats import bernoulli
from itertools import combinations
from pprint import pprint
from tqdm.auto import trange
from numpy.linalg import eig, solve, inv
from numpy import diag
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_gaussian_quantiles


from numba import jit, vectorize, int32, int8

@jit
def dot_product(series, binary):
    output = np.sum(series*binary)
    return output
#     return series.dot(binary.T)


def convert_to_int(series, order_index = None):
    N = len(series)
    series = np.array(series)
    if order_index:
        series = series[order_index]
    binary = np.array([2**(N-1-i) for i in range(N)])
#     integer = series.dot(binary.T)
    integer = dot_product(series, binary)
    return integer


def get_order_index(X, by):
    from numpy.linalg import eig, solve, inv
    from numpy import diag
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.datasets import make_gaussian_quantiles

    if isinstance(by, str):
        cov = EmpiricalCovariance().fit(X)
        cov_X = cov.covariance_
        if by.lower() in ['variance','standard deviation', 'var','std','sd']:
            order_list = [np.where(cov.location_ == x)[0][0] for x in -np.sort(-cov.location_)]
            return order_list
        elif by.lower() in ['covariance','cov']:
            w,v = eig(cov_X)
            sorted_eig = -np.sort(-w)
            sort_index = [np.where(w == x)[0][0] for x in sorted_eig]
            sorted_eigvec = v[:,sort_index]
            del sort_index
            total = sorted_eig.sum()
            thres = 0
            num = 0
            while thres<.8:
                thres += sorted_eig[num]/total
                num += 1
                value = sorted_eig[num]
            selected_eigvec = sorted_eigvec[:, :num-1]
            sorted_eig[sorted_eig<= value] = 0
            N = selected_eigvec.shape[0]
            del thres, num, value, total, sorted_eigvec
            INV = inv(selected_eigvec.T.dot(diag(sorted_eig)).dot(selected_eigvec))
            result1 = diag(np.ones(N)) - selected_eigvec.dot(INV).dot(selected_eigvec.T)
            distance = np.array([result1[:,i].T.dot(result1[:,i]) for i in range(N)])
            distance_index = [np.where(distance == x)[0][0] for x in np.sort(distance)]
            return distance_index

    













class RandomGenerator:
    def __init__(self, p, size, prob_range = (.2, .8), interaction_size = None, use_dask = False):
        self.varnum = p
        self.prob_range = prob_range
        self.N = size
        self.rng = random.Random()
        self.numpy_rng = np.random.default_rng()
        self._config = {'p': p, 'sample size': size, 'interactions': {}}
        self.genBinary
        self.validateInteractionSize(interaction_size = interaction_size)
        self.use_dask = use_dask
        if self.interaction_size:
            self.addInteractionTerms()

        
    def validateInteractionSize(self, interaction_size):
        if interaction_size == None:
            self.interaction_size = self._X.shape[1]
        elif not isinstance(interaction_size, int):
            raise ValueError("interaction_size must be an integer")
        elif interaction_size < 2:
            print("You don't have an interaction in the linear model statement")
            self.interaction_size = 0
        else:
            self.interaction_size = interaction_size

    
    @property
    def genBinary(self):
        p = self.varnum
        prob = [self.rng.uniform(self.prob_range[0], self.prob_range[1]) for x in range(p)]
        self._config['bernoulli parameters'] = {f"X_{i}" : p for i,p in enumerate(prob)}
        dictionary = {f"X_{i}":bernoulli.rvs(probability, size = self.N) for i,probability in enumerate(prob) }
        df = pd.DataFrame(dictionary)
        self._X = df
        return df
    
    @property
    def config(self):
        pprint(self._config)
    
    
    @property
    def interactions(self):
        if self.interaction_size >= 2:
            df = self._X.copy()
            self._interactions = {}
            for r in range(1, self.interaction_size):
                r += 1
                candidates = list(combinations(df.columns, r))
                k = self.rng.randint(0, len(candidates))
                k = k//3 if k > 3 else k
                if k > 0:
                    interactions = self.numpy_rng.choice(candidates, size = k, replace = False)
                    self._interactions[f"{r}-way"] = interactions.tolist()
            self._config['interactions'] = self._interactions
            return(self._interactions)
        else:
            self._interactions = None
            return None
    
    
    def addInteractionTerms(self):
        try:
            df = self._X.copy()
        except:
            df = self.genBinary.copy()
        try:
            interactions = self._interactions
        except:
            interactions = self.interactions

        interaction_all = list(interactions.values())
        N = len(interaction_all)
        for k in trange(N, leave = False):
            interaction_list = interaction_all[k]
            n = len(interaction_list)
            for i in trange(n, leave = False):
                interaction  = interaction_list[i]
                p = len(interaction)
                colname = ''.join(interaction)
                df[colname] = df.loc[:, interaction].apply(np.prod, axis = 1)
        self._X_full = df
        return df
    
    def genCoefficients(self, mean = 2, error = 2):
        try:
            df = self._X_full.copy()
        except:
            df = self.addInteractionTerms().copy()
        columns = df.columns.tolist()
        num_var = len(columns)
        abs_beta = [abs(self.rng.gauss(mean, error)) for x in range(num_var)]
        signs =    [x*2 -1 for x in bernoulli.rvs(.5, size = num_var)]
        beta = [x*y for x,y in zip(abs_beta, signs)]
        self.beta = beta
        dictionary =  dict(zip(columns, beta))
        self._config['coefficients'] = dictionary
        return beta
        
    def genResponse(self, intercept = 5, error = 5):
        beta = np.array(self.genCoefficients())
        raw = self._X_full.copy()
        y = raw.apply(lambda x: beta.dot(x) + self.rng.gauss(intercept, error), axis = 1)
        self.df = self._X.copy()
        self.df['y'] = y
        return self.df
    
    def __call__(self):
        df = self.genResponse()
        return df