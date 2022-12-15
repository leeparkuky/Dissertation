# import pandas
import pandas as pd
# import scipy
from scipy.stats import bernoulli
from scipy.stats import f, ncf
# import numpy
import numpy as np
from numpy.linalg import eig, solve, inv
from numpy import diag
# import sklearn
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_gaussian_quantiles
from sklearn.tree import DecisionTreeRegressor

# import dask
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client

# import others
from functools import partial
from itertools import combinations
from pprint import pprint
from tqdm.auto import trange
import random
import os

# import numba
from numba import jit, vectorize, int32, int8

# import utils.py
from utils import GeneratorConfig


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
            distance = np.array([result1[i,:].T.dot(result1[i,:]) for i in range(N)])
#             distance = np.array([result1[:,i].T.dot(result1[:,i]) for i in range(N)])
            distance_index = [np.where(distance == x)[0][0] for x in np.sort(distance)]
            return distance_index







class RandomGenerator:
    def __init__(self, p, size, order_by = 'cov', prob_range = (.2, .8), interaction_size = None, use_dask = False, verbose = True):
        self.varnum = p
        self.prob_range = prob_range
        self.N = size
        self.rng = random.Random()
        self.numpy_rng = np.random.default_rng()
        self._config = {'p': p, 'sample_size': size, 'interactions': {}}
        self.verbose = verbose
        if order_by:
            self.genBinary(order = True, by = order_by)
        else:
            self.genBinary(order = False)
        self.validateInteractionSize(interaction_size = interaction_size)
        self.use_dask = use_dask
        if use_dask:
            client = Client(n_workers = os.cpu_count(), threads_per_worker = 2)
            self.dashboard_link = client.dashboard_link

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

    def reorder(self, by):
        columns = self._X.columns[self._X.columns.str.contains('X')]
        columns2 = self._X.columns[~self._X.columns.str.contains('X')].tolist()

        if (by == None) or (by in ['random','Random','none']):
            columns = self.rng.sample(columns.tolist(), len(columns))
            self._X = self._X.loc[:, columns + columns2]
        else:
            order = get_order_index(self._X.loc[:, columns], by = by)
            columns_in_order = columns[order].tolist()
            self._X = self._X.loc[:, columns_in_order + columns2]
        return self._X
    
    def genBinary(self, order = False, by = 'cov'):
        p = self.varnum
        verbose = self.verbose
        prob = [self.rng.uniform(self.prob_range[0], self.prob_range[1]) for x in range(p)]
        self._config['bernoulli_parameters'] = {f"X_{i}" : p for i,p in enumerate(prob)}
        dictionary = {f"X_{i}":bernoulli.rvs(probability, size = self.N) for i,probability in enumerate(prob) }
        df = pd.DataFrame(dictionary)
        if order:
            df = self.reorder_data(df, by = by)
            if verbose:
                print(f"dataset is ordered by {by}")
        self._X = df.astype(np.byte)
        return df
    
    @staticmethod
    def reorder_data(data, by, columns = None):
        if isinstance(data, pd.DataFrame):
            if columns:
                X = data.loc[:, columns]
            else:
                columns = data.columns
                X = data
            order_index = get_order_index(X, by = by)
            new_columns = columns[order_index]
            data.loc[:, columns] = X.loc[:, new_columns].to_numpy()
            return data
        else:
            raise ValueError("data must be a pandas dataframe")
        
    
    
    
    @property
    def config(self):
        self._config['parameter_size'] = self._config['p'] + sum([len(v) for k, v in self._config['interactions'].items()]) + 1
        return GeneratorConfig(**self._config)
    
    
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
    
    @staticmethod
    def compute_response(series, coefficients):
        index_1 = series.index[series.eq(1)].tolist()
        y = 0
        for i in range(1, len(index_1)+1):
            if i == 1:
                for idx in index_1:
                    y += coefficients[idx]
            else:
                interactions = [''.join(x) for x in list(combinations(index_1, i))]
                for term in interactions:
                    try:
                        y += coefficients[term]
                    except:
                        pass
        return y
    
    
    def addInteractionTerms(self, use_dask = False):
        verbose = self.verbose
        try:
            df = self._X
        except:
            df = self.genBinary
        try:
            interactions = self._interactions
        except:
            interactions = self.interactions
        interaction_all = list(interactions.values())
        N = len(interaction_all)
        colnames = df.columns.tolist()
        if verbose:
            for k in trange(N, leave = False):
                interaction_list = interaction_all[k]
                n = len(interaction_list)
                for i in trange(n, leave = False):
                    interaction  = interaction_list[i]
                    p = len(interaction)
                    colname = ''.join(interaction)
                    colnames.append(colname)
        else:
            for k in range(N):
                interaction_list = interaction_all[k]
                n = len(interaction_list)
                for i in range(n):
                    interaction  = interaction_list[i]
                    p = len(interaction)
                    colname = ''.join(interaction)
                    colnames.append(colname)
        self.columns = colnames
        return colnames
    
    def genCoefficients(self, mean = 2, error = 2): # This later needs a decorator
        columns = self.columns
        num_var = len(columns)
        abs_beta = [abs(self.rng.gauss(mean, error)) for x in range(num_var)]
        signs =    [x*2 -1 for x in bernoulli.rvs(.5, size = num_var)]
        beta = [x*y for x,y in zip(abs_beta, signs)]
        self.beta = beta
        dictionary =  dict(zip(columns, beta))
        self._config['coefficients'] = dictionary
        return beta
        
    def genResponse(self, intercept = None, error = None, use_dask = False, save_parquet = None):
        if intercept == None:
            intercept = self.rng.uniform(3, 10)
        if error == None:
            error = self.rng.uniform(3, 8)
        try:
            self._config['coefficients']['intercept'] = intercept
            self._config['sigma'] = error
        except:
            mean, coef_error = self.rng.uniform(3, 8), self.rng.uniform(3, 8)
            self.genCoefficients(mean = self.rng.uniform(3, 8), error = coef_error)
            self._config['coefficient_generator_config'] = {'distribution':'Gauss',
                                                            'mean' : mean,
                                                            'error': coef_error}
            self._config['coefficients']['intercept'] = intercept
            self._config['sigma'] = error
            
        if use_dask:
            try:
                df = self._X
            except:
                df = self.genBinary
            try:
                interactions = self._interactions
            except:
                interactions = self.interactions
            ddf = dd.from_pandas(df.astype(np.byte), chunksize = 1000)
            interaction_dict = interactions
            interactions = []
            for val in interaction_dict.values():
                interactions += val
            coef = self._config['coefficients']
                
            find_y = partial(self.compute_response, coefficients = coef)
            result = ddf.apply(find_y, axis = 1, meta = (None, 'float64'))
            result += da.random.normal(loc = intercept, scale = error, size = df.shape[0], chunks = 1000)
            ddf['y'] = result
            if save_parquet:
                dd.to_parquet(df=ddf,path=save_parquet)
                del ddf
                return None
            else:
                self.df = ddf.compute()
                return self.df
        else:
            try:
                df = self._X
            except:
                df = self.genBinary
            try:
                interactions = self._interactions
            except:
                interactions = self.interactions
            interaction_dict = interactions
            interactions = []
            for val in interaction_dict.values():
                interactions += val
            coef = self._config['coefficients']
                
            find_y = partial(self.compute_response, coefficients = coef)
            result = df.apply(find_y, axis = 1)
            result += np.random.normal(loc = intercept, scale = error, size = df.shape[0])
            df['y'] = result
            if save_parquet:
                df.to_parquet(path=save_parquet)
                del df
                return None
            else:
                self.df = df
                return self.df
            
            
            
            
            
            
            
            beta = np.array(self.genCoefficients())
            raw = self._X_full
            y = raw.apply(lambda x: beta.dot(x) + self.rng.gauss(intercept, error), axis = 1)
            self.df = self._X
            self.df['y'] = y
            return self.df
    
    def __call__(self, parquet_filename = None):
        if self.use_dask:
            print(self.dashboard_link)
        if parquet_filename:
            self.genResponse(use_dask = self.use_dask, save_parquet = parquet_filename)
            del self._X
            return None
        else:
            df = self.genResponse(use_dask = self.use_dask)
            return df

        
        
        
        
