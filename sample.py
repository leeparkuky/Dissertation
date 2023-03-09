from dataclasses import dataclass
from typing import List, Union, Tuple
import numpy as np
from scipy.stats import bernoulli, norm
import pandas as pd
from functools import partial
from itertools import combinations, product
from math import comb, sqrt

@dataclass
class sample_generator:
    p: int
    sample_size: int
    num_interactions: int = None
    rng: np.random._generator.Generator = np.random.default_rng()
    beta_range: Tuple[Union[int, float]] = (3, 8)
    pi_range: Tuple[Union[float]] = (.2, .8)
    error_scale: float = sqrt(5)
        
    @property
    def interactions(self):
        if hasattr(self, '_interactions'):
            pass
        else:
            beta = [f'beta_{i}' for i in range(1, self.p+1)]
            beta_interaction_coef = []
            if self.num_interactions:
                assert self.num_interactions < 2**self.p -1 - self.p
            else:
                self.num_interactions = int(self.rng.uniform(0, 2**self.p - 1 - self.p))
            r = self.num_interactions 
            size_total_int = 2**self.p -1 - self.p
            chosen_index = np.sort(self.rng.choice(range(size_total_int), r, replace = False))
            chosen_interactions = []
            i = 0
            for k in range(2, self.p + 1):
                chosen_interactions += ['*'.join(beta_name) for idx, beta_name in \
                                        zip(range(i, i+comb(self.p, k)), combinations(beta, k)) \
                                        if idx in chosen_index.tolist() ]
                i += comb(self.p, k)
            chosen_index = [1 for _ in range(self.p + 1)] + \
            [1 if x in chosen_index.tolist() else 0 for x in range(size_total_int)]
            self.interactions_coef = [x if self.rng.random() < .5 else -x for x \
                                      in self.rng.uniform(low=self.beta_range[0], high=self.beta_range[1], size=r)]
            self.beta_effective = chosen_index
            self._interactions = {x.replace('beta_', 'X'):val for x, val in zip(chosen_interactions, self.interactions_coef)}
        return self._interactions
        
    @property
    def beta(self):
        if hasattr(self, '_beta'):
            pass
        else:
            self._beta = [x if self.rng.random() < .5 else -x for x in self.rng.uniform(low=self.beta_range[0], high=self.beta_range[1], size=self.p+1)]
        return {f'beta_{i}': v for i,v in enumerate(self._beta)}
    
    @property
    def beta_names(self):
        if hasattr(self, '_beta_names'):
            pass
        else:
            beta_raw_names = [f'beta{i}' for i in range(1, self.p + 1)]
            beta_names = [f'beta{i}' for i in range(self.p + 1)]
            for r in range(2, self.p):
                beta_names += ['*'.join(x) for x in combinations(beta_raw_names, r)]
            beta_names += ['*'.join(beta_raw_names)]
            self._beta_names = beta_names
            del beta_names
            del beta_raw_names
        return self._beta_names
    
    @property
    def pi(self):
        if hasattr(self, '_pi'):
            pass
        else:
            self._pi = self.rng.uniform(low = self.pi_range[0], high = self.pi_range[1], size = self.p)
        return {f'pi_{i+1}': v for i,v in enumerate(self._pi)}
    
    @staticmethod
    def x(p, n):
        return bernoulli.rvs(p, size= n)
    
    @property
    def X(self):
        if hasattr(self, '_X'):
            pass
        else:
            self._X = pd.DataFrame({f'X{i+1}':self.x(pi, self.sample_size) for i, pi in enumerate(self.pi.values())}).astype(np.ubyte)
        return self._X
        
        
    @property
    def y(self):
        if hasattr(self, '_y'):
            pass
        else:
            beta = self.beta
            interactions = self.interactions
            y = []
            
            def find_interaction_effect(row, interactions = self.interactions):
                output = 0
                for key, val in interactions.items():
                    output += row[key.split('*')].prod() * val
                return output
            
            
            for idx, row in self.X.iterrows():
                main = beta['beta_0'] + sum([x*y for x,y in zip(list(beta.values())[1:], row)])
                interaction_eff = find_interaction_effect(row)
                error = norm.rvs(scale = self.error_scale)
                y.append(main + interaction_eff + error)
            
            self._y = np.array(y).reshape(-1,1)
        return self._y
            
            
        
# for idx, row in rng.X.iterrows():
#     print(rng.beta['beta_0'] + sum([x*y for x,y in zip(list(rng.beta.values())[1:], row)]))        
        
        
        
    @property
    def barcode(self):
        if hasattr(self, '_barcode'):
            pass
        else:
            self._barcode = self.gen_barcode(self.X)
        return self._barcode
    
    @staticmethod
    def gen_barcode(X):
        pack_bits = np.packbits(np.array(X), axis = -1)
        if pack_bits.shape[1]>1:
            def gen_barcode(a, k, return_float = False):
                m = len(a)-1
                total_sum = 0
                for i, x in enumerate(a):
                    if i < m:
                        if return_float:
                            adjust = (x * 2**(m-i-1) * 2**k)
                            total_sum += float(adjust)
                        else:
                            total_sum += (x * 2**(m-i-1) << k)
                    else:
                        total_sum += (x >> (8-k))
                return total_sum
            if X.shape[1] > 500:
                barcode = partial(gen_barcode, k = X.shape[1]%8, return_float = True)
            else:
                barcode = partial(gen_barcode, k = X.shape[1]%8)
        else:
            def adjust_barcode(a, k):
                a = a >> k
                return a
            barcode = partial(adjust_barcode, k = 8-X.shape[1])
        output = np.apply_along_axis(barcode, 1, pack_bits)
        return output

    @staticmethod
    def barcode_to_beta(barcode):
        if isinstance(barcode, list):
            output = [1] + barcode
        else:
            output = [1] + list(barcode)
        N = len(output)
        for i in range(2, N):
            output += [np.prod(x) for x in combinations(barcode, i)]
    #     output += [np.prod(output)]
        return output
    
    @property
    def L(self):
        if hasattr(self, '_L'):
            pass
        else:
            all_sets = list(set(product([0,1], repeat = self.p))); all_sets.sort()
            self._L = np.array([self.barcode_to_beta(x) for x in all_sets]).T
        return self._L

            
# Just for a simulation ####################################################################################
################################################################################################################
########################################################v########################################################


@dataclass
class simple_sample_generator:
    p: int
    sample_size: int
    num_interactions: int = None
    rng: np.random._generator.Generator = np.random.default_rng()
    beta_range: Tuple[Union[int, float]] = (1,1)
    pi_range: Tuple[Union[float]] = (.5,.5)
    error_scale: float = sqrt(1)
        
    @property
    def interactions(self):
        if hasattr(self, '_interactions'):
            pass
        else:
            beta = [f'beta_{i}' for i in range(1, self.p+1)]
            beta_interaction_coef = []
            if self.num_interactions:
                assert self.num_interactions < 2**self.p -1 - self.p
            else:
                self.num_interactions = int(self.rng.uniform(0, 2**self.p - 1 - self.p))
            r = self.num_interactions 
            size_total_int = 2**self.p -1 - self.p
            chosen_index = np.sort(self.rng.choice(range(size_total_int), r, replace = False))
            chosen_interactions = []
            i = 0
            for k in range(2, self.p + 1):
                chosen_interactions += ['*'.join(beta_name) for idx, beta_name in \
                                        zip(range(i, i+comb(self.p, k)), combinations(beta, k)) \
                                        if idx in chosen_index.tolist() ]
                i += comb(self.p, k)
            chosen_index = [1 for _ in range(self.p + 1)] + \
            [1 if x in chosen_index.tolist() else 0 for x in range(size_total_int)]
            self.interactions_coef = [x for x \
                                      in self.rng.uniform(low=self.beta_range[0], high=self.beta_range[1], size=r)]
            self.beta_effective = chosen_index
            self._interactions = {x.replace('beta_', 'X'):val for x, val in zip(chosen_interactions, self.interactions_coef)}
        return self._interactions
        
    @property
    def beta(self):
        if hasattr(self, '_beta'):
            pass
        else:
            self._beta = [x for x in self.rng.uniform(low=self.beta_range[0], high=self.beta_range[1], size=self.p+1)]
        return {f'beta_{i}': v for i,v in enumerate(self._beta)}
    
    @property
    def beta_names(self):
        if hasattr(self, '_beta_names'):
            pass
        else:
            beta_raw_names = [f'beta{i}' for i in range(1, self.p + 1)]
            beta_names = [f'beta{i}' for i in range(self.p + 1)]
            for r in range(2, self.p):
                beta_names += ['*'.join(x) for x in combinations(beta_raw_names, r)]
            beta_names += ['*'.join(beta_raw_names)]
            self._beta_names = beta_names
            del beta_names
            del beta_raw_names
        return self._beta_names
    
    @property
    def pi(self):
        if hasattr(self, '_pi'):
            pass
        else:
            self._pi = self.rng.uniform(low = self.pi_range[0], high = self.pi_range[1], size = self.p)
        return {f'pi_{i+1}': v for i,v in enumerate(self._pi)}
    
    @staticmethod
    def x(p, n):
        return bernoulli.rvs(p, size= n)
    
    @property
    def X(self):
        if hasattr(self, '_X'):
            pass
        else:
            self._X = pd.DataFrame({f'X{i+1}':self.x(pi, self.sample_size) for i, pi in enumerate(self.pi.values())}).astype(np.ubyte)
        return self._X
        
        
    @property
    def y(self):
        if hasattr(self, '_y'):
            pass
        else:
            beta = self.beta
            interactions = self.interactions
            y = []
            
            def find_interaction_effect(row, interactions = self.interactions):
                output = 0
                for key, val in interactions.items():
                    output += row[key.split('*')].prod() * val
                return output
            
            
            for idx, row in self.X.iterrows():
                main = beta['beta_0'] + sum([x*y for x,y in zip(list(beta.values())[1:], row)])
                interaction_eff = find_interaction_effect(row)
                error = norm.rvs(scale = self.error_scale)
                y.append(main + interaction_eff + error)
            
            self._y = np.array(y).reshape(-1,1)
        return self._y
            
            
        
# for idx, row in rng.X.iterrows():
#     print(rng.beta['beta_0'] + sum([x*y for x,y in zip(list(rng.beta.values())[1:], row)]))        
        
        
        
    @property
    def barcode(self):
        if hasattr(self, '_barcode'):
            pass
        else:
            self._barcode = self.gen_barcode(self.X)
        return self._barcode
    
    @staticmethod
    def gen_barcode(X):
        pack_bits = np.packbits(np.array(X), axis = -1)
        if pack_bits.shape[1]>1:
            def gen_barcode(a, k, return_float = False):
                m = len(a)-1
                total_sum = 0
                for i, x in enumerate(a):
                    if i < m:
                        if return_float:
                            adjust = (x * 2**(m-i-1) * 2**k)
                            total_sum += float(adjust)
                        else:
                            total_sum += (x * 2**(m-i-1) << k)
                    else:
                        total_sum += (x >> (8-k))
                return total_sum
            if X.shape[1] > 500:
                barcode = partial(gen_barcode, k = X.shape[1]%8, return_float = True)
            else:
                barcode = partial(gen_barcode, k = X.shape[1]%8)
        else:
            def adjust_barcode(a, k):
                a = a >> k
                return a
            barcode = partial(adjust_barcode, k = 8-X.shape[1])
        output = np.apply_along_axis(barcode, 1, pack_bits)
        return output

    @staticmethod
    def barcode_to_beta(barcode):
        if isinstance(barcode, list):
            output = [1] + barcode
        else:
            output = [1] + list(barcode)
        N = len(output)
        for i in range(2, N):
            output += [np.prod(x) for x in combinations(barcode, i)]
    #     output += [np.prod(output)]
        return output
    
    @property
    def L(self):
        if hasattr(self, '_L'):
            pass
        else:
            all_sets = list(set(product([0,1], repeat = self.p))); all_sets.sort()
            self._L = np.array([self.barcode_to_beta(x) for x in all_sets]).T
        return self._L


                