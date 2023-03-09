#python default packages
from dataclasses import dataclass
from itertools import product
from functools import partial
#pandas
import pandas as pd
#numpy and scipy
import numpy as np
from numpy.linalg import norm, inv
from scipy.stats import f

# within dissertation2 
import sample
import estimators
#joblib
from joblib import Parallel, delayed



@dataclass
class cluster_barcode_scanner:
    sample_generator: sample.sample_generator
    cluster_estimator: estimators.ClusteredSegmentation
    multi_processing: bool = True
        
    @property
    def C(self):
        if hasattr(self, '_C'):
            pass
        else:
            def gen_contrast(groupby_series, p):
                if len(groupby_series) == 1:
                    return None
                else:
                    result = []
                    for idx, mean_idx in enumerate(groupby_series.values.tolist()):
                        if idx == 0:
                            array_1 = np.array([1 if x == mean_idx else 0 for x in range(2**p)])
                        else:
                            array_2 = np.array([[1 if x == mean_idx else 0 for x in range(2**p)]])
                            contrast = (array_1 - array_2).reshape(1, -1)
                            result.append(contrast)
                    return np.concatenate(result, axis = 0)
            result = self.cluster_estimator.full_to_reduced_with_counts.copy()
            C = np.concatenate([x for x in result.groupby('reduced')['full'].apply(gen_contrast, p = self.sample_generator.p) if x is not None ], axis = 0)
            self._C = C
        return self._C


    @property
    def L_inv(self):
        if hasattr(self, '_L_inv'):
            pass
        else:
            L = self.sample_generator.L
            self._L_inv = inv(L)
        return self._L_inv
    
    @property
    def full_var(self):
        if hasattr(self, '_full_var'):
            pass
        else:
            self._full_var = self.cluster_estimator.full_var
        return self._full_var
        
    @staticmethod
    def get_single_distance(beta, L_inv, C, orth = False):
        """ given a contrast for beta, it computes the normalized mahalanobis distance between 
        L_inv*beta and I - P_c where P_c is a projection matrix with respect to C"""
        if isinstance(beta, list):
            beta = np.array(beta)
        assert max(beta.shape) == L_inv.shape[1]
        beta = beta.reshape(-1)
        ctc_inv = np.linalg.inv(C.dot(C.T))
        if orth:
            H = C.T.dot(ctc_inv).dot(C)
        else:
            H = (np.diag(np.ones(max(C.shape))) - C.T.dot(ctc_inv).dot(C))
        l_inv_b = L_inv.dot(beta)
        denom = np.array(l_inv_b.dot(l_inv_b))
        num   = np.array(l_inv_b.T.dot(H).dot(l_inv_b))
        try:
            output = num/denom
            return output
        except:
            return 0
    
    @staticmethod
    def get_cdf(beta_contrasts, L_inv, full_var, group_means, N, p):
        c = beta_contrasts #self.gen_beta_contrasts(beta)
        assert c.shape[0] <= c.shape[1]
        LC = L_inv.dot(c.T)
        mu = LC.T.dot(group_means) #self.cluster_estimator.full_to_reduced_with_counts.y
        var = LC.T.dot(full_var).dot(LC)
        f_value = mu.T.dot(inv(var)).dot(mu)/c.shape[0]
        cdf = f.cdf(f_value, c.shape[0], N-p, loc=0, scale=1)
        return cdf    
    
    
    @property
    def cdf_table(self):
        if hasattr(self, '_cdf_table'):
            pass
        else:
            all_beta = self.all_available_beta_for_test
            L_inv = self.L_inv
            full_var = self.full_var
            # fill the missing grouping
            # This part needs to be replaced with alternative methods flling the missing barcode
            # we used the overall mean to impute the missing groups mean
            if self.cluster_estimator.full_to_reduced_with_counts.shape[0] < self.cluster_estimator.full_p:
                max_group_id = self.cluster_estimator.full_to_reduced_with_counts.reduced.max()
                df = pd.DataFrame(range(self.cluster_estimator.full_p), columns = ['full'])
                df = df.merge(self.cluster_estimator.full_to_reduced_with_counts, how = 'left')
                df.counts.fillna(0, inplace = True); df.reduced.fillna(max_group_id + 1, inplace = True)
                df.y.fillna(self.cluster_estimator.overall_mean, inplace = True)
            else:
                df = self.cluster_estimator.full_to_reduced_with_counts
            self.cp = df
            group_means = df.y
            N = self.cluster_estimator.n
            p = self.cluster_estimator.full_p
            
            if self.multi_processing:
                cdf_func = partial(self.get_cdf, L_inv = L_inv, full_var = full_var, group_means = group_means,
                                   N = N, p = p)
                cdf = Parallel(n_jobs = -1, prefer = 'threads')(delayed(cdf_func)(self.gen_beta_contrasts(beta)) for beta in all_beta)
            else:
                cdf = []
                for beta in all_beta:
                    c = self.gen_beta_contrasts(beta)
                    cdf_value = self.get_cdf(c, L_inv, full_var, group_means, N, p)
                    cdf.append(cdf_value)
            df_list = []
            for beta, p_val in zip(all_beta, cdf):
                seq = list(beta) + [p_val]
                df_list.append(seq)
            df = pd.DataFrame(df_list, columns = self.sample_generator.beta_names + ['cdf'])
            columns = df.columns[df.columns.str.contains('\*')].tolist() + ['cdf']
            df = df.loc[:, columns].sort_values(['cdf'], ascending = False).reset_index(drop = True)
            df.cdf.fillna(0, inplace = True)
            self._cdf_table = df
        return self._cdf_table
        
        
    @property
    def cdf_ranking(self):
        if hasattr(self, '_cdf_ranking'):
            pass
        else:
            df = self.cdf_table
            df = df.loc[df.cdf.gt(0), :].reset_index(drop = True)
            result = df.apply(np.average, weights = df.cdf, axis = 0).sort_values(ascending = False)
            result.pop('cdf')
            ranking = pd.DataFrame(zip(result.index.tolist(), result), columns = ['coefficients', 'score'])
            ranking['ranking'] = ranking.score.rank(ascending = False, method = 'min')
            self._cdf_ranking = ranking
            del result, ranking
        return self._cdf_ranking
    
    @property
    def distance_table(self):
        if hasattr(self, '_distance_table'):
            pass
        else:
            all_beta = self.all_available_beta_for_test
#             avg_distance = []
            def return_distance(beta, L_inv, C):
                beta = self.gen_beta_contrasts(beta)
                dist = []
                for c in beta:
                    dist.append(self.get_single_distance(c, L_inv, C))
                return np.mean(dist)
            
            def gen_C_distance(all_beta, orth = False):
                distance = partial(return_distance, L_inv = self.L_inv, C = self.C)
                ###################################################################################################
                if self.multi_processing:
                    avg_distance = Parallel(n_jobs=-1, prefer = 'threads')(delayed(distance)(beta) for beta in all_beta) 
                else:
                    avg_distance = []
                    for beta in all_beta:
                        dist = []
                        beta = self.gen_beta_contrasts(beta)
                        for c in beta:
                            dist.append(self.get_single_distance(c, self.L_inv, self.C, orth = orth))
                        avg_distance.append(np.mean(dist))
                return avg_distance
            avg_distance = gen_C_distance(all_beta)
            df_list = []
            for beta, dist in zip(all_beta, avg_distance):
                seq = list(beta) + [dist]
                df_list.append(seq)
            df = pd.DataFrame(df_list, columns = self.sample_generator.beta_names + ['distance'])
            columns = df.columns[df.columns.str.contains('\*')].tolist() + ['distance']
            df = df.loc[:, columns].sort_values(['distance'], ascending = False).reset_index(drop = True)
            try:
                base_line = df.loc[df.sum(axis = 1)-df.distance == 0,'distance'].values[0]
#             base_line_2 = df.loc[df.sum(axis = 1)-df.distance == 0, 'orth_dist'].values[0]
                if df.distance.gt(base_line).sum():
                    self._distance_table = df.loc[df.distance.gt(base_line),:]
                else:
                    self._distance_table = df #if all the distance is less than the base_line, return first 25%
            except:
                self._distance_table = df
            del df, df_list, avg_distance, seq
        return self._distance_table
    
    
    def get_distance_ranking(self, percentile = .75, normalize = True):
        if (1-percentile)*self.distance_table.shape[0] <2:
            N = 2 # at least two lines
        else:
            N = int((1-percentile)*self.distance_table.shape[0])
        df = self.distance_table
        new_df = df.copy()
        prop = 1-percentile
        weights_seq = new_df.distance[:N]
        if normalize:
            self._weights = (weights_seq - weights_seq.min() + 1e-3)/weights_seq.std()
        else:
            self._weights = weights_seq
        avg = partial(np.average, weights = self._weights)
        if prop == 1:
            result = new_df
        result = new_df.iloc[:N,:].apply(avg, axis = 0).sort_values(ascending = False)
        result.pop('distance')
        ranking = pd.DataFrame(zip(result.index.tolist(), result), columns = ['coefficients', 'proportion'])
        ranking['ranking'] = ranking.proportion.rank(ascending = False, method = 'min')
        return ranking
        
    def set_beta_sum_range(self, low = None, high = None):
        if low:
            assert low > 0
        else:
            low = 1
        if high:
            assert high < 2**self.sample_generator.p - self.sample_generator.p
        else:
            high = 4
        self._beta_sum_range = (low, high)
        return self._beta_sum_range
        
        
    @property
    def beta_sum_range(self):
        if hasattr(self, '_beta_sum_range'):
            pass
        else:
            self._beta_sum_range = self.set_beta_sum_range()
        return self._beta_sum_range

    @property
    def all_available_beta_for_test(self):
        if hasattr(self, '_all_available_beta_for_test'):
            pass
        else:
            non_zero_part = product([0,1], repeat = 2**self.sample_generator.p - self.sample_generator.p - 1)
            zeros = [0] + [0 for _ in range(self.sample_generator.p)]
            if hasattr(self, '_beta_sum_range'):
                beta_range = list(range(self.beta_sum_range[0], self.beta_sum_range[1]+1))
                self._all_available_beta_for_test = np.array([zeros + list(x) for x in non_zero_part if sum(x) in beta_range ]) #if sum(x)
            else:
                self._all_available_beta_for_test = np.array([zeros + list(x) for x in non_zero_part if sum(x)]) #if sum(x)
        return self._all_available_beta_for_test
    
   
    
    @property
    def all_distance(self): # this part needs to be done in multi-processing fashion later...
        if hasattr(self, '_all_distance'):
            pass
        else:
            from scipy.linalg import svd
            all_beta = self.all_available_beta_for_test
            avg_distance = []
            for beta in all_beta:
                dist = []
                k = sum(beta)
                beta = self.gen_beta_contrasts(beta)
                orths = svd(beta.T)[0][:, :k].T
                for c in orths:
                    dist.append(self.get_single_distance(c, scanner.L_inv, scanner.C))
                avg_distance.append(np.mean(dist))
            df_list = []
            for beta, dist in zip(all_beta, avg_distance):
                seq = list(beta) + [dist]
                df_list.append(seq)
            self._all_distance = pd.DataFrame(df_list) # need to add columns
        return self._all_distance
    
    
    
    def gen_beta_contrasts(self, beta):
        if isinstance(beta, list):
            beta = np.array(beta)
        beta = beta.reshape(-1)
        assert beta.shape[0] == 2**self.sample_generator.p
        assert beta.max() == 1
        where_ones= np.where(beta)[0]
        if where_ones.shape[0] == 1:
            return beta.reshape(1, -1)
        else:
            N = 2**self.sample_generator.p
            contrasts = []
            for idx, beta_idx in enumerate(where_ones):
                if idx:
                    c_compare = np.zeros(N, dtype = np.byte).reshape(1, -1); c_compare[0,beta_idx] = 1
                    c = c_base - c_compare
                    contrasts.append(c/norm(c, 2))
                    c_base = c.copy(); c_base[0, c_base_index ] += 1
                else:
                    c_base = np.zeros(N, dtype = np.byte).reshape(1, -1)
                    c_base[0,beta_idx] = 1
                    contrasts.append(c_base)
                    c_base_index = beta_idx
            contrasts = np.concatenate(contrasts, axis = 0)
            return contrasts