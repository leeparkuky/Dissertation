import numpy as np
import pandas as pd
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
# custom packages
from Dissertation import RandomGenerator, convert_to_int, get_order_index
from utils import *
from estimators import ClusteredSegmentation
# scipy
import scipy
from scipy.sparse import vstack, identity, csr_array, vstack, csc_matrix
from scipy.sparse.linalg import svds, inv
from scipy.stats import f, ncf
from scipy.spatial.distance import mahalanobis, euclidean, jensenshannon
# sklearn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.covariance import EmpiricalCovariance, MinCovDet
# others
from tqdm.notebook import trange
from itertools import combinations, starmap
from functools import partial
from typing import List
from joblib import Parallel, delayed




def check_ones(seq_a, seq_b):
    return np.all(np.isin(np.where(seq_a ==1)[0], np.where(seq_b ==1)[0]))

def check_zeros(seq_a, seq_b):
    return np.all(np.isin(np.where(seq_a ==0)[0], np.where(seq_b ==0)[0]))

def check_both(seq_a, seq_b):
    return check_ones(seq_a, seq_b)*check_zeros(seq_a, seq_b)

def find_prob(missing_array, weighted_mean):
    return np.prod([1-w if e == 0 else w  for e, w in zip(missing_array, weighted_mean)])








class barcodeScanner:
    def __init__(self, estimator, config):
        self.config = config
        self.barcode_length = config.p
        if hasattr(estimator, "full_to_reduced"):
            self.segmentation_table = estimator.full_to_reduced_with_counts.copy()
            self.num_clusters = estimator.n_clusters
            self.num_parameters_full = self.segmentation_table.shape[0]
            self.num_missing_pairs = 2**self.barcode_length - self.segmentation_table.shape[0]
            self.missing_pairs_decimal_repr = list(filter(lambda x: x not in self.segmentation_table.full.values, range(2**self.barcode_length)))
        else:
            raise AttributeError("'estimator' must have been fit and have 'full_to_reduced' attribute")
        self.var_names = [f"X_{i}" for i in range(self.barcode_length)]
        self.var_names_all = expand_var_names(self.var_names)
        
    def MLEscan(self): # Needs to be edited (After fully investigating the chapter 5)
        B = vstack([self.raw_contrast, scanner.missingPairsMLE()])
        proj = B.T @ inv(B @ B.T) @ B
    
    

    def find_reduced_jensenshannon(self, missing, all_arrays = None):
        distance = []
        if all_arrays:
            pass
        else:
            all_arrays = [self.num_to_barcode(x, scanner.barcode_length).toarray().reshape(-1) for x in range(2**self.barcode_length)]
        for weighted_means in self.weighted_segment_means:
            cond = partial(check_both, weighted_means)
            prob = partial(find_prob, weighted_mean = weighted_means)
            prob_seq = [prob(x) for x in all_arrays if cond(x)]
            prob_domain = [x.tolist() for x in all_arrays if cond(x)] if missing.tolist() in [x.tolist() for x in all_arrays if cond(x)] else  [x.tolist() for x in all_arrays if cond(x)] + [missing.tolist()]
            if len(prob_seq) < len(prob_domain):
                prob_seq.append(0)
            missing_prob_seq = [0 for _ in range(len(prob_seq)-1)] + [1]
            d = jensenshannon(prob_seq, missing_prob_seq)
            distance.append(d)
        return distance.index(min(distance))

    def assign_missing_jensenshannon(self, independece = True):   # Need to add lines for when independence = False
        if hasattr(self, '_segmentation_table_jensenshannon'):
            pass
        else:
            table = self.segmentation_table.copy().drop('counts',axis = 1)
            all_arrays = [self.num_to_barcode(x, self.barcode_length).toarray().reshape(
                -1) for x in range(2**self.barcode_length)]
            reduced = Parallel(n_jobs = -1, verbose = 0)(
                delayed(self.find_reduced_jensenshannon)(missing, all_arrays) for missing in self.missing_arrays)
            self._segmentation_table_jensenshannon =  pd.concat([table, pd.DataFrame(zip(self.missing_pairs_decimal_repr, reduced),
                                                                                     columns = ['full','reduced'])],
                                                             ignore_index = True).sort_values('full').reset_index(drop = True)
        return self._segmentation_table_jensenshannon
        
    
    def assign_missing_mahalanobis(self, pooled= False, robust = False):
        if pooled:
            if hasattr(self, '_segmentation_table_mahalanobis_pooled'):
                pass
            else:
                from numpy.linalg import inv
                table = self.segmentation_table.copy() # segmentation_table
                table['barcode'] = table.apply(lambda x: self.num_to_barcode(
                    x['full'], self.barcode_length).toarray()[0], axis = 1)
                X_list = []
                for barcode, count in zip(table.barcode, table.counts):
                    X_list += [barcode.tolist() for _ in range(count)]  
                if robust:
                    cov = MinCovDet()
                else:
                    cov = EmpiricalCovariance()
                cov.fit(X_list)
                VI = inv(cov.covariance_)
                reduced = []
                for missing in self.missing_arrays:
                    pooled_dist = [mahalanobis(w, missing.reshape(-1), VI) for w in self.weighted_segment_means]
                    reduced.append(pooled_dist.index(min(pooled_dist)))
                self._segmentation_table_mahalanobis_pooled = pd.concat(
                            [table[['full','reduced']], pd.DataFrame(zip(
                                self.missing_pairs_decimal_repr, reduced), columns = ['full','reduced'])],
                                    ignore_index=True).sort_values('full').reset_index(drop = True)  
            return self._segmentation_table_mahalanobis_pooled
        else:
            if hasattr(self, '_segmentation_table_mahalanobis'):
                pass
            else:
                table = self.segmentation_table.copy() # segmentation_table
                table['barcode'] = table.apply(lambda x: self.num_to_barcode(
                    x['full'], self.barcode_length).toarray()[0], axis = 1)
                reduced = np.sort(table.reduced.unique())
                covs = []
                for r in reduced:
                    barcodes = []
                    r_table = table.loc[table.reduced.eq(r), ['barcode','counts']]
                    for b, c in zip(r_table.barcode, r_table['counts']):
                        barcodes += [b for _ in range(c)]
                    barcodes = np.concatenate(barcodes).reshape(-1, 10)
                    if robust:
                        cov = MinCovDet()
                    else:
                        cov = EmpiricalCovariance()
                    cov.fit(barcodes)
                    covs.append(cov)
                    del barcodes


                full = []
                full_to_reduced = []
                for missing, missing_dec in zip(self.missing_arrays, self.missing_pairs_decimal_repr):
                    m_dist = [
                                (r, cov.mahalanobis(missing.reshape(1,-1))[0]) for cov, r in zip(covs, reduced) if (
                                    cov.covariance_ == 0).mean() < 1]
                    non_zero_min = min([x  for y, x in m_dist if x>0])
                    seg_index = [y for x, y in m_dist].index(non_zero_min)
                    full_to_reduced.append(m_dist[seg_index][0]), full.append(missing_dec)
                self._segmentation_table_mahalanobis = pd.concat(
                            [table[['full','reduced']], pd.DataFrame(zip(
                                full, full_to_reduced), columns = ['full','reduced'])],ignore_index=True).sort_values(
                            'full').reset_index(drop = True)  
                del table, covs, cov, full, full_to_reduced, m_dist
            return self._segmentation_table_mahalanobis
            
    
    
    
    
    
    @property
    def assign_missing_euclidean(self):
        if hasattr(self, '_segmentation_table_euclidean'):
            pass
        else:
            cluster_segments = self.segmentation_table.copy().drop('counts', axis = 1) # segmentation_table
            reduced = []
            full = []
            for missing, missing_dec in zip(self.missing_arrays, self.missing_pairs_decimal_repr):
                euclidean_distance = [euclidean(x, missing) for x in self.weighted_segment_means ]
                segment_to_assign = euclidean_distance.index(min(euclidean_distance))
                del euclidean_distance
                full.append(missing_dec); reduced.append(segment_to_assign)
                del segment_to_assign
            self._segmentation_table_euclidean = pd.concat(
                [cluster_segments, pd.DataFrame(zip(full, reduced), columns = ['full','reduced'])], 
                ignore_index = True).sort_values('full').reset_index(drop = True)
            del cluster_segments, full, reduced
        return self._segmentation_table_euclidean

    
    
    @property
    def weighted_segment_means(self):
        if hasattr(self, '_weighted_means'):
            pass
        else:
            table = self.segmentation_table.copy()
            table['barcode'] = table.apply(lambda x: self.num_to_barcode(x['full'], 10).toarray()[0], axis = 1)
            self._weighted_means = table[['barcode','counts','reduced']].groupby('reduced', as_index= False).apply(
                lambda x: np.average(np.concatenate(list(x['barcode'])).reshape(-1, 10), axis = 0, weights = x['counts']))
            del table
        return self._weighted_means
        
    
    @property
    def raw_contrast(self):
        if hasattr(self, "raw_contrast_"):
            pass
        else:
            result = []
            for key in self.groupby_expanded_barcode.keys():
                result.append(self.gen_contrast(self.groupby_expanded_barcode[key]))
            self.raw_contrast_ = vstack(result)
            del result
        return self.raw_contrast_

    @staticmethod
    def gen_contrast(csc_matrix: scipy.sparse._csc.csc_matrix)-> scipy.sparse._csc.csc_matrix:
        fixed_array = csc_matrix.getrow(0)
        result = []
        for i in range(1, csc_matrix.shape[0]):
            result.append(fixed_array - csc_matrix.getrow(i))
        if len(result) > 0:
            return vstack(result)
        else:
            return None
    

    @property
    def groupby_expanded_barcode(self):
        if hasattr(self, "groupby_expanded_barcode_"):
            pass
        else:
            self.groupby_expanded_barcode_ = self.segmentation_table.groupby('reduced')['full'].apply(self.num_to_expanded_barcode_batch).to_dict()
        return self.groupby_expanded_barcode_
    
    def num_to_expanded_barcode_batch(self, seq):
        func = partial(self.num_to_barcode, length = self.barcode_length) # using self.num_to_barcode with the barcode_length
        result = []
        for num in seq:
            result.append(func(num))
        barcode_csc = vstack(result)
        del result
        result = []
        for barcode_array in barcode_csc:
            result.append(self.expand_barcode(barcode_array))
        return vstack(result)

    @staticmethod
    def num_to_barcode(i:int, length:int):
        binary = bin(i)[2:]
        while len(binary) < length:
            binary = '0' + binary
        return csc_matrix([int(x) for x in list(binary)], dtype = np.byte)
    
    @property
    def missing_arrays(self):
        if hasattr(self, '_missing_arrays'):
            pass
        else:
            missing_pairs_dec = self.missing_pairs_decimal_repr
            missing_arrays = [self.num_to_barcode(x, self.barcode_length).toarray().reshape(-1) for x in missing_pairs_dec]
            self._missing_arrays = missing_arrays
        return self._missing_arrays

    
    
    
    def expand_barcode(self, barcode):
        ones_index_main = [f'X_{i}' for i in barcode.nonzero()[1]]
        ones_index = ones_index_main.copy()
        n = len(ones_index_main)+1
        for i in range(2, n):
            m = starmap(sum_string, combinations(ones_index_main, i))
            ones_index = ones_index + list(m)
        col = np.array([self.var_names_all.index(index)+1 for index in ones_index])
        col = np.insert(col, 0, 0)
        row = np.zeros((len(col),), dtype = np.byte)
        data = np.ones((len(col),), dtype = np.byte)
        csr_result = csr_array((data, (row, col)), shape=(1, 2**self.barcode_length))
        return csc_matrix(csr_result, dtype = np.byte)
    
    def missingPairsMLE(self):
        if self.num_missing_pairs > 0:  
            self.null_space_size = 2**self.barcode_length - self.num_clusters
            missing_pairs_expanded = self.num_to_expanded_barcode_batch(self.missing_pairs_decimal_repr)
            result = []
            for array in missing_pairs_expanded:
                max_index = array.nonzero()[1].max()
                col = np.array([max_index])
                row = np.zeros((1,), dtype = np.byte)
                data = np.ones((1,), dtype = np.byte)
                result.append(csc_matrix((data, (row, col)), shape=(1, 2**self.barcode_length)))
            return vstack(result)
        else:
            return None