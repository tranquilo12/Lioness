# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:17:22 2017

@author: Greg Sigalov <arrowstem@gmail.com>
Arrow Science & Technology

"""

import csv
import statsmodels.api as sm
import numpy as np
from sklearn import linear_model, datasets
import time
from timetools import NiceTimeString

def is_number(s):
    if isinstance(s, str): 
        if len(s) == 0:
            return(False)
        elif s.lower() == 'inf' or s.lower() == 'nan':
            return(False)
    elif s == float('nan') or s == float('inf'):
        return(False)
    try:
        float(s)
        return(True)
    except ValueError:
        return(False)


class RobustLinearRegression:

    # training_file - CSV file name (with path) with training data
    
    def __init__(self, training_file):
        self.norms = ['HuberT', 'LeastSquares', 'RamsayE', 'AndrewWave', 'TrimmedMean', 'Hampel', 'TukeyBiweight']
        self.norms_short = ['HT', 'LS', 'RamsayE', 'AW', 'TM', 'H', 'TB']
        self._Kfold = 1 # no cross-validation by default
        self.__training_file = training_file
        self._read_training_data() # read data
        self._count_numeric_rows() # see which columns can be used for regression
        self.predictors = [] # captions of the columns to be used as predictors
        self.X = [] # selected training data (predictors)
        self.y = [] # dependent variable (target)
        self._clean_training_data()  
        
        
    """ This function reads the training data from a CSV file
    """
    def _read_training_data(self):
        self.__header = []
        self.__data_train = []
        with open(self.__training_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            i = 0 # row
            for row in csvreader:
                if i == 0:
                    self.__header = row
                else:
                    self.__data_train.append(row)
                i = i + 1
        

    """ This function reads the test data from a CSV file
    """
    def read_test_data(self, test_file):
        self.X_test = []
        with open(test_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            i = 0 # row
            for row in csvreader:
                if i == 0:
                    header_test = row
                    var_index = []
                    # Find the correspondence between the captions of the training
                    # and test sets (they are not necessarily identical)
                    n = len(header_test)
                    for s in self.predictors:
                        k = -1
                        for j in range(n):
                            if header_test[j] == s:
                                k = j
                                break
                        if k >= 0:
                            var_index.append(k) # this predictor is located at the kth place in row
                        
                    m = len(self.predictors)
                    if len(var_index) < m:
                        print('The test set doesn\'t have all predictors of the training set!')
                        print('predictors:', self.predictors)
                        print('test set header:', header_test)
                        return
                    
                else:
                    x_row = []
                    for j in range(m):
                        k = var_index[j]
                        z = row[k]
                        if is_number(z):
                            x_row.append(float(z))
                        else:
                            x_row.append(float('nan'))
                    self.X_test.append(x_row)
                
                i = i + 1


    """ This function counts cells with numeric data in each column. 
        Columns with zero numeric cells will be ignored. 
        While counting, non-numeric elements will be replaced by NAN
    """
    def _count_numeric_rows(self):
        n = len(self.__header) # number of columns
        self.__numeric_row_count = [0 for i in range(n)]
        data_train_new = [] # updated array with non-numbers replaced by NAN
        for row in self.__data_train:
            j = 0 # column
            row_with_nan = []
            for s in row:
                if is_number(s) and float(s) != float('nan') and float(s) != float('inf'):
                    self.__numeric_row_count[j] = self.__numeric_row_count[j] + 1
                    row_with_nan.append(float(s))
                else:
                    row_with_nan.append(float('nan'))
                j = j + 1
            data_train_new.append(row_with_nan)
        self.__data_train = data_train_new # overwrite with numeric only data
            

    """ This function creates the list of predictors, the data array that includes
        only the predictors, and the target data vector
    """
    def _clean_training_data(self):
        n = len(self.__header) # number of columns
        target_col = n-1 # last column by default (to override, the caption must be TARGET (any case))
        
        # select the columns to use as the predictors and the target
        for j in range(n): # consider each column
            if self.__numeric_row_count[j]: # this column contains numeric data
                if self.__header[j].lower() == 'target':
                    target_col = j
                else:
                    self.predictors.append(self.__header[j])
                    
        # Let's make sure than all predictors are unique
        u = set(self.predictors)
        if len(u) < len(self.predictors):
            print('Some predictors are not unique!')
            return
        
        # select data for the predictors and the target
        k = len(self.__data_train)
        i = 0
        for row in self.__data_train: # consider each row
            m = len(row)
            if m != n:
                print('n =', n, '   m =', m, '   i =', i)
            i = i + 1
            if m != n:
                continue
            row_cleaned = []
            yy = float('nan')
            for j in range(n): # consider each column
                if j == target_col:
                    yy = row[j]
                elif self.__numeric_row_count[j]: # this column contains numeric data
                    row_cleaned.append(row[j])
                    
            self.X.append(row_cleaned)
            self.y.append(yy)


    def _set_cross_validation_number(self, K):
        if is_number(K) and K == int(K) and K > 0:
            self._Kfold = K
            return 1 # OK
        else:
            print('wrong cross-validation number (Kfold)', K)
            return(0) # error


    """ This function fits (trains) the model on the ENTIRE test set, see:
        http://www.statsmodels.org/dev/generated/statsmodels.robust.robust_linear_model.RLM.html
        
        M : statsmodels.robust.norms.RobustNorm, optional
        The robust criterion function for downweighting outliers. The current options are: 
            LeastSquares
            HuberT 
            RamsayE 
            AndrewWave 
            TrimmedMean 
            Hampel 
            TukeyBiweight 
        The default is HuberT(). 
    """
    def train_model(self, use_norm = 'HuberT', max_num_var = 0):
        for i in range(len(self.norms)):
            if use_norm == self.norms[i] or use_norm == self.norms_short[i] or use_norm == i:
                n = i
                break
        print('norm used:', self.norms[n])
        print('Number of observations:', len(self.X))

        # Option to limit the number of variables
        row = self.X[0]
        num_var = len(row)
        if max_num_var > 0 and max_num_var < num_var: # ignore some variables
            print('using', max_num_var, 'variables out of', num_var)
            x = []
            for row in self.X:
                x.append(row[:max_num_var])
        else:
            x = self.X

        params = self._train_model(n, x, self.y)
        return params      
        

    def _train_model(self, use_norm, X_row, y_row): # use_norm, X, y must be explicitely specified
        X, y = self._select_numeric_rows(X_row, y_row) # Keep only rows in which all variables are numeric
        x = sm.add_constant(X)
        n = use_norm
        
        if n == 1:
            rlm_model = sm.RLM(y, x, M=sm.robust.norms.LeastSquares())
        elif n == 2:
            rlm_model = sm.RLM(y, x, M=sm.robust.norms.RamsayE())
        elif n == 3:
            rlm_model = sm.RLM(y, x, M=sm.robust.norms.AndrewWave())
        elif n == 4:
            rlm_model = sm.RLM(y, x, M=sm.robust.norms.TrimmedMean())
        elif n == 5:
            rlm_model = sm.RLM(y, x, M=sm.robust.norms.Hampel())
        elif n == 6:
            rlm_model = sm.RLM(y, x, M=sm.robust.norms.TukeyBiweight())
        else:        
            rlm_model = sm.RLM(y, x, M=sm.robust.norms.HuberT())
            
        rlm_results = rlm_model.fit()
#        print rlm_results.params 
        return rlm_results.params       

        
    # Split the training data into K folds of ROUGHLY the same size
    def _create_random_folds(self):
        n = len(self.X) # number of observations
        K = self._Kfold
        print('number of observations =', n)
        if K > 1:
            print('K-fold cross-validation, K =', K)
        else:
            print('no K-fold cross-validation')
        fold_index = np.random.randint(K, size=n)
        fold_index_count = [0 for i in range(K)]
        for k in fold_index:
            fold_index_count[k] = fold_index_count[k] + 1
        for k in range(K):
            print('k =', k, '   count =', fold_index_count[k])
        return(fold_index)
    
        
    # Split the training data into K folds of EXACTLY the same size
    def _create_random_folds_equal_sized(self):
        n = len(self.X) # number of observations
        K = self._Kfold
#        print 'number of observations =', n
#        if K > 1:
#            print 'K-fold cross-validation, K =', K
#        else:
#            print 'no K-fold cross-validation'
            
        # index of rows that haven't been assigned to a fold yet
        index = [i for i in range(n)] 
        fold_index = [-1 for i in range(n)]  # to be filled out
        k = 0
        while len(index):
            i = np.random.randint(len(index)) # pick ONE random number
            j = index[i]
            if j >= len(fold_index):
                print('j =', j, '  len(fold_index) =', len(fold_index), '  len(index) =', len(index))
            fold_index[j] = k
            index.pop(i)
            k = k + 1
            if k == K:
                k = 0
        
        fold_index_count = [0 for j in range(K)]
        for k in fold_index:
            fold_index_count[k] = fold_index_count[k] + 1
#        for k in range(K):
#            print 'k =', k, '   count =', fold_index_count[k]
        return(fold_index)
    
        
    """ Creates subset of arrays X and y that leaves out the kth fold """
    def _create_subset_Kfold(self, fold_index, k):
        sub_X = []
        sub_y = []
        n = len(fold_index) # number of observations
        for i in range(n):
            if fold_index[i] == k:
                continue # omit
            sub_X.append(self.X[i])
            sub_y.append(self.y[i])
        return sub_X, sub_y


    def train_model_Kfold(self, K, use_norm = 'HuberT'):
        if not self._set_cross_validation_number(K): # wrong K
            return []
        
        for i in range(len(self.norms)):
            if (use_norm == self.norms[i]) or (use_norm == self.norms_short[i]) or (use_norm == i):
                n = i
                break
        print('norm used:', self.norms[n])

        fold_index = self._create_random_folds_equal_sized()
        print('training the K-fold model, K =', K)
        for k in range(K): 
            t1 = time.time()
            sub_X, sub_y = self._create_subset_Kfold(fold_index, k)
            params = self._train_model(use_norm, sub_X, sub_y)
            if k == 0:
                params_mean = params
            else:
                for i in range(len(params)):
                    params_mean[i] = params_mean[i] + params[i] # cumulative
            t2 = time.time()
            print('fold', k+1, 'out of', K, 'took', NiceTimeString(t2-t1))

        # now calculate the mean parameters
        for i in range(len(params)):
            params_mean[i] = params_mean[i] / K
        return params_mean      


    def predict_response(self, params):
        x = sm.add_constant(self.X_test)
        y = np.dot(x,params)
        return y    

    def _save_temp_file(self, x, y):
        temp_file = 'temp_file_xy.csv'
        with open(temp_file, 'wb') as csvfile:
            wrt = csv.writer(csvfile, delimiter=',')
            n = len(x[0])
            p = self.predictors[:n]
            p.append('TARGET')
            wrt.writerow(p)
            for i in range(len(x)):
                z = x[i]
                z.append(y[i])
                wrt.writerow(z)


    """ Keep only rows in which all variables are numeric """
    def _select_numeric_rows(self, x, y):
        x_new = []
        y_new = []
        n = len(y)
        for i in range(n):
            if y[i] != float(y[i]):
                continue
            good_row = 1
            for z in x[i]:
                if z != float(z):
                    good_row = 0
                    break
            if good_row == 0:
                continue
            x_new.append(x[i])
            y_new.append(y[i])
        return x_new, y_new
