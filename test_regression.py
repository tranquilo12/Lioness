# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:26:59 2017

@author: Greg Sigalov <arrowstem@gmail.com>
Arrow Science & Technology

"""

from robustregression import RebustLinearRegression
import csv
import time
from timetools import NiceTimeString
import numpy as np

# Script to test Robust Linear Regression

t1 = time.time()

training_file = 'RegressionTrain.csv'
test_file = 'RegressionTest.csv'

output_param = 'RegressionParameters.csv'
output_predict = 'RegressionPrediction.csv'

K = 10 # K-fold cross-validation

model = RebustLinearRegression(training_file) # creating the regression model
model.read_test_data(test_file)

""" The robust criterion function for downweighting outliers. Acceptable options are: 
        HuberT (default, can be dropped)
        LeastSquares / LS / 1
        RamsayE / RE / 2
        AndrewWave / AW / 3
        TrimmedMean / TM / 4
        Hampel / H / 5
        TukeyBiweight / TB / 6 (default in Matlab)
"""
use_norm = 'TukeyBiweight'

predictors = model.predictors
predictors = ['Intercept'] + predictors

params_all = []
predict_all = []
header_param = ['Variable']
header_predict = ['Observation']

header_param.append('Coefficient')
header_predict.append('Prediction')

params = model.train_model_Kfold(K, use_norm)
params_all.append(params)

with open(output_param, 'wb') as csvfile:
    wrt = csv.writer(csvfile, delimiter=',')
    wrt.writerow(header_param)
    n_norm = len(params_all)
    for i in range(len(params)):
        row = [predictors[i]]
        for j in range(n_norm):
            row.append(params_all[j][i])
        wrt.writerow(row)
        
# Predict the response
ypred = model.predict_response(params)
predict_all.append(ypred)

with open(output_predict, 'wb') as csvfile:
    wrt = csv.writer(csvfile, delimiter=',')
    wrt.writerow(header_predict)
    for i in range(len(ypred)):
        row = [i+1]
        for j in range(n_norm):
            row.append(predict_all[j][i])
        wrt.writerow(row)

t2 = time.time()
print '\ntime elapsed:', NiceTimeString(t2-t1), 'from the start\n'