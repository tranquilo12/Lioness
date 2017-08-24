import sys
sys.path.append(r'C:/Users/user/Desktop/Shriram/PythonTools/')

from methods import *
import numpy as np 
import pandas as pd
import xgboost as xgb
import random

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# warnings.filterwarnings("ignore")
# import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import model_selection, preprocessing 
from sklearn import decomposition, pipeline, tree 
from sklearn import feature_selection, externals, ensemble, linear_model

## Matplotlib for just simple plotting
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
## Bokeh for great plotting
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import gridplot, row
from bokeh.plotting import figure, show, output_file
output_notebook()

save_path = r'C:/Users/user/Desktop/Shriram/Data/xg/'
train_path = r'C:/Users/user/Desktop/Shriram/Data/19.07.17/EURUSDODM4H4NormalisedTrain.csv'
test_path =  r'C:/Users/user/Desktop/Shriram/Data/19.07.17/EURUSDODM4H4NormalisedTest.csv'

# # ----------------------------------
# # read files
# x_df, y_df, df = clean(path=train_path, ClassificationCol='YClassH28', pd=pd)
# x2_df, y2_df, df2 = clean(path=test_path, pd=pd)
# # ----------------------------------