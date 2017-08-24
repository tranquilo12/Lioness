# ----------------------------------
# imports

import sys
sys.path.append(r'C:/Users/user/Desktop/Shriram/PythonTools/')

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import feature_selection, externals, ensemble, linear_model
from sklearn import preprocessing, decomposition
from sklearn import pipeline, model_selection

# Matplotlib for just simple plotting
import matplotlib
import matplotlib.pyplot as plt

# Bokeh for great plotting
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from methods import *

# -----------------------------------------


def clean_wrapper(pd=pd):
    # ----------------------------------
    # read files
    test_path = r'C:\\Users\\user\\Desktop\\Shriram\\Data\\13th July\\USDJPY1916H1Test.csv'
    train_path = r'C:\\Users\\user\\Desktop\\Shriram\\Data\\13th July\\USDJPY1916H1Train.csv'
    x_df, y_df, df = clean_data(
        train_path=train_path, RegressionCol='YCloseH10', pd=pd)
    x2_df, y2_df, df2 = clean_data(test_path=test_path, pd=pd)
    return(x_df, y_df, df, x2_df, y2_df, df2)

# ----------------------------------
# execute


def execute(x_df, y_df, df,
            x2_df, df2, save_path,
            pca=False,
            n_pca=None,
            k=False,
            n_klist=None,
            grid=False,
            cv=5,
            preprocessing=preprocessing,
            model_selection=model_selection,
            pipeline=pipeline):

    model_name = 'Basic'
    model_pca = ('pca', decomposition.PCA())
    model_Kfeatures = ('select_best', feature_selection.SelectKBest())
    algo = ('ml', xgb.XGBRegressor())

    process = []
    search_space = {}
    if pca:
        if model_name == 'Basic':
            model_name = 'PCA'
        else:
            model_name = model_name + 'PCA'
        x_scaled = preprocessing.scale(x_df)
        x_df = x_scaled
        x2_df = preprocessing.scale(x2_df)

        if n_pca:
            model_name = model_name + str(n_pca)
            if isinstance(n_pca, list):
                search_space['pca__n_components'] = n_pca
                message = "\nPCA included.. " + str(n_pca) + "\n"
            elif isinstance(n_pca, int):
                search_space['pca__n_components'] = list(n_pca)
                message = "\nPCA included.. " + str(n_pca) + "\n"
        else:
            message = "\nPCA included.. " + "\n"

        model_name = model_name + '_'
        process.append(model_pca)

    if k:
        if model_name == 'Basic':
            model_name = 'K'
        else:
            model_name = model_name + 'K'
        if n_klist:
            if isinstance(n_klist, list):
                model_name = model_name + str(n_klist)
                search_space['select_best__k'] = n_klist
                message = message + "SelectKFeatures included.." + \
                    str(n_klist) + "\n"
            elif isinstance(n_klist, int):
                model_name = model_name + str(n_klist)
                search_space['select_best__k'] = list(n_klist)
                message = message + "SelectKFeatures included.." + \
                    str(n_klist) + "\n"
        else:
            message = message + "SelectKFeatures included.." + "\n"

        model_name = model_name + '_'
        process.append(model_Kfeatures)

    # add algos as well

    process.append(algo)
    model = pipeline.Pipeline(process)

    if grid == True:
        model_name = model_name + 'Grid'
        hypermodel = model_selection.GridSearchCV(
            model, param_grid=search_space, cv=cv, n_jobs=-1)
        hypermodel.fit(x_df, y_df)
        results = model_selection.cross_val_score(
            hypermodel, x_df, y_df, n_jobs=-1)
        print("Prediction score of the model: %.2f%s (%.5f standard deviation) Fitness" % (
            results.mean() * 100, "%", results.std()), "\n\n")
        prediction = hypermodel.predict(x2_df)
    else:
        model.fit(x_df, y_df)
        results = model_selection.cross_val_score(model, x_df, y_df, n_jobs=-1)
        print("Prediction score of the model: %.2f%s (%.5f standard deviation) Fitness" % (
            results.mean() * 100, "%", results.std()))
        prediction = model.predict(x2_df)

    prediction = model.predict(x2_df)
    # save_as(model_name=model_name, prediction=prediction, df2=df2[['DATE','TIME','H1.CLOSE']],  save_path=save_path)
    # print('Saved as %s' % model_name)

    plotter(date=pd.to_datetime(df2['DATE']), prediction=prediction, close=df2[
            'H1.CLOSE'], figure=figure, show=show)


if __name__ == '__main__':

    x_df, y_df, df, x2_df, y2_df, df2 = clean_wrapper(pd=pd)
    save_path = r'C:\\Users\\user\\Desktop\\Shriram\\Data\\xg\\'

    print('---')
    print('Executing Pipeline ...')

    execute(x_df, y_df, df,
            x2_df, df2, save_path,
            pca=False,
            n_pca=None,
            k=False,
            n_klist=None,
            grid=False,
            cv=1,
            # pd=pd,
            preprocessing=preprocessing,
            model_selection=model_selection,
            pipeline=pipeline)
