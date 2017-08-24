import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
#get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (30,12)

from sklearn import model_selection, preprocessing, decomposition
from sklearn import pipeline, feature_selection, linear_model, externals

# train_path = r"C:\Users\user\Desktop\Shriram\Data\13th July\USDJPY1916H1Train.csv"
# test_path = r"C:\Users\user\Desktop\Shriram\Data\13th July\USDJPY1916H1Test.csv"

def clean_data(test_path=False, train_path=False):
    if test_path!=False:
        # open file as df
        df = pd.read_csv(test_path, header=0)
    if train_path!=False:
        # open file as df
        df = pd.read_csv(train_path, header=0)
    # check if there is a column called TARGETClassification
    # if yes, then it is a training file
    RegressionCol = 'YCloseH10'
    if RegressionCol in df.columns:
        # it is a taining file
        print('Training file found')
        #drop unwanted columns for df's
        dropcols = ['DATE', 'TIME'] + [RegressionCol]
        x_df = df.drop(dropcols, axis=1)
        x_df = x_df.dropna(axis=1)
        print(x_df.shape)
        y_df = df[RegressionCol]
        # validate df shapes
        if x_df.shape[0]!=y_df.shape[0]:
            print('Shapes dont match, x_df= ', x_df.shape, 
              'y_df=', y_df.shape)
            exit(0)
        #remove object types from the data
        objcols = list(x_df.select_dtypes(include=['object']).columns)
        x_df[objcols] = x_df[objcols].apply(pd.to_numeric, errors='coerce')
        # verify that there are no more objects left
        verify = list(x_df.select_dtypes(include=['object']).columns)
        if len(verify)==0:
            pass;
        else:
            print('Objects still exist, please have a look at the data again')
            exit(0)
        # fill all na's with the mean of columsn
        x_df[objcols] = x_df[objcols].fillna(x_df[objcols].mean())
        # scale data 
        # scaling reduces mean to 0 and 1st deviation
        x_scaled = preprocessing.scale(x_df)
        return(x_scaled, x_df, y_df, df)
    else:
        #it's a test file
        print('Test file found')
        #drop unwanted columns for df's
        dropcols = ['DATE', 'TIME']
        x_df = df.drop(dropcols, axis=1)
        x_df = x_df.dropna()
        objcols = list(x_df.select_dtypes(include=['object']).columns)
        x_df[objcols] = x_df[objcols].apply(pd.to_numeric, errors='coerce')
        # verify that there are no more objects left
        verify = list(x_df.select_dtypes(include=['object']).columns)
        if len(verify)==0:
            pass;
        else:
            print('Objects still exist, please have a look at the data again')
            exit(0)
        # fill all na's with the mean of columsn
        x_df[objcols] = x_df[objcols].fillna(x_df[objcols].mean())
        # scale data: reduces mean to 0 and 1st deviation
        x_scaled = preprocessing.scale(x_df)
        y_df = []
        return(x_scaled, x_df, y_df, df)

# dimentional reduction should be handled by the pipeline, not in a seperate function
# That is until further experiments
def dimentional_reduction(x_data_scaled, n, n_list=None, plot=False):
    model_pca = decomposition.PCA(n_components=n)
    model_pca = model_pca.fit(x_data_scaled)
    variance = model_pca.explained_variance_ratio_
    cum_variance = np.cumsum(np.round(variance, decimals=4)*100)
    if plot==True:
        plt.plot(cum_variance)
    else:
        pass;
    return(model_pca, variance, cum_variance)


def setup_gridsearch_parameters(valid=False):
    # turned off, so nothing is tuned as default
    if valid==True:
        # setup pca variables
        pca_min = 5
        pca_max = 10
        pca_interval = 5
        n_pca = list(range(pca_min, pca_max, pca_interval))

        # set up feature(k) tuning
        bestk_min = 1
        bestk_max = 5
        bestk_interval = 3
        n_select_best = list(range(bestk_min, bestk_max, bestk_interval))
    
    if valid==False:
        n_pca = []
        n_select_best = []

    return(n_pca, n_select_best)

def execute_pipeline(x_data_scaled, y_data, n_pca, n_select_best):
    # make sure the model knows 
    # 1. Method of dimentional reduction, with nothing reduced
    # if n-components is not set, 
    # all components are used (http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    model_pca = decomposition.PCA()

    # 2. Method of feture selection
    # the all option bypasses selection, 
    # for use in a parameter search (http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)
    model_variable_best = feature_selection.SelectKBest(k=all)

    # 3. Machine Learning algo to use
    machine_learner = linear_model.RANSACRegressor()

    # Package the process
    process = [
        ('pca', model_pca), 
        ('select_best', model_variable_best), 
        ('ml', machine_learner)
    ]

    # line it up to exexute 
    pipeline_process = pipeline.Pipeline(process)

    # search space, where all the process keys can be modified
    search_space = dict(
        pca__n_components=n_pca, 
        select_best__k=n_select_best
        )

    # number of crossfolding validations to be done
    n_cv = 10

    # tell the model, that there will be some crossfolding to be done, and pass it the pipleine for machine learning 
    model = model_selection.GridSearchCV(pipeline_process, param_grid=search_space, cv=n_cv, n_jobs=-1)

    # run the model
    model.fit(x_data_scaled, y_data)

    # ask for results
    results = model_selection.cross_val_score(model, x_data_scaled, y_data, n_jobs=-1)
    print("Prediction score of the model: %.2f%s (%.5f standard deviation) Fitness" % (results.mean()*100, "%",results.std()))

def simple_pipeline(x_scaled, x_df, y_df):
    machine_learner = linear_model.LinearRegression()
    process = [('ml', machine_learner)]
    pipeline_process = pipeline.Pipeline(process)
    search_space = {}
    model = model_selection.GridSearchCV(pipeline_process, param_grid=search_space, cv=10, n_jobs=-1)
    model.fit(x_df, y_df)
    results=model_selection.cross_val_score(model, x_df, y_df, n_jobs=-1)
    print("Prediction score of the model: %.2f%s (%.5f standard deviation) Fitness" % (results.mean()*100, "%",results.std()))
    return(model)

def plotter1(ydf,df):
    f, (ax1, ax2) = plt.subplots(2)
    ax1.plot(df)
    ax2.plot(ydf)
    f.subplots_adjust(hspace=0)
    plt.show()

def main():
    x_scaled, x_df, y_df, df = clean_data(train_path=train_path)
    model = simple_pipeline(x_scaled, x_df, y_df)
    
    x2_scaled, x2_df, y2_df, df2 = clean_data(test_path=test_path)    
    df2['TARGETRegression'] = model.predict(x2_df)
    
    plotter(df2['TARGETRegression'], y_df)
    final_df = df2[['DATE', 'TIME', 'H1.CLOSE','TARGETRegression']]
    final_df['DATE']= pd.to_datetime(final_df['DATE'])

    # final_df.to_csv('LinerRegression_results.csv')

    plotter(final_df['DATE'], final_df['TARGETRegression'])

    fig, ax = plt.subplots()
    ax.plot_date(final_df['DATE'], final_df['TARGETRegression'])
    ax.plot_date(final_df['DATE'], final_df['H1.CLOSE'])
    plt.show()
