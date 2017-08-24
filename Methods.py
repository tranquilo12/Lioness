# -----------------------------------------
# A list of methods used for cleaning data, and pushing them through a pipeline
# -----------------------------------------

# A few things to consider:
# -- All required imports HAVE to be passed into these
# methods below.
# -- The above is to avoid multiple imports.

# -----------------------------------------


def clean(path=None,
          RegressionCol=False,
          ClassificationCol=False,
          pd=None):

    if path == None:
        print('No path provided for data... exiting...')
        return 0

    df = pd.read_csv(path, header=0)
    df['DATETIME'] = df.DATE.map(str) + df.TIME.map(str)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.index = df.DATETIME

    old_shape = df.shape
    # find from path if its a train or test
    # ^ todo
    print('\n--- Clean ---')
    #
    # drop unwanted columns for df's
    # and get y_df ready for specifc instances
    if ClassificationCol is not False:
        dropcols = ['DATE', 'TIME', 'DATETIME'] + [ClassificationCol]
        y_df = pd.DataFrame(df[ClassificationCol])
    if RegressionCol is not False:
        dropcols = ['DATE', 'TIME', 'DATETIME'] + [RegressionCol]
        y_df = pd.DataFrame(df[RegressionCol])
    if (RegressionCol is False) and (ClassificationCol is False):
        dropcols = ['DATE', 'TIME', 'DATETIME']
        y_df = pd.DataFrame({'A': []})
    #
    # drops
    #
    x_df = df.drop(dropcols, axis=1)
    print('Dropped Cols: ', dropcols)

    len_before = len(x_df)
    x_df = x_df.dropna(axis=1)
    len_after = len(x_df)
    if len_before > len_after:
        print('Dropped %d na columns' % len_before - len_after)

    if 'USD/JPY.H1.CORREL' in df.columns:
        x_df = x_df.drop(labels='USD/JPY.H1.CORREL', axis=1)
        print('Dropped USD/JPY.H1.CORREL')
    #
    # validate df shapes
    #
    if (x_df.shape[0] != y_df.shape[0]) or (x_df.shape[1] != y_df.shape[1]):
        print('!!! ALERT !!! Shapes dont match, x_df=', x_df.shape,
              'y_df=', y_df.shape)
    #
    # remove object types from the data
    #
    objcols = list(x_df.select_dtypes(include=['object']).columns)
    verify_old = list(x_df.select_dtypes(include=['object']).columns)
    print("""!!! Number of object columns remaining before conversion: %d !!!""" % len(
        verify_old))
    #
    # verify that there are no more objects left
    #
    if len(verify_old) == 0:
        print('!!! No object columns exist')
    else:
        x_df[objcols] = x_df[objcols].apply(pd.to_numeric, errors='coerce')
        verify_new = list(x_df.select_dtypes(include=['object']).columns)
        print('!!! Number of object columns remaining: %d !!!' %
              len(verify_new))
        print('!!! Objects can still exist, please have a look at the data again !!!')
    #
    # fill all na's with the mean of columns
    #
    x_df[objcols] = x_df[objcols].fillna(x_df[objcols].mean())

    print("--- old shape: " + str(old_shape),
          ", new shape: " + str(x_df.shape) + " ---")

    # Replace all column labels which have a [] with a double (( ))
    x_df.columns = pd.Series(x_df.columns).str.replace('[', '((')
    x_df.columns = pd.Series(x_df.columns).str.replace(']', '))')

    return(x_df, y_df, df)

# -----------------------------------------
# plotter


def plotter(date,
            prediction=None,
            close=None,
            title=None,
            regressor=False,
            figure=None,
            show=None):

    # basic plotting feature
    p1 = figure(x_axis_type="datetime", title=title, plot_width=1600,
                plot_height=400, background_fill_color="#EFE8E2")
    if regressor is True:
        if prediction is not None:
            p1.line(date, prediction, color="#E08E79", legend='prediction')
        if close is not None:
            p1.line(date, close, color="#3B8686", legend='close')
    else:
        if prediction is not None:
            p1.scatter(date, prediction, color="#E08E79", legend='prediction')
        if close is not None:
            p1.scatter(date, close, color="#3B8686", legend='close')

    # aesthetic mapping
    p1.grid.grid_line_alpha = 0.1
    p1.xaxis.axis_label = "Date"
    p1.yaxis.axis_label = "Price"
    p1.legend.location = "top_left"
    p1.ygrid.band_fill_alpha = 0.2
    show(p1)

# -----------------------------------------
# Save file


def save_as(model_name, prediction, save_path, df2):
    df2['TARGET_' + model_name] = prediction
    df2.to_csv(save_path + model_name + '.csv')

# -----------------------------------------
# -----------------------------------------
# Extras

# Trick to find out the % of types within df
# df2.groupby([RegressionCol]).size()/len(df)

# Neat trick to get all object types in df as a dict
# g = x2_df.columns.to_series().groupby(df2[RegressionCol]).groups
# {k.name: v for k,v in g.items()}

# -----------------------------------------
# -----------------------------------------

# old
# model_pca = decomposition.PCA()
# model_Kfeatures = feature_selection.SelectKBest()

# super_machine_learner = [ xgb.XGBRegressor()]
# machine_learner = xgb.XGBRegressor()

# process = [
#     ('pca', model_pca),
#     ('select_best', model_Kfeatures),
#     ('ml', machine_learner)
# ]

# -----------------------------------------
# -----------------------------------------

# full_pipeline needs the following:
# 1) x_scaled, x_df, y_df: passed from clean_data
# 2) pipeline: sklearn module
# 3) model_selection: sklearn module
# 4) process: a list, contains the full pipeline process
# in linear order eg: pca -> kbest -> ml_algo
# process = [
#    ('pca', model_pca), // [0]
#   ('select_best', model_variable_best), // [1]
#   ('ml', algo)]
# Where model_pca = decomposition.PCA()
# & model_variable_best = feature_selection.SelectKBest()
# & machine_learner = linear_model.RANSACRegressor()
# & finally line it up to execute
# pipeline_process = pipeline.Pipeline(process)

# 5) search_space: hyperparameter keys modifier for GridSearchCV
# search_space = dict(
#         pca__n_components=n_pca,
#         select_best__k=n_select_best,
#         ml__n_estimators=n_trees
#         )
# References:
# [0] http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# [1] http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
