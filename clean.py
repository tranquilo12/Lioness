

def clean(path=None,
          RegressionCol=False,
          ClassificationCol=False,
          pd=None):
    
    """
    Note: 

    if cleaning a training file, make sure that either the classification column
    or the regression col name is specified. NOT BOTH. Use one, and keep the other
    as false.
    
    if cleaning a test file, leave BOTH column names as false. 

    pass pd (import pandas as pd)  
    """

    # todo -> find from path if its a train or test
    #
    if path == None:
        print('No path provided for data... exiting...')
        return 0

    df = pd.read_csv(path, header=0)
    df['DATETIME'] = df.DATE.map(str) + df.TIME.map(str)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.index = df.DATETIME
    old_shape = df.shape

    print('\n--- Clean ---')

    #
    # drop unwanted columns for df's
    # and get y_df ready for specifc instances
    #
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
    # all drops consolidated here
    #
    x_df = df.drop(dropcols, axis=1)
    x_df = x_df.dropna(axis=1)
    if 'USD/JPY.H1.CORREL' in df.columns:
        x_df = x_df.drop(labels='USD/JPY.H1.CORREL', axis=1)

    #
    # validate df shapes
    #
    if (RegressionCol is not False) and (ClassificationCol is not False):
        if x_df.shape[0] != y_df.shape[0]:
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
        pass
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

    # replace some dodgy column labels with ones that dont throw errors
    x_df.columns = pd.Series(x_df.columns).str.replace('[', '((')
    x_df.columns = pd.Series(x_df.columns).str.replace(']', '))')

    return(x_df, y_df, df)
