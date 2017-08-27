### Class Lioness
---
##### Methods
- `__init__(self, x_df, y_df, start, split, end, scale, scaler, datetime, pd, bokeh)`
Instanciates a class with the following variables:
    - **self.datetime, self.pd, self.bokeh** (datetime.datetime object, pandas and bokeh objects passed into the class).
    - **self.scale, self.scaler,** where scale = bool, and scaler is a __sklearn.Preprocessing.MinMaxScaler__ executable.
    - **self.data** [dict] contains keys :
        - **x_df, y_df,** each instances of the output from **Methods.clean**.
        - **xs_train, ys_train, xs_val, ys_val,** when scale = True, **x_df** is converted to a numpy array and a __MinMaxScaler__ is applied.
        - **x_train, y_train, x_val, y_val,** when scale = False, **x_df** is converted to a numpy array only. No scaler is applied.
        - **X, Y, X_index, Y_index,** for plotting, with indices of the split data preserved.
    - **self.shifted** of type(dict) contains keys:
        - **X**, **Y**, **X_index**, **Y_index** for the lookback dataset
    - **self.dates** of type(dict) contains keys:
        - **start**, **split**, **end** which split the data into the specified dates

- `split_data(self, scale=bool)`
Splits data into two sets in ranges [start:split] and [split:end]
    - If scale is set to `True`, then the data is scaled, then split up to preserve continuity. Variables such as **xs_train,val**, **ys_train,val** [in the dict 'data'] are initialised and can be used.
    - If scale is set to `False`, the data is just split, and variables such as **xs_train,val** and **ys_train,val** [in 'data'] are not initialised at all.
- `shift(self, dataset='Placeholder')`
A lookback method, which results in X, and Y, that can be thought of splitting the time dimensions of     the dataset. That is to say that they cannot be plotted egainst each other, but signify a shift in the     rows of the data that is then split at that point here X = t, Y = t+1. 
    - The 'dataset' variable can be 'y_train' or 'y_val' of type(str).
    - Makes a new set of arrays that 'lookback' on the main dataset as directed by the lookback variable.
    - The default lookback variable is 1.
    - Larger lookbacks crash the program due to O^n write complexity.
    - 