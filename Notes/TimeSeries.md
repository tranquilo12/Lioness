TF-K-RNN-LSTM-TS
Tags: (Tensorflow, Keras, RNN, LSTM, TimeSeries)

##Tensorflow, Keras and input sizes of arrays in LSTM's

Keeping in mind the data is Time Series numpy arrays, with shape (1000, 1), we are also assuming that samples=rows, dimentions=columns=features.

The LSTM is defined as LSTM(output_neurons, input_shape/batch_input_shape)
input_shape matches X_train shape (nb_samples, timesteps, input_dims). 

If the number of input_dims > 1, its a multivariate time series. 

Where:
- nb_samples = num of rows
- timesteps 
- input_dims = num of columns

For example, the below table has a shape (2, 1, 12) as a tensor (excluding headers).

-------------------------------------------------------------------------------

| D/T  | COL2 | COL3 | COl4 | COL5 | COL6 | COl7 | COL8 | COL9 | COl10 | COL11
|:--
 1.00 | 1.87 | 1.99 | 1.00 | 1.87 | 1.99 | 1.00 | 1.87 | 1.99 | 1.00  | 1.87  | 1.00 | 1.87 | 1.99 | 1.00 | 1.87 | 1.99 | 1.00 | 1.87 | 1.99 | 1.00  | 1.87

----------------------------------

If we look into one sample(row) of X_train, it consists of 2 dimensions, 
timesteps & input_dims, which makes it (1, 12) 

-----------

1.00 | 1.87 | 1.99 | 1.00 | 1.87 | 1.99 | 1.00 | 1.87 | 1.99 | 1.00  | 1.87  |

------------



(Reshaping)[reshape link] is necessary, where input_shape has to be a 3D tensor (x, y, z).
Where:
- x = num of rows
- y = timesteps
- z = num of columns

timesteps: the number of steps that the algo will predict. 
For instance, if your one dimensional data is originally shaped at (300,1), 
you can reshape it to (300,1,1), where the y index = timesteps and it is assured that your LSTM neuron will now try and predict one timestep ahead.

It is unclear to me if the timesteps actually indicate movement through rows or movement through columns. This needs to be read into further, but I have got no answers till now.


The LSTM will then be defined as LSTM(1, input_shape=(lookback, data[1]))
Here lookback refers 


First:
- output_neurons is the number of neurons you want this LSTM layer to feed into the next layer
- input_shape corresponds to the shape of the training array

Lets suppose your training set