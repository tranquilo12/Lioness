# A class which has some helpful functions for
# 1. Reshaping data
# 2. plotting data that comes within the inteest of that model


class DataTamer(object):

	def __init__(self,
		x_df, y_df,
		start, split, end,
		scale, scaler=scaler,
		datetime=datetime,
		pd=pd, 
		bokeh=bokeh):

		self.datetime = datetime
		self.pd = pd
		self.bokeh = bokeh
		
		self.bokeh.output_notebook()

		# instanciate each dataframe for self
		self.x_df = x_df
		self.y_df = y_df
		self.scale = scale
		self.scaler = scaler

		# init them as none, and check for them to verify if function has
		# excuted/executed correctly
		# xs_train indicates a scaled x_train
		self.xs_train = None
		self.ys_train = None
		self.xs_val = None
		self.ys_val = None

		# x_train is just split without scaling
		self.x_train = None
		self.y_train = None
		self.x_val = None
		self.y_val = None

		# need indices for the split data
		self.x_train_index = None
		self.y_train_index = None

		# variable for lookback
		self.lookback = 1

	
		# check if start, split, end are datetime objects
		if isinstance(start, datetime.date):
			self.start = start
		else:
			self.start = datetime.strptime(start, '%Y-%M-%d').date()
		if isinstance(split, datetime.date):
			self.split = split
		else:
			self.split = datetime.strptime(split, '%Y-%M-%d').date()
		if isinstance(end, datetime.date):
			self.end = end
		else:
			self.end = datetime.strptime(end, '%Y-%M-%d').date()

	def split_data(self):
		# split the dataframe, convert it to a numpy array and send it
		# back with its indices
		self.x_train, self.y_train = self.x_df[self.start:self.split], self.y_df[self.start:self.split]
		self.x_val, self.y_val = self.x_df[self.split:self.end], self.y_df[self.split:self.end]

		self.x_train_index = self.x_train.index.values
		self.y_train_index = self.y_train.index.values

		# check if scaling is required, if not, dont use it
		if self.scale == True:
			# find the actual length of the date range
			train_set_len = len(self.x_df[self.start:self.split])
			val_set_len = len(self.y_df[self.split:self.end])

			# convert the dataframes to numpy arrays
			x_np, x_npIndex = self.x_df.values.astype('float32'), self.x_df.index.values
			y_np, y_npIndex = self.y_df.values.astype('float32'), self.y_df.index.values

			# transform the numpy arrays, according to the scaler
			xs, ys = scaler.fit_transform(x_np), scaler.fit_transform(y_np)

			# reshape them as necessary, splitting them into training,validation
			self.xs_train, self.ys_train = xs[0:train_set_len], ys[0:train_set_len]
			self.xs_val = xs[train_set_len:(train_set_len + val_set_len)]
			self.ys_val = ys[train_set_len:(train_set_len + val_set_len)]
			return(self.xs_train, self.xs_val, self.x_train_index,
					self.ys_train, self.ys_val, self.y_train_index)

		return(self.x_train.values.astype('float32'),
			self.x_val, .values.astype('float32'),
			self.x_train_index,
			self.y_train.values.astype('float32'),
			self.y_val.values.astype('float32'),
			self.y_train_index)

	def shift(self, dataset=None):
		# X, and Y can be thought of splitting the time dimensions of the dataset
	    # that is to say that they cannot be plotted egainst each other,
	    # but signify a shift in the rows of the data
	    # that is then split at that point
	    X, Y = [], []
	    X_index, Y_index = [], []
	    if dataset==None:
	    	print('No dataset mentioned')
	    	return(0)
	    if dataset=='y_train':
		    for i in range(len(self.y_train) - self.lookback - 1):
		        X.append(self.y_train[i:(i + self.lookback),])
		        X_index.append(self.y_train_index[i:(i + self.lookback),])
		        
		        Y.append(self.y_train[i + self.lookback,])
		        Y_index.append(y_train_index[i + self.lookback,])
	    return(np.array(X), np.array(X_index), np.array(Y), np.array(Y_index))


	def bokplot(self):
		from self.bokeh.plotting import figure, show, output_file
	    p1 = figure(x_axis_type="datetime", 
	    	title=title, 
	    	plot_width=1400, 
	    	plot_height=400, 
	    	background_fill_color="#EFE8E2")

	    if self.y_train is not None: 
	    	p1.line(self.y_train_index, self.y_train, color='#E08E79', legend='y_train')
	    if self.y_val is not None: 
	    	p1.line(self.y_val_index, self.y_val, color='#3B8686', legend='y_val')
	    
	    # aesthetic mapping
	    p1.grid.grid_line_alpha=0.1
	    p1.xaxis.axis_label = "Date"
	    p1.yaxis.axis_label = "Price"
	    p1.legend.location = "top_left"
	    p1.ygrid.band_fill_alpha = 0.2
	    show(p1)