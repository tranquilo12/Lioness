# A class which has some helpful functions for
# 1. Reshaping data
# 2. plotting data that comes within the inteest of that model


class Lioness(object):

	def __init__(self,
		x_df, y_df,
		start, split, end,
		scale, scaler,
		datetime,
		pd,
		bokeh):

		self.datetime = datetime
		self.pd = pd
		self.bokeh = bokeh

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

		# somethings for the plotter
		self.X = None
		self.Y = None
		self.X_index = None
		self.Y_index = None

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
			self.start = datetime.datetime.strptime(start, '%Y-%M-%d').date()
		if isinstance(split, datetime.date):
			self.split = split
		else:
			self.split = datetime.datetime.strptime(split, '%Y-%M-%d').date()
		if isinstance(end, datetime.date):
			self.end = end
		else:
			self.end = datetime.datetime.strptime(end, '%Y-%M-%d').date()

		# make sure an object can be instantiated
		super(DataTamer, self).__init__()

	def split_data(self):
		# split the dataframe, convert it to a numpy array and send it
		# back with its indices
		self.x_train, self.y_train = self.x_df[self.start:
			self.split], self.y_df[self.start:self.split]
		self.x_val, self.y_val = self.x_df[self.split:
			self.end], self.y_df[self.split:self.end]

		self.x_train_index = self.x_train.index.values
		self.y_train_index = self.y_train.index.values

		self.x_val_index = self.x_val.index.values
		self.y_val_index = self.y_val.index.values

		# check if scaling is required, if not, dont use it
		if self.scale == True:
			# find the actual length of the date range
			train_set_len = len(self.x_df[self.start:self.split])
			val_set_len = len(self.y_df[self.split:self.end])

			# convert the dataframes to numpy arrays
			x_np, x_npIndex = self.x_df.values.astype('float32'), self.x_df.index.values
			y_np, y_npIndex = self.y_df.values.astype('float32'), self.y_df.index.values

			# transform the numpy arrays, according to the scaler
			xs, ys = self.scaler.fit_transform(x_np), self.scaler.fit_transform(y_np)

			# reshape them as necessary, splitting them into training,validation
			self.xs_train, self.ys_train = xs[0:train_set_len], ys[0:train_set_len]
			self.xs_val = xs[train_set_len:(train_set_len + val_set_len)]
			self.ys_val = ys[train_set_len:(train_set_len + val_set_len)]

		return(self)

	def shift(self, dataset=None):
		# X, and Y can be thought of splitting the time dimensions of the dataset
		# that is to say that they cannot be plotted egainst each other,
		# but signify a shift in the rows of the data
		# that is then split at that point
		# here X=t, Y=t+1
		self.X, self.Y = [], []
		self.X_index, self.Y_index = [], []

		if dataset == None:
			print('No dataset mentioned')
			return(0)

		if dataset == 'y_train':
			for i in range(len(self.y_train) - self.lookback - 1):
				self.X.append(self.y_train[i:(i + self.lookback), ])
				self.X_index.append(self.y_train_index[i:(i + self.lookback), ])
				self.Y.append(self.y_train[i + self.lookback, ])
				self.Y_index.append(y_train_index[i + self.lookback, ])

		if dataset == 'y_val':
			for i in range(len(self.y_val) - self.lookback - 1):
				self.X.append(self.y_val[i:(i + self.lookback), ])
				self.X_index.append(self.y_val_index[i:(i + self.lookback), ])
				self.Y.append(self.y_val[i + self.lookback, ])
				self.Y_index.append(y_val_index[i + self.lookback, ])

		self.X = np.array(self.X)
		self.Y = np.array(self.Y)
		self.X_index = np.array(self.X_index)
		self.Y_index = np.array(self.Y_index)
		return(self)

	def bokplot(self, title='Placeholder'):

		from bokeh.io import output_notebook
		from bokeh.plotting import figure, show, output_file

		output_notebook()

		p1 = figure(title=title,
					x_axis_type="datetime",
					plot_width=1400,
					plot_height=400,
					background_fill_color="#EFE8E2")

		# check if scaling has been applied
		# use the same indices for both cases
		if self.scale == True:
			# check if ys_train exists, and plot if yes
			if self.ys_train is not None:
				if isinstance(self.ys_train, self.pd.DataFrame):
					local_y_train = self.ys_train.values.astype('float32')
					p1.line(self.y_train_index,
							local_ys_train.reshape(len(local_ys_train)),
							color='#E08E79',
							legend='ys_train')
				else:  # if it's a np array
					p1.line(self.y_train_index,
							self.ys_train.reshape(len(self.ys_train)),
								color='#E08E79',
								legend='ys_train')

			# check if ys_val exists, and plot if yes
			if self.ys_val is not None:
				if isinstance(self.ys_val, self.pd.DataFrame):
					local_ys_val=self.ys_val.values.astype('float32')
					p1.line(self.y_val_index,
							local_ys_val.reshape(len(local_ys_val)),
							color='#3B8686',
							legend='ys_val')
				else:  # if it's an np.array
					p1.line(self.y_val_index,
							self.ys_val.reshape(len(self.ys_val)),
							color='#3B8686',
							legend='ys_val')

		if self.scale == False:

			if self.y_train is not None:
				if isinstance(self.y_train, self.pd.DataFrame):
					local_y_train=self.y_train.values.astype('float32')
					p1.line(self.y_train_index,
							local_y_train.reshape(len(local_y_train)),
							color='#E08E79',
							legend='y_train')
				else:  # if it's an np.array
					p1.line(self.y_train_index,
							self.y_train.reshape(len(local_y_train)),
							color='#E08E79',
							legend='y_train')

			if self.y_val is not None:
				if isinstance(self.y_val, self.pd.DataFrame):
					local_y_val=self.y_val.values.astype('float32')
					p1.line(self.y_val_index,
							local_y_val.reshape(len(local_y_val)),
							color='#3B8686',
							legend='y_val')
				else:  # if it's an np.array
					p1.line(self.y_val_index,
							self.y_val.reshape(len(local_y_val)),
							color='#3B8686',
							legend='y_val')


		# aesthetic mapping
		p1.grid.grid_line_alpha=0.1
		p1.xaxis.axis_label="Date"
		p1.yaxis.axis_label="Price"
		p1.legend.location="top_left"
		p1.ygrid.band_fill_alpha=0.2
		show(p1)