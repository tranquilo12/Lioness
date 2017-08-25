# A class which has some helpful functions for
# 1. Reshaping data
# 2. plotting data that comes within the inteest of that model
# 3. Plotting any general pair of data as long as x_axis_type = datetime

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

		self.scale = scale
		self.scaler = scaler

		# instantiate each dataframe for self
		self.data = {}
		self.data['x_df'] = x_df
		self.data['y_df'] = y_df

		# init numpy arrays as none, and verify if function has
		# excuted/executed correctly through their values
		# x_train indicates a non-scaled x_df converted to a numpy array
		self.data['x_train'] = None
		self.data['y_train'] = None

		self.data['x_val'] = None
		self.data['y_val'] = None

		# need indices for the split data
		# use the same for scaled and non-scaled
		self.data['x_train_index'] = None
		self.data['y_train_index'] = None

		# xs_train indicates a scaled x_train
		self.data['xs_train'] = None
		self.data['ys_train'] = None

		self.data['xs_val'] = None
		self.data['ys_val'] = None

		# somethings for the plotter
		self.shifted = {}
		self.shifted['X'] = None
		self.shifted['Y'] = None
		self.shifted['X_index'] = None
		self.shifted['Y_index'] = None

		# variable for lookback
		self.lookback = 1

		# check if start, split, end are datetime objects
		self.dates = {}

		if isinstance(start, datetime.date):
			self.dates['start'] = start
		else:
			self.dates['start'] = datetime.datetime.strptime(
				start, '%Y-%M-%d').date()
		if isinstance(split, datetime.date):
			self.dates['split'] = split
		else:
			self.dates['split'] = datetime.datetime.strptime(
				split, '%Y-%M-%d').date()
		if isinstance(end, datetime.date):
			self.dates['end'] = end
		else:
			self.dates['end'] = datetime.datetime.strptime(
				end, '%Y-%M-%d').date()

		# make sure an object can be instantiated
		# not sure super instantiation is needed
		# super(DataTamer, self).__init__()

	def split_data(self):
		# split the dataframe, convert it to a numpy array and send it
		# back with its indices
		x_train = self.data['x_df'][self.dates['start']:self.dates['split']]
		y_train = self.data['y_df'][self.dates['start']:self.dates['split']]

		self.data['x_train'] = x_train
		self.data['y_train'] = y_train

		x_val = self.data['x_df'][self.dates['split']:self.dates['end']]
		y_val = self.data['y_df'][self.dates['split']:self.dates['end']]

		self.data['x_val'] = x_val
		self.data['y_val'] = y_val

		self.data['x_train_index'] = x_train.index.values
		self.data['y_train_index'] = y_train.index.values

		self.data['x_val_index'] = x_val.index.values
		self.data['y_val_index'] = y_val.index.values

		# check if scaling is required, if not, dont use it
		if self.scale == True:
			# find the actual length of the date range
			train_set_len = len(
				self.data['x_df'][self.date['start']:self.date['split']])
			val_set_len = len(
				self.data['y_df'][self.date['start']:self.date['split']])

			# convert the dataframes to numpy arrays
			x_np = self.data['x_df'].values.astype('float32')
			x_npIndex = self.data['x_df'].index.values

			y_np = self.data['y_df'].values.astype('float32')
			y_npIndex = self.data['y_df'].index.values

			# transform the numpy arrays, according to the scaler
			xs = self.scaler.fit_transform(x_np)
			ys = self.scaler.fit_transform(y_np)

			# reshape them as necessary, splitting them into training,validation
			self.data['xs_train'] = xs[0:train_set_len]
			self.data['ys_train'] = ys[0:train_set_len]

			self.data['xs_val'] = xs[train_set_len:(
				train_set_len + val_set_len)]
			self.data['ys_val'] = ys[train_set_len:(
				train_set_len + val_set_len)]

		return(self)

	def shift(self, dataset=None):
		# X, and Y can be thought of splitting the time dimensions of the dataset
		# that is to say that they cannot be plotted egainst each other,
		# but signify a shift in the rows of the data
		# that is then split at that point
		# here X=t, Y=t+1
		self.shifted['X'], self.shifted['Y'] = [], []
		self.shifted['X_index'], self.shifted['Y_index'] = [], []

		if dataset == None:
			print('No dataset mentioned (y_train, y_test)')
			return(0)

		if dataset == 'y_train':
			for i in range(len(self.data['y_train']) - self.lookback - 1):
				self.shifted['X'].append(
					self.data['y_train'][i:(i + self.lookback), ])
				self.shifted['X_index'].append(
					self.data['y_train_index'][i:(i + self.lookback), ])
				self.shifted['Y'].append(
					self.data['y_train'][i + self.lookback, ])
				self.shifted['Y_index'].append(
					self.data['y_train_index'][i + self.lookback, ])

		if dataset == 'y_val':
			for i in range(len(self.data['y_val']) - self.lookback - 1):
				self.shifted['X'].append(
					self.data['y_val'][i:(i + self.lookback), ])
				self.shifted['X_index'].append(
					self.data['y_val_index'][i:(i + self.lookback), ])
				self.shifted['Y'].append(
					self.data['y_val'][i + self.lookback, ])
				self.shifted['Y_index'].append(
					self.data['y_val_index'][i + self.lookback, ])

		# Convert to numpy arrays
		self.shifted['X'] = np.array(self.shifted['X'])
		self.shifted['Y'] = np.array(self.shifted['Y'])
		self.shifted['X_index'] = np.array(self.shifted['X_index'])
		self.shifted['Y_index'] = np.array(self.shifted['Y_index'])

		return(self)

	# make corrections: 1) Ensure atleast one pair is passed as arg
	def bokplotGen(self, *args, **kwargs):
		from bokeh.io import output_notebook
		from bokeh.plotting import figure, show, output_file
		output_notebook()

		p1 = figure(title='Placeholder',
					x_axis_type='datetime',
					plot_width=1400,
					plot_height=400,
					background_fill_color='#EFE8E2')

		keys = sorted(list(kwargs.keys()))
		# the keys now look like ['x0','x1', 'y0', 'y1']

		index, amount, legend = [], [], []

		# first append indices into the large index array
		for i in range(int(len(keys) / 2)):
			if isinstance(kwargs[keys[i]][0], datetime.date):
				index.append(kwargs[keys[i]])
				continue
			else:
				print('x' + str(i) + ' axis not of type datetime, try again')

		# necxt values
		for i in range(int(len(keys) / 2), len(keys)):
			if isinstance(kwargs[keys[i]], np.ndarray):
				amount.append(kwargs[keys[i]])
				legend.append(str(keys[i]))
				continue
			else:
				print('y' + str(i) + ' axis not of type np.array, try again')
				return(0)

		for i in range(len(index)):
			p1.line(index[i], amount[i], legend=legend[i], color=mypalette[i])

		# aesthetic mapping
		p1.grid.grid_line_alpha = 0.1
		p1.xaxis.axis_label = "Date"
		p1.yaxis.axis_label = "Price"
		p1.legend.location = "top_left"
		p1.ygrid.band_fill_alpha = 0.2
		show(p1)
		return(self)

	def bokplot(self, bokeh=bokeh, title='Placeholder'):
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
			if self.data['ys_train'] is not None:
				if isinstance(self.data['ys_train'], self.pd.DataFrame):
					local_y_train = self.data['ys_train'].values.astype(
						'float32')
					p1.line(self.data['y_train_index'],
							local_ys_train.reshape(len(local_ys_train)),
							color='#E08E79',
							legend='ys_train')
				else:  # if it's a np array
					p1.line(self.data['y_train_index'],
							local_y_train.reshape(len(self.data['ys_train'])),
							color='#E08E79',
							legend='ys_train')

			# check if ys_val exists, and plot if yes
			if self.data['ys_val'] is not None:
				if isinstance(self.data['ys_val'], self.pd.DataFrame):
					local_ys_val = self.data['ys_val'].values.astype('float32')
					p1.line(self.data['y_val_index'],
							local_ys_val.reshape(len(local_ys_val)),
							color='#3B8686',
							legend='ys_val')
				else:  # if it's an np.array
					p1.line(self.data['y_val_index'],
							local_ys_val.reshape(len(self.data['ys_val'])),
							color='#3B8686',
							legend='ys_val')

		if self.scale == False:
			if self.data['y_train'] is not None:
				if isinstance(self.data['y_train'], self.pd.DataFrame):
					local_y_train = self.data['y_train'].values.astype(
						'float32')
					p1.line(self.data['y_train_index'],
							local_y_train.reshape(len(local_y_train)),
							color='#E08E79',
							legend='y_train')
				else:  # if it's an np.array
					p1.line(self.data['y_train_index'],
							local_y_train.reshape(len(local_y_train)),
							color='#E08E79',
							legend='y_train')

			if self.data['y_val'] is not None:
				if isinstance(self.data['y_val'], self.pd.DataFrame):
					local_y_val = self.data['y_val'].values.astype('float32')
					p1.line(self.data['y_val_index'],
							local_y_val.reshape(len(local_y_val)),
							color='#3B8686',
							legend='y_val')
				else:  # if it's an np.array
					p1.line(self.data['y_val_index'],
							local_y_val.reshape(len(local_y_val)),
							color='#3B8686',
							legend='y_val')

		# aesthetic mapping
		p1.grid.grid_line_alpha = 0.1
		p1.xaxis.axis_label = "Date"
		p1.yaxis.axis_label = "Price"
		p1.legend.location = "top_left"
		p1.ygrid.band_fill_alpha = 0.2
		show(p1)
