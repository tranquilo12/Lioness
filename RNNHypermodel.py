# A class which has some helpful functions for 
# 1. Reshaping data 
# 2. plotting data that comes within the inteest of that model

class Model_Maker(object):

	def __init__(self, 
		train_path, test_path, 
		start, split, end, 
		scale, 
		):
		# check if paths are valid paths to the file object
		self.train_path = train_path
		self.test_path = test_path

		# check if start, split, end are datetime objects
		if isinstance(start, datetime.date)
		self.start = start
		self.split = split
		self.end = end 