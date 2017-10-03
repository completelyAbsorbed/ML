# lastima.py
# lstm wrapper and tools
# some stuff from machinelearningmastery, particularly : https://machinelearningmastery.com/handle-missing-timesteps-sequence-prediction-problems-python/
 # https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
 
from random import random
from numpy import array
from pandas import concat
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
 
# generate a sequence of random values
def generate_sequence(n_timesteps):
	return [random() for _ in range(n_timesteps)]
 
# generate data for the lstm
def generate_data(n_timesteps):
	# generate sequence
	sequence = generate_sequence(n_timesteps)
	sequence = array(sequence)
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	# replace missing values with -1
	df.fillna(-1, inplace=True)
	values = df.values
	# specify input and output data
	X, y = values, values[:, 1]
	# reshape
	X = X.reshape(len(X), 2, 1)
	y = y.reshape(len(y), 1)
	return X, y
 
# remove missing

# mark missing, model learns their meaning

# mask missing, make model ignore them 

# convert a time series to supervised learning format
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
 
raw = DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
data = series_to_supervised(values, 1, 2)
print(data)