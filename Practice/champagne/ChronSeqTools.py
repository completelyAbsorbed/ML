# ChronSeqTools.py - Chrono Sequence Tools : tools for Machine Learning with Time Series
# some code copied or adapted from https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/
from pandas import Series
from pandas import TimeGrouper
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA 
from statsmodels.tsa.arima_model import ARIMAResults
import numpy
import warnings

# create ACF and PACF plots
def autocorrelation_plots(dataset):
	pyplot.figure()
	pyplot.subplot(211)
	plot_acf(dataset, ax=pyplot.gca())
	pyplot.subplot(212)
	plot_pacf(dataset, ax=pyplot.gca())
	pyplot.show()

# print dataset summary and show line, seasonal line, density, box-and-whisker plots
def summaries(dataset, group_start, group_end, tg_param):
	print
	print "  Summary Statistics :"
	print
	print dataset.describe()
	makespace()
	# line plot
	dataset.plot()
	pyplot.show()
	# seasonal line plots
	groups = dataset[group_start:group_end].groupby(TimeGrouper(tg_param))
	years = DataFrame()
	pyplot.figure()
	i = 1
	n_groups = len(groups)
	for name, group in groups:
		pyplot.subplot((n_groups*100) + 10 + i)
		i += 1
		pyplot.plot(group)
	pyplot.show()
	# density plot
	pyplot.figure(1)
	pyplot.subplot(211)
	dataset.hist()
	pyplot.subplot(212)
	dataset.plot(kind='kde')
	pyplot.show()
	# box-and-whisker plots 
	years = DataFrame()
	for name, group in groups:
		years[name.year] = group.values
	years.boxplot()
	pyplot.show()

# persistence model for performance baseline
def persistence(train, test, print_flag = True):
	# walk-forward validation
	if print_flag:
		print '  Persistence : '
		print
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		# predict
		yhat = history[-1]
		predictions.append(yhat)
		# observation
		obs = test[i]
		history.append(obs)
		if print_flag:
			print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
	# report performance
	rmse = RMSE(test, predictions, print_flag = False)
	if print_flag:
		print
		print('Persistence RMSE: %.3f' % rmse)
	return rmse

# prepare data for train test split
def train_test_split(dataset, train_size_factor = 0.5):
	X = dataset.values
	X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	return train, test

# Root Mean Squared Error function
def RMSE(test, predictions, print_flag = True):
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	if print_flag:
		print('RMSE: %.3f' % rmse)
	return rmse
# make some space to make output easier to read 
def makespace(lines = 5, space_print = ''):
	for x in range(0, lines):
		print space_print
	
# load data and return/write training, validation sets
def load(filename, prefix_split, n_last_periods):
	series = Series.from_csv(filename, header=0)
	split_point = len(series) - n_last_periods
	dataset, validation = series[0:split_point], series[split_point:]
	print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
	dataset.to_csv(prefix_split + 'Training.csv')
	validation.to_csv(prefix_split + 'Validation.csv')
	return dataset, validation


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		# difference data
		months_in_year = 12
		diff = difference(history, months_in_year)
		model = ARIMA(diff, order=arima_order)
		model_fit = model.fit(trend='nc', disp=0)
		yhat = model_fit.forecast()[0]
		yhat = inverse_difference(history, yhat, months_in_year)
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = RMSE(test, predictions, print_flag = False)
	return rmse
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s RMSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
