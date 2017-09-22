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




# run ARIMA and review residual error. a wrapper function
def examine_residual_error(p, d , q , dataset, train_size_factor, trend , months_in_year = 12, print_flag = True):
	return run_arima(p=p, d=d, q=q, dataset=dataset, train_size_factor=train_size_factor, trend=trend, months_in_year=months_in_year, print_flag=print_flag, return_residual=True)


# preform grid search on ARIMA, return best configuration, best score
def run_grid_search_arima(p_grid, d_grid, q_grid, dataset, train_size_factor, trend, months_in_year = 12, full_print_flag = False):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_grid:
		for d in d_grid:
			for q in q_grid:
				order = (p,d,q)
				try:
					rmse = run_arima(p=p, d=d, q=q, dataset=dataset, train_size_factor=train_size_factor, trend=trend, months_in_year=months_in_year, print_flag = full_print_flag, return_residual = False)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))	
	return best_score, best_cfg

# do walk-forward ARIMA, return rmse 
def run_arima(p, d, q, dataset, train_size_factor = 0.5, trend = 'nc', months_in_year = 12, print_flag = True, return_residual = False):
	# load data
	series = dataset
	# prepare data
	X = series.values
	X = X.astype('float32')
	train_size = int(len(X) * train_size_factor)
	train, test = X[0:train_size], X[train_size:]
	# walk-forward validation
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		# difference data
		#months_in_year = months_in_year
		diff = difference_pd(history, months_in_year)
		# predict
		model = ARIMA(diff, order=(p,d,q))
		model_fit = model.fit(trend=trend, disp=0)
		yhat = model_fit.forecast()[0]
		yhat = inverse_difference_pd(history, yhat, months_in_year)
		predictions.append(yhat)
		# observation
		obs = test[i]
		history.append(obs)
		if print_flag:
			print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
	# report performance
	rmse_arima = RMSE(test, predictions, print_flag)
	if return_residual:
		residuals = [test[i]-predictions[i] for i in range(len(test))]
		residuals = DataFrame(residuals)
		if print_flag:
			print(residuals.describe())
		# plot
		pyplot.figure()
		pyplot.subplot(211)
		residuals.hist(ax=pyplot.gca())
		pyplot.subplot(212)
		residuals.plot(kind='kde', ax=pyplot.gca())
		pyplot.show()
		return residuals
	else:
		return rmse_arima
	
# create a differenced ndarray
def difference_pd(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value...
def inverse_difference_pd(history, yhat, interval=1):
	return yhat + history[-interval]	

# compute stationary, show line plot 
def make_stationary(dataset, months_in_year, filename, show_plot = True):
	X = dataset.astype('float32')
	stationary = difference(X, months_in_year)
	stationary.index = dataset.index[months_in_year:]
	# check if stationary
	result = adfuller(stationary)
	print('ADF Statistic: %f' % result[0])
	print('p-value: %f' % result[1])
	print('Critical Values:')
	for key, value in result[4].items():
		print('\t%s: %.3f' % (key, value))
	# save
	stationary.to_csv(filename)
	# plot
	if show_plot:
		stationary.plot()
		pyplot.show()
	return stationary

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
	return Series(diff)
 
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
