# champagne.py
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
import numpy
import warnings


# https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/

series = Series.from_csv('champagne.csv', header=0)
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict 
	yhat = history[-1]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%3.f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)

print dataset.describe()

## look at the data in a line plot
#dataset.plot()
#pyplot.show()
## look at seasonal line plots
#groups = dataset['1964':'1970'].groupby(TimeGrouper('A'))
#years = DataFrame()
#pyplot.figure()
#i = 1
#n_groups = len(groups)
#for name, group in groups:
#	pyplot.subplot((n_groups*100) + 10 + i)
#	i += 1
#	pyplot.plot(group)
#pyplot.show()
## Density Plot
#pyplot.figure(1)
#pyplot.subplot(211)
#series.hist()
#pyplot.subplot(212)
#dataset.plot(kind='kde')
#pyplot.show()
## Box and Whisker Plots
#groups = dataset['1964':'1970'].groupby(TimeGrouper('A'))
#years = DataFrame()
#for name, group in groups:
#	print name
#	#print group
#	years[name.year] = group.values
#years.boxplot()
#pyplot.show()
## deseasonalize the data 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i-interval]
		diff.append(value)
	return Series(diff)
	
#X = dataset.values
#X = X.astype('float32')	
# difference data
#months_in_year = 12
#stationary = difference(X, months_in_year)
#stationary.index = dataset.index[months_in_year:]
# check if stationary
#result = adfuller(stationary)
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])
#print('Critical Values:')
#for key, value in result[4].items():
#	print('\t%s %.3f' % (key, value))
# save
#stationary.to_csv('stationary.csv')
# plot
#stationary.plot()
#pyplot.show()
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


def makespace(space=5):
	for x in range(0,space):
		print('...')
	return 

## examine ACF and PACF plots to determine AR and MA parameters for ARIMA
#series = Series.from_csv('stationary.csv')
#pyplot.figure()
#pyplot.subplot(211)
#plot_acf(series, ax=pyplot.gca())
#pyplot.subplot(212)
#plot_pacf(series, ax=pyplot.gca())
#pyplot.show()

## demonstrate performance of the ARIMA model on the test harness
#makespace()
# load data
#series = Series.from_csv('dataset.csv')
# prepare data
#X = series.values
#X = X.astype('float32')
#train_size = int(len(X) * 0.50)
#train, test = X[0:train_size], X[train_size:]
# walk-forward validation
#history = [x for x in train]
#print history 
#makespace()
#predictions = list()
#for i in range(len(test)):
#	# difference data
#	months_in_year = 12
#	diff = difference(history, months_in_year)
#	# predict
#	model = ARIMA(diff.as_matrix(), order=(1,1,1))
#	model_fit = model.fit(trend='nc', disp=0)
#	yhat = model_fit.forecast()[0]
#	yhat = inverse_difference(history, yhat, months_in_year)
#	predictions.append(yhat)
#	# observation
#	obs = test[i]
#	history.append(obs)
#	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
#se = mean_squared_error(test, predictions)
#rmse = sqrt(mse)
#print('RMSE: %.3f' % rmse)
## grid search ARIMA 
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
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
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
 
# load dataset
series = Series.from_csv('dataset.csv')
# evaluate parameters
p_values = range(0, 7)
d_values = range(0, 3)
q_values = range(0, 7)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)









# log what step I'm on here : 6.3
