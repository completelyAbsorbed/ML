# champagne.py
from pandas import Series
from pandas import TimeGrouper
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot



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
groups = dataset['1964':'1970'].groupby(TimeGrouper('A'))
years = DataFrame()
for name, group in groups:
	print name
	#print group
	years[name.year] = group.values
years.boxplot()
pyplot.show()


# log what step I'm on here : 6