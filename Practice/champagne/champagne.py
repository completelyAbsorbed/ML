# champagne.py
from pandas import Series


# https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/

series = Series.from_csv('champagne.csv', header=0)
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# start at step 3