# Load CSV using Pandas from URL

import pandas

#from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing as preprocessing
import numpy



url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)

array = data.values

# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]

min_max_scaler = preprocessing.MinMaxScaler()
rescaledX = min_max_scaler.fit_transform(X)

numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])


# check # Standardize ~ mean 0 sd 1 using scale and center

# Normalize numerical data (e.g. to a range of 0-1) using range 

# Explore more advanced feature engineering such as Binarizing