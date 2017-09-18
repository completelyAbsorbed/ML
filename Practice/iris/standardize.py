# Load CSV using Pandas from URL

import pandas

from sklearn.preprocessing import StandardScaler
import numpy



url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)

array = data.values

# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])


# check # Standardize ~ mean 0 sd 1 using scale and center

# Normalize numerical data (e.g. to a range of 0-1) using range 

# Explore more advanced feature engineering such as Binarizing