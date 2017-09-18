# Load CSV using Pandas from URL

import pandas
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
array = data.values

# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]

###################################################################################

# kfold = KFold(n_splits=8, random_state=7)
model = LogisticRegression()
loo = LeaveOneOut()
looCV = loo.get_n_splits(X)

results = cross_val_score(model, X, Y, cv=loo)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)