# iris2.py
import numpy as np
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr
from scipy.stats import describe
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris 
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
import random

def makespace(lines=10):
	for counter in range(0,lines):
		print '...'

# set the seed
random.seed(303)

# load data, initialize feature/target names
iris = load_iris()
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# do an 80/20 (120/30) split Train / Test 

rowLength = 150
split_point = int(0.80 * rowLength)
rowListShuffled = range(0, rowLength)
random.shuffle(rowListShuffled)

# segment the testing and training portions
X_train = pandas.DataFrame(data = iris.data[rowListShuffled[:split_point],:], columns = names[:4])
y_train = pandas.DataFrame(data = iris.target[rowListShuffled[:split_point]])
X_test = pandas.DataFrame(data = iris.data[rowListShuffled[split_point:],:], columns = names[:4])
y_test = pandas.DataFrame(data = iris.target[rowListShuffled[split_point:]])

### 1. Understanding your data using descriptive statistics and visualization.

###### get a description
makespace(5)
description = X_train.describe()
print(description)

#        sepal_length  sepal_width  petal_length  petal_width
# count    120.000000   120.000000    120.000000   120.000000
# mean       5.823333     3.049167      3.750000     1.190000
# std        0.846519     0.444196      1.784904     0.763605
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.700000     3.000000      4.200000     1.300000
# 75%        6.400000     3.325000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000

###### look at histogram
X_train.hist()
###### look at box-whisker plot
X_train.plot(kind='box')
###### look at pairwise scatter_matrix
scatter_matrix(X_train)

######################## notes to self on 1. ######################## 
#
###### hist()
#
# petal_length has two distinct regions : {1,~2.15} and {~2.8,~6.85}
#	# maybe this can be tested, does the region determine the category?
# petal_length lower region appears decreasing in frequency
# petal_length greater region has bell-like shape
#
# petal_width has a very low-frequency between ~{0.55,0.75}
# petal_width does not exhibit bell-like shape, 
# frequencies shows considerable variation
#
# sepal_length shows (slightly)somewhat bell-like shape, as does sepal_width
#
###### box-whisker
# 
# sepal_length mean is close to center
#
# petal_length, petal_width means are somewhat skewed up
# 
# sepal_width mean is slightly skewed down
# 
# sepal_width is the only feature exhibiting outliers, 1 lower, 2 upper
# 
# each attribute shows a shorter tail than head
# 
###### scatter_matrix
#
# the following pairs show significant correlation : (consider for feature engineering)
#		- petal_width, petal_length
#		- sepal_length, petal_length
# the following pair shows perhaps some correlation :
#		- petal_width, sepal_length
# the following pairs show little to no correlation :
#		- petal_width, sepal_width
#		- petal_length, sepal_width
#		- sepal length, sepal width
# consider using combinations of the above when feature engineering
#
# look at scatter matrix for more variable splitting options
#	# use KNN?
# 
# scatter_matrix shows some attribute pairs have distinct regions, could be useful

### 2. Preprocessing the data to best expose the structure of the problem.
# https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/
## Feature Importance
# look at correlation to class using spearmanr(because categorical target. try pearson for regression/numbers)

makespace(5)

for attribute_index in range(0,4):
	attribute_name = names[attribute_index]
	spearman_result = spearmanr(X_train[names[attribute_index]], y_train)
	print "Attribute '%s', correlation : %f, p-value : %f" %(attribute_name, spearman_result.correlation, spearman_result.pvalue)
# Attribute 'sepal_length', correlation : 0.819411, p-value : 0.000000
# Attribute 'sepal_width', correlation : -0.423327, p-value : 0.000001
# Attribute 'petal_length', correlation : 0.935554, p-value : 0.000000
# Attribute 'petal_width', correlation : 0.935717, p-value : 0.000000
## Feature Extraction
# our dataset is small, so we will skip feature extraction for now, though intend to revisit
# consider using this link https://machinelearningmastery.com/feature-selection-machine-learning-python/
## Feature Selection
# it's important to remember that feature selection should be performed before training
# for each fold when doing cross-validation. so setting up a method may be important!
# and it is bad practice to do the feature selection on the whole data set
#
# code goes here if needed
## Feature Construction
# want to try constructing some new features, evaluating their spearmanr, and then
# trying to train models on constructed-only data, as well as mixed-data 

sepal_width = X_train['sepal_width']
sepal_length = X_train['sepal_length']
petal_width = X_train['petal_width']
petal_length = X_train['petal_length']
target = y_train

new_feature = sepal_length_plus_petal_length = sepal_length + petal_length
attribute_name = 'sepal_length_plus_petal_length'
spearman_result = spearmanr(new_feature, y_train)
print "Attribute '%s', correlation : %f, p-value : %f" %(attribute_name, spearman_result.correlation, spearman_result.pvalue)

new_feature = sepal_width_plus_petal_length = sepal_width + petal_length
attribute_name = 'sepal_width_plus_petal_length'
spearman_result = spearmanr(new_feature, y_train)
print "Attribute '%s', correlation : %f, p-value : %f" %(attribute_name, spearman_result.correlation, spearman_result.pvalue)

new_feature = petal_width_plus_petal_length = petal_width + petal_length
attribute_name = 'petal_width_plus_petal_length'
spearman_result = spearmanr(new_feature, y_train)
print "Attribute '%s', correlation : %f, p-value : %f" %(attribute_name, spearman_result.correlation, spearman_result.pvalue)

new_feature = petal_width_plus_sepal_length = petal_width + sepal_length
attribute_name = 'petal_width_plus_sepal_length'
spearman_result = spearmanr(new_feature, y_train)
print "Attribute '%s', correlation : %f, p-value : %f" %(attribute_name, spearman_result.correlation, spearman_result.pvalue)

new_feature = sepal_width_divby_sepal_length = sepal_width / sepal_length
attribute_name = 'sepal_width_divby_sepal_length'
spearman_result = spearmanr(new_feature, y_train)
print "Attribute '%s', correlation : %f, p-value : %f" %(attribute_name, spearman_result.correlation, spearman_result.pvalue)

new_feature = sepal_length_divby_sepal_width = sepal_length / sepal_width
attribute_name = 'sepal_length_divby_sepal_width'
spearman_result = spearmanr(new_feature, y_train)
print "Attribute '%s', correlation : %f, p-value : %f" %(attribute_name, spearman_result.correlation, spearman_result.pvalue)

new_feature = petal_width_divby_sepal_width = petal_width / sepal_width
attribute_name = 'petal_width_divby_sepal_width'
spearman_result = spearmanr(new_feature, y_train)
print "Attribute '%s', correlation : %f, p-value : %f" %(attribute_name, spearman_result.correlation, spearman_result.pvalue)

new_feature = sepal_width_divby_petal_width = sepal_width / petal_width
attribute_name = 'sepal_width_divby_petal_width'
spearman_result = spearmanr(new_feature, y_train)
print "Attribute '%s', correlation : %f, p-value : %f" %(attribute_name, spearman_result.correlation, spearman_result.pvalue)

features = [sepal_length, sepal_width, petal_length, petal_width,
			sepal_length_plus_petal_length, sepal_width_plus_petal_length,
			petal_width_plus_petal_length, petal_width_plus_sepal_length,
			sepal_width_divby_sepal_length, sepal_length_divby_sepal_width,
			petal_width_divby_sepal_width, sepal_width_divby_petal_width]
			
featureNames = ['sl', 'sw', 'pl', 'pw', 'slPpl', 'swPpl', 'pwPpl', 'pwPsl', 'swDsl', 'slDsw', 'pwDsw', 'swDpw']

makespace(5)
print pandas.DataFrame(np.around(np.corrcoef(features), decimals=2), index = featureNames, columns=featureNames)

## Feature Selection 
# Recursive Feature Elimination (RFE). lots of code borrowed from MachineLearningMastery
# https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/
# choose 5 features now, but use grid-search or other optimization on this in later step

extended_feature_names = ['sl', 'sw', 'pl', 'pw', 'slPpl', 'swPpl', 'pwPpl', 'pwPsl', 'swDsl', 'slDsw', 'pwDsw', 'swDpw']

extended_features = [sepal_length, sepal_width, petal_length, petal_width,
					sepal_length_plus_petal_length, sepal_width_plus_petal_length,
					petal_width_plus_petal_length, petal_width_plus_sepal_length,
					sepal_width_divby_sepal_length, sepal_length_divby_sepal_width,
					petal_width_divby_sepal_width, sepal_width_divby_petal_width]
extended_features_df = pandas.concat(extended_features, axis=1, keys=extended_feature_names)

# Feature Selection plays into step 3, maybe merge to there
model = LogisticRegression()
# create RFE model, select 5 attributes
rfe = RFE(model,5)
rfe = rfe.fit(extended_features_df, target.values.ravel())
# create ExtraTreesClassifier model
etc = ExtraTreesClassifier()
etc.fit(extended_features_df, target.values.ravel())
# display the relative importance of each attribute
makespace(5)
print 'ExtraTreesClassifier feature importance : '
print etc.feature_importances_ # this changes every time I run it, why?
## Feature Learning
# not doing anything here right now, not needed, too advanced. look into at a future date

### 3. Spot-checking a number of algorithms using your own test harness.
makespace(5)
# initialize kfold and scoring, other macro variables
kfold = KFold(n_splits=5, random_state=777)
# can I do cross validation with RFE and ETC? ... figure out how ...

scoring = 'neg_log_loss'

model = LogisticRegression()
rfe = RFE(model,5)
results = cross_val_score(rfe, extended_features_df, target.values.ravel(), cv=kfold, scoring=scoring)
print("LogisticRegression with rfe @5 neg_log_loss; mean, std: %.3f (%.3f)") % (results.mean(), results.std())

etc = ExtraTreesClassifier(random_state=777)
results = cross_val_score(etc, extended_features_df, target.values.ravel(), cv=kfold, scoring=scoring)
print("ExtraTreesClassifier neg_log_loss; mean, std: %.3f (%.3f)") % (results.mean(), results.std())

### 4. Improving results using algorithm parameter tuning.
# time for grid search!










# leave plot show call at end, it's less annoying to me this way xD
# also leave it off when I don't need it.
# plt.show()