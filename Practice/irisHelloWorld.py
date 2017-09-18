# irisHelloWorld.py
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# at some point, rework this project with better validation approach
# outline etc
# https://machinelearningmastery.com/python-machine-learning-mini-course/
# train/test/kfold
# https://medium.com/towards-data-science/train-test-split-and-cross-validation-in-python-80b61beca4b6
# 4-part validation series
# https://rapidminer.com/validate-models-training-test-error/
# https://rapidminer.com/validate-models-ignore-training-errors/
# https://rapidminer.com/validate-models-cross-validation/
# https://rapidminer.com/learn-right-way-validate-models-part-4-accidental-contamination/

# filePath = "C:\MACHINE LEARNING\iris.csv"
filePath = "D:\ML\Practice\iris.csv"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# read in data
dataframe = read_csv(filePath, names=names)
array = dataframe.values

# split data and target(class)
X = array[:,0:4]
Y = array[:,4]

### 1. Understanding your data using descriptive statistics and visualization.

###### get a description
description = dataframe.describe()
print(description)

#        sepal_length  sepal_width  petal_length  petal_width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.054000      3.758667     1.198667
# std        0.828066     0.433594      1.764420     0.763161
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000

###### look at histogram
dataframe.hist()
# plt.show()
###### look at box-whisker plot
dataframe.plot(kind='box')
# plt.show()
###### look at pairwise scatter_matrix
scatter_matrix(dataframe)
# plt.show() # moving to end of file 

######################## notes to self on 1. ######################## 
#
###### hist()
#
# petal_length has two distinct regions : {1,~2.15} and {~2.8,~6.85}
#	# maybe this can be tested, does the region determine the category?
# petal_length lower region appears decreasing
# petal_length greater region has bell-like shape
#
# petal_width has a very low-frequency between ~{0.55,0.8}
# petal_width does not exhibit bell-like shape, shows considerable variation
#
# sepal_length shows somewhat bell-like shape, as does sepal_width
#
###### box-whisker
# 
# sepal_length mean is slightly skewed up
#
# petal_length, petal_width means are somewhat skewed up
# 
# sepal_width mean is slightly skewed down
# 
# sepal_width is the only feature exhibiting outliers, 1 lower, 3 upper
# 
###### scatter_matrix
#
# the following pairs show significant correlation : (consider for feature engineering)
#		- petal_width, petal_length
#		- sepal_length, petal_length
# the following pair shows perhaps some correlation :
#		- petal_width, sepal_length
# consider using combinations of the above when feature engineering
#
# look at scatter matrix for more variable splitting options
#	# use KNN?

### S. Split the data into a number of pieces. Start loop.
# functionalize remaining steps so loop is readable and isn't unwieldy



######################## notes to self on S. ######################## 
#
# iris data set is 150 rows (5 columns)
#
# splits to try : 50, 50, 50
#				  30, 30, 30, 30, 30
#				  10(x15)
#				  35, 35, 35, 35, 10
#				  40, 40, 40, 30


### 2. Preprocessing the data to best expose the structure of the problem.
## Feature Importance
# look at correlation to class using spearmanr(because categorical target, try pearson for numbers only)
for name in names[0:4]:
	print name
	print spearmanr(dataframe[name], dataframe['class']) # correlation to class
# sepal_length
# SpearmanrResult(correlation=0.79807811724205491, pvalue=2.2480123863519486e-34)
# sepal_width
# SpearmanrResult(correlation=-0.43434773571359908, pvalue=2.803225600774315e-08)
# petal_length
# SpearmanrResult(correlation=0.93544135003637963, pvalue=1.0069230299071028e-68)
# petal_width
# SpearmanrResult(correlation=0.93785004223816115, pvalue=6.6048198741735682e-70)	
## Feature Extraction
# our dataset is small, so we will skip feature extraction for now, though intend to revisit
## Feature Selection
# it's important to remember that feature selection should be performed before training
# for each fold when doing cross-validation. so setting up a method may be important!
# and it is bad practice to do the feature selection on the whole data set
#
# code goes here if needed
## Feature Construction
# want to try constructing some new features, evaluating their spearmanr, and then
# trying to train models on constructed-only data, as well as mixed-data 
sw = dataframe['sepal_width']
sl = dataframe['sepal_length']
pw = dataframe['petal_width']
pl = dataframe['petal_length']
target = dataframe['class']

slPpl = sl + pl
print 'slPpl'
print spearmanr(slPpl, target)
swPpl = sw + pl
print 'swPpl'
print spearmanr(swPpl, target)
pwPpl = pw + pl
print 'pwPpl'
print spearmanr(pwPpl, target)
pwPsl = pw + sl
print 'pwPsl'
print spearmanr(pwPsl, target)
swDsl = sw / sl
print 'swDsl'
print spearmanr(swDsl, target)
slDsw = sl / sw
print 'slDsw'
print spearmanr(slDsw, target)
pwDsw = pw / sw
print 'pwDsw'
print spearmanr(pwDsw, target)
swDpw = sw / pw
print 'swDpw'
print spearmanr(swDpw, target)
# look at the correlation matrices of our 8 newly engineered features
#	make a double-loop to fill a 2D object with the correlations
features = [sl, sw, pl, pw, slPpl, swPpl, pwPpl, pwPsl, swDsl, slDsw, pwDsw, swDpw]
featureNames = ['sl', 'sw', 'pl', 'pw', 'slPpl', 'swPpl', 'pwPpl', 'pwPsl', 'swDsl', 'slDsw', 'pwDsw', 'swDpw']

print pandas.DataFrame(np.around(np.corrcoef(features), decimals=2), index = featureNames, columns=featureNames)
print ''
print ''
print ''
## Feature Learning
# not doing anything here right now, not needed, too advanced. look into at a future date
######################## notes to self on 2. ######################## 
# 
###### Feature Importance
# 
# petal_length, petal_width  > 0.90, definitely keep
# sepal_length              ~= 0.80, consider keep
# sepal_width               ~= 0.43, consider scrap
###### Feature Extraction
# 
# our dataset is small, so we will skip feature extraction for now, though intend to revisit
###### Feature Selection
# 
# using spearmanr(), keep features according to the following schemes :
#	(A) spearmanr() > 0.50
#	(B) spearmanr() > 0.60
#	(C) spearmanr() > 0.70
#	(D) spearmanr() > 0.80
#	(E) spearmanr() > 0.90
#	(F) spearmanr() < 0.50
# I expect A, B, C, D, E will perform reasonably well, and F will perform inadequately
###### Feature Construction
# 
# want to try : split petal_length into two variables based on contiguous regions in hist()
# 
# features :
#	+ (i)   {original features} : sl, sw, pl, pw
#	+ (ii)  {adding highly correlated columns}
#	* slPpl : sepal_length + petal_length
#	* swPpl : sepal_width + petal_length
#	* pwPpl : petal_width + petal_length
#	* pwPsl : petal_width + sepal_length
#	+ (iii) {dividing less correlated columns}
# 	* swDsl : sepal_width / sepal_length
#	* (swDsl)**-1 : (sepal_width / sepal_length)**-1
#	* pwDsw : petal_width / sepal_width
#	* (pwDsw)**-1 : (petal_width / sepal_width)**-1
#
# current plan is to run these features as described above and using the Feature Selection
# guidelines described. a next step may be investigating more nuanced feature selection on
# engineered features, and devising methods for selection based on relative correlation
# combined with correlation to target (i.e. eliminate inverses and other highly correlated features?)
# 
# so far, we have 18 models to consider, and many more corners to explore if we want :
#   i.A,   i.B,   i.C,   i.D,   i.E,   i.F
#  ii.A,  ii.B,  ii.C,  ii.D,  ii.E,  ii.F
# iii.A, iii.B, iii.C, iii.D, iii.E, iii.F 
#
# one example of a corner we could explore is (ii)&(iii) combined
###### Feature Learning
# not doing anything here right now, not needed, too advanced. look into at a future date

### 3. Spot-checking a number of algorithms using your own test harness.
# initialize kfold and scoring, other macro variables
kfold = KFold(n_splits=2, random_state=777)
scoring = 'neg_log_loss'

# logistic regression, modify to split on A-F, i-iii
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())

# 

# next I want to split apart the kfold and crossval process so I can do unbiased feature selection...


### 4. Improving results using algorithm parameter tuning.
### 5. Improving results using ensemble methods.
### 6. Finalize the model ready for future use.




########### hold below for future copy+paste ########### 

### 1. Understanding your data using descriptive statistics and visualization.
### S. Split the data into a number of pieces. Start loop.
### 2. Preprocessing the data to best expose the structure of the problem.
## Feature Importance
## Feature Extraction
## Feature Selection
## Feature Construction
## Feature Learning
### 3. Spot-checking a number of algorithms using your own test harness.
### 4. Improving results using algorithm parameter tuning.
### 5. Improving results using ensemble methods.
### 6. Finalize the model ready for future use.




plt.show()


