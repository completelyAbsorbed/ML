dictClass = {}


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
url = "https://goo.gl/sXleFv"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]


# KNN Regression
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
dictClass["KNN"] = results.mean()

# SVM
from sklearn import preprocessing
from sklearn import utils
from sklearn import svm
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(Y)
clf = svm.SVC()
clf.fit(X,encoded.astype('int'))
scoring = 'neg_mean_squared_error'
results = cross_val_score(clf, X, encoded.astype('int'), cv=kfold, scoring=scoring)
dictClass["SVC"] = results.mean()

# CART ... sklearn uses CART algo for Decision Trees
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, encoded.astype('int'))
scoring = 'neg_mean_squared_error'
results = cross_val_score(clf, X, encoded.astype('int'), cv=kfold, scoring=scoring)
dictClass["CART"] = results.mean()
print(dictClass)



dictLin = {}

# linear regression
from sklearn import linear_model
linReg = linear_model.LinearRegression()
linReg.fit(X,Y)
results = cross_val_score(linReg, X, Y, cv=kfold, scoring=scoring)
# results = cross_val_score(linReg, X, Y, scoring=scoring)
dictLin["Linear Regression"] = results.mean()

# logistic regression
logReg = linear_model.LogisticRegression()
logReg.fit(X,encoded.astype('int'))
results = cross_val_score(logReg, X, encoded.astype('int'), cv=kfold, scoring=scoring)
# results = cross_val_score(logReg, X, encoded.astype('int'), scoring=scoring)
dictLin["Logistic Regression"] = results.mean()

# linear discriminate analysis
from sklearn.lda import LDA
ldaModel = LDA(solver='lsqr', shrinkage=None).fit(X,encoded.astype('int'))
results = cross_val_score(ldaModel, X, encoded.astype('int'), cv=kfold, scoring=scoring)
# results = cross_val_score(logReg, X, encoded.astype('int'), scoring=scoring)
dictLin["Linear Discriminant Analysis"] = results.mean()
print(dictLin)
