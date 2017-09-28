# bike_share_regression_nn.py
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# preparing the environment : https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/
# tutorial that inspired this script : https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# how to switch keras backend from tensorflow to theano
# https://stackoverflow.com/questions/42177658/how-to-switch-backend-with-keras-from-tensorflow-to-theano



# load dataset
dataframe = pandas.read_csv("hour_deinstance.csv")
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 1:15]   # date is represented in subsequent columns, so we skip column 0
Y = dataset[:, 15]

# define base model_selection
def baseline_model(): # 0.9149(2.7374)
	# create model
	model = Sequential()
	model.add(Dense(14, input_dim = 14, kernel_initializer = 'normal', activation = 'relu'))
	model.add(Dense(1, kernel_initializer = 'normal'))
	# Compile model
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')
	return model
	
seed = 99								# fix random seed for reproducibility
# numpy.random.seed(seed)		# evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn = baseline_model, nb_epoch=100, batch_size=5, verbose=0)

# kfold = KFold(n_splits = 10, random_state = seed)		# evaluate the baseline model with 10-fold cross validation
# results = cross_val_score(estimator, X, Y, cv = kfold)
# print("Results: %.4f (%.4f) MSE") % (results.mean(), results.std())

# numpy.random.seed(seed)	# create Pipeline to perform standardization avoiding data leak and  evaluate model with standardized dataset # 0.0240 (0.0258)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized: %.4f (%.4f) MSE" % (results.mean(), results.std()))

####################################################################################

# convenient example

# def long_1_model(): 		#define long_1 model # mean(std)
	# model = Sequential()	# create model
	# model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(13, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=long_1_model, epochs=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("long_1: %.4f (%.4f) MSE" % (results.mean(), results.std()))
	
####################################################################################

# def initial_1_model(): 		#define initial_1 model # 0.4368(0.7357)
	# model = Sequential()	# create model
	# model.add(Dense(14, input_dim=14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=initial_1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("initial_1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def initial_2_model(): 		#define initial_2 model # 0.0316(0.0310)
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=initial_2_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("initial_2: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def initial_3_model(): 		#define initial_3 model # 0.0871(0.1171)
	# model = Sequential()	# create model
	# model.add(Dense(14, input_dim=14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=initial_3_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("initial_3: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def initial_4_model(): 		#define initial_4 model # 0.0112(0.0077)			# great improvement!
	# model = Sequential()	# create model
	# model.add(Dense(20, input_dim=14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=initial_4_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("initial_4: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def initial_5_model(): 		#define initial_5 model # 0.0081(0.0068)		# very good!!!
	# model = Sequential()	# create model
	# model.add(Dense(18, input_dim=14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=initial_5_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("initial_5: %.4f (%.4f) MSE" % (results.mean(), results.std()))

####################################################################################
# there seems to be a vector of improvement along models initial_4, initial_5, which differ only by decreasing in 
#		hidden layer neuron number, while being greater than input number(14)
#	
# therefore, try 19, 17, 16, 15, and 21(one higher than initial_4)
####################################################################################

def hidden_1_model(): 		#define hidden_1 model # 0.0056(0.0034)
	model = Sequential()	# create model
	model.add(Dense(19, input_dim=14, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=hidden_1_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=4, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("hidden_1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

def hidden_2_model(): 		#define hidden_2 model # 0.0035(0.0020)
	model = Sequential()	# create model
	model.add(Dense(17, input_dim=14, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=hidden_2_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=4, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("hidden_2: %.4f (%.4f) MSE" % (results.mean(), results.std()))

def hidden_3_model(): 		#define hidden_3 model # 0.0171(0.0264)
	model = Sequential()	# create model
	model.add(Dense(16, input_dim=14, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=hidden_3_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=4, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("hidden_3: %.4f (%.4f) MSE" % (results.mean(), results.std()))

def hidden_4_model(): 		#define hidden_4 model # 0.0515(0.0789)
	model = Sequential()	# create model
	model.add(Dense(15, input_dim=14, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=hidden_4_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=4, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("hidden_4: %.4f (%.4f) MSE" % (results.mean(), results.std()))

def hidden_5_model(): 		#define hidden_5 model # 0.1029(0.1095)
	model = Sequential()	# create model
	model.add(Dense(21, input_dim=14, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=hidden_5_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=4, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("hidden_5: %.4f (%.4f) MSE" % (results.mean(), results.std()))
