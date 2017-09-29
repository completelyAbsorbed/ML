# istanbul_diff.py
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# I took the Istanbul Stock market data from UCI : https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE
# 
# I added a column, representing the difference between ISE(in tl) and ISE(in usd), this will be our target
# 
# we will ignore date, ISE_tl, and ISE_usd, for now


# preparing the environment : https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/
# tutorial that inspired this script : https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# how to switch keras backend from tensorflow to theano
# https://stackoverflow.com/questions/42177658/how-to-switch-backend-with-keras-from-tensorflow-to-theano



# load dataset
dataframe = pandas.read_csv("istanbul_diff.csv")
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 3:10]   # date is represented in subsequent columns, so we skip column 0
Y = dataset[:, 10]

# define base model_selection
def baseline_model(): # 0.00003703 (0.00002138) MSE
	# create model
	model = Sequential()
	model.add(Dense(7, input_dim = 7, kernel_initializer = 'normal', activation = 'relu'))
	model.add(Dense(1, kernel_initializer = 'normal'))
	# Compile model
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')
	return model
	
seed = 99								# fix random seed for reproducibility
numpy.random.seed(seed)		# evaluate model with standardized dataset



# estimator = KerasRegressor(build_fn = baseline_model, nb_epoch=100, batch_size=5, verbose=0)

# kfold = KFold(n_splits = 10, random_state = seed)		# evaluate the baseline model with 10-fold cross validation
# results = cross_val_score(estimator, X, Y, cv = kfold)
# print("Results: %.8f (%.8f) MSE") % (results.mean(), results.std()) # 0.00003703 (0.00002138) MSE

# numpy.random.seed(seed)	# create Pipeline to perform standardization avoiding data leak and  evaluate model with standardized dataset 
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized: %.8f (%.8f) MSE" % (results.mean(), results.std()))

####################################################################################

# convenient example

# def long_1_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(13, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# save space and functionalize the model run	
	
def run_model(model_run, model_name):
	numpy.random.seed(seed)
	estimators = []
	estimators.append(('standardize', StandardScaler()))
	estimators.append(('mlp', KerasRegressor(build_fn=model_run, epochs=100, batch_size=5, verbose=0)))
	pipeline = Pipeline(estimators)
	kfold = KFold(n_splits=10, random_state=seed)
	results = cross_val_score(pipeline, X, Y, cv=kfold)
	print(model_name + ": %.8f (%.8f) MSE" % (results.mean(), results.std()))
		
####################################################################################

def initial_1_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = initial_1_model, model_name = 'initial_1')	

def initial_2_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = initial_2_model, model_name = 'initial_2')	

def initial_3_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = initial_3_model, model_name = 'initial_3')	

def initial_4_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(14, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = initial_4_model, model_name = 'initial_4')	

def initial_5_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(10, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = initial_5_model, model_name = 'initial_5')	

####################################################################################
# initial run notes
# 

# 
####################################################################################

	