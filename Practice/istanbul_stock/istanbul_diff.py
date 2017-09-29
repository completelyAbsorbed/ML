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
def baseline_model(): # 
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

# def initial_1_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = initial_1_model, model_name = 'initial_1')	

# def initial_2_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = initial_2_model, model_name = 'initial_2')	

# def initial_3_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = initial_3_model, model_name = 'initial_3')	

# def initial_4_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(14, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = initial_4_model, model_name = 'initial_4')	

# def initial_5_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(10, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = initial_5_model, model_name = 'initial_5')	

####################################################################################
# initial run notes
# 
# initial_1: 0.00004172 (0.00002517) MSE
# initial_2: 0.00003946 (0.00002124) MSE
# initial_3: 0.00003955 (0.00002184) MSE
# initial_4: 0.00004207 (0.00002453) MSE
# initial_5: 0.00004341 (0.00002821) MSE
#
# try more! stacking models this time 
####################################################################################

# def stack_1_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = stack_1_model, model_name = 'stack_1')	

# def stack_2_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = stack_2_model, model_name = 'stack_2')	

# def stack_3_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = stack_3_model, model_name = 'stack_3')	

# def stack_4_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = stack_4_model, model_name = 'stack_4')	

# def stack_5_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = stack_5_model, model_name = 'stack_5')	

####################################################################################
# stack run notes
# 
# stack_1: 0.00004078 (0.00002594) MSE
# stack_2: 0.00004076 (0.00002121) MSE
# stack_3: 0.00005295 (0.00004227) MSE
# stack_4: 0.00006445 (0.00004245) MSE
# stack_5: 0.00006505 (0.00004033) MSE
#
# keep trying new designs!
####################################################################################

# def new_shapes_1_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = new_shapes_1_model, model_name = 'new_shapes_1')	

# def new_shapes_2_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = new_shapes_2_model, model_name = 'new_shapes_2')	

# def new_shapes_3_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(14, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = new_shapes_3_model, model_name = 'new_shapes_3')	

# def new_shapes_4_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(21, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = new_shapes_4_model, model_name = 'new_shapes_4')	

# def new_shapes_5_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(21, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = new_shapes_5_model, model_name = 'new_shapes_5')	

####################################################################################
# new_shapes run notes
# 
# new_shapes_1: 0.00004986 (0.00002715) MSE
# new_shapes_2: 0.00005190 (0.00003402) MSE
# new_shapes_3: 0.00004598 (0.00002443) MSE
# new_shapes_4: 0.00004712 (0.00002699) MSE
# new_shapes_5: 0.00004865 (0.00003180) MSE
#
# we've done a few runs, our three best results so far are :
#		- initial_2
#		- initial_3
#		- stack_2
# 
#  next, make new designs, taking these into consideration
####################################################################################

def pyramids_1_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = pyramids_1_model, model_name = 'pyramids_1')	

def pyramids_2_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = pyramids_2_model, model_name = 'pyramids_2')	

def pyramids_3_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = pyramids_3_model, model_name = 'pyramids_3')	

def pyramids_4_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = pyramids_4_model, model_name = 'pyramids_4')	

def pyramids_5_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = pyramids_5_model, model_name = 'pyramids_5')	

def pyramids_6_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = pyramids_6_model, model_name = 'pyramids_6')	

def derivative_1_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = derivative_1_model, model_name = 'derivative_1')	

def derivative_2_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = derivative_2_model, model_name = 'derivative_2')	

def derivative_3_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = derivative_3_model, model_name = 'derivative_3')	

def derivative_4_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = derivative_4_model, model_name = 'derivative_4')	

def derivative_5_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = derivative_5_model, model_name = 'derivative_5')	

def derivative_6_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = derivative_6_model, model_name = 'derivative_6')	

def derivative_7_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model

	
run_model(model_run = derivative_7_model, model_name = 'derivative_7')	

def derivative_8_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = derivative_8_model, model_name = 'derivative_8')	

def derivative_9_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = derivative_9_model, model_name = 'derivative_9')	

def derivative_10_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = derivative_10_model, model_name = 'derivative_10')	

####################################################################################
# pyramids, derivative run notes
# 

#
#  
####################################################################################
