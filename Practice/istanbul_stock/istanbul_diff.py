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

# def pyramids_1_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = pyramids_1_model, model_name = 'pyramids_1')	

# def pyramids_2_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = pyramids_2_model, model_name = 'pyramids_2')	

# def pyramids_3_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = pyramids_3_model, model_name = 'pyramids_3')	

# def pyramids_4_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = pyramids_4_model, model_name = 'pyramids_4')	

# def pyramids_5_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = pyramids_5_model, model_name = 'pyramids_5')	

# def pyramids_6_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = pyramids_6_model, model_name = 'pyramids_6')	

# def derivative_1_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = derivative_1_model, model_name = 'derivative_1')	

# def derivative_2_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = derivative_2_model, model_name = 'derivative_2')	

# def derivative_3_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = derivative_3_model, model_name = 'derivative_3')	

# def derivative_4_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = derivative_4_model, model_name = 'derivative_4')	

# def derivative_5_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = derivative_5_model, model_name = 'derivative_5')	

# def derivative_6_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = derivative_6_model, model_name = 'derivative_6')	

# def derivative_7_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

	
# run_model(model_run = derivative_7_model, model_name = 'derivative_7')	

# def derivative_8_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = derivative_8_model, model_name = 'derivative_8')	

# def derivative_9_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = derivative_9_model, model_name = 'derivative_9')	

# def derivative_10_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = derivative_10_model, model_name = 'derivative_10')	

####################################################################################
# pyramids, derivative run notes
# 
# pyramids_1: 0.00006523 (0.00003958) MSE
# pyramids_2: 0.00006505 (0.00004033) MSE
# pyramids_3: 0.00006426 (0.00004261) MSE
# pyramids_4: 0.00005989 (0.00004357) MSE
# pyramids_5: 0.00004049 (0.00001963) MSE
# pyramids_6: 0.00004524 (0.00003244) MSE
# derivative_1: 0.00006170 (0.00004181) MSE
# derivative_2: 0.00005452 (0.00004161) MSE
# derivative_3: 0.00006445 (0.00004245) MSE
# derivative_4: 0.00006350 (0.00004324) MSE
# derivative_5: 0.00006505 (0.00004033) MSE
# derivative_6: 0.00006505 (0.00004033) MSE
# derivative_7: 0.00005658 (0.00004103) MSE
# derivative_8: 0.00006344 (0.00004124) MSE
# derivative_9: 0.00006445 (0.00004245) MSE
# derivative_10: 0.00006445 (0.00004245) MSE
#
# the only notable results here are from :
#		- pyramids_5
#		- and minorly pyramids_6 
#
# make secondary models based on the two pyramid models,
#	and initial_1 and initial_2
####################################################################################

# def secondary_1_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_1_model, model_name = 'secondary_1')	

# def secondary_2_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_2_model, model_name = 'secondary_2')	

# def secondary_3_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_3_model, model_name = 'secondary_3')	

# def secondary_4_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_4_model, model_name = 'secondary_4')	

# def secondary_5_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_5_model, model_name = 'secondary_5')	

# def secondary_6_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_6_model, model_name = 'secondary_6')	

# def secondary_7_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_7_model, model_name = 'secondary_7')	

# def secondary_8_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_8_model, model_name = 'secondary_8')	

## take a note from new_shapes_1 and use a first hidden layer width of 49

# def secondary_9_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_9_model, model_name = 'secondary_9')	

# def secondary_10_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_10_model, model_name = 'secondary_10')	

# def secondary_11_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_11_model, model_name = 'secondary_11')	

# def secondary_12_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_12_model, model_name = 'secondary_12')	

# def secondary_13_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_13_model, model_name = 'secondary_13')	

# def secondary_14_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_14_model, model_name = 'secondary_14')	

# def secondary_15_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_15_model, model_name = 'secondary_15')	

# def secondary_16_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = secondary_16_model, model_name = 'secondary_16')	

####################################################################################
# seconday run notes
# 
# secondary_1: 0.00005823 (0.00004056) MSE
# secondary_2: 0.00004226 (0.00003117) MSE	*
# secondary_3: 0.00006118 (0.00004256) MSE
# secondary_4: 0.00004575 (0.00003185) MSE	*
# secondary_5: 0.00005391 (0.00004240) MSE
# secondary_6: 0.00005237 (0.00004216) MSE
# secondary_7: 0.00005989 (0.00004357) MSE
# secondary_8: 0.00004364 (0.00003177) MSE	*
# secondary_9: 0.00005823 (0.00003446) MSE
# secondary_10: 0.00005051 (0.00003361) MSE
# secondary_11: 0.00005033 (0.00003429) MSE
# secondary_12: 0.00004795 (0.00002763) MSE	*
# secondary_13: 0.00006370 (0.00004118) MSE
# secondary_14: 0.00005301 (0.00003580) MSE
# secondary_15: 0.00005309 (0.00002346) MSE
# secondary_16: 0.00004957 (0.00002246) MSE	*
#
# 
# somewhat disappointing results, hoping for a breakthrough to beat 0.00003946
####################################################################################

# def further_1_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_1_model, model_name = 'further_1')	

# def further_2_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_2_model, model_name = 'further_2')	

# def further_3_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_3_model, model_name = 'further_3')	

# def further_4_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_4_model, model_name = 'further_4')	

# def further_5_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_5_model, model_name = 'further_5')	

#

# def further_6_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(42, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_6_model, model_name = 'further_6')	

# def further_7_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(42, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_7_model, model_name = 'further_7')	

# def further_8_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(42, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_8_model, model_name = 'further_8')	

# def further_9_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(42, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_9_model, model_name = 'further_9')	

# def further_10_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(42, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_10_model, model_name = 'further_10')	

#

# def further_11_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_11_model, model_name = 'further_11')	

# def further_12_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_12_model, model_name = 'further_12')	

# def further_13_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_13_model, model_name = 'further_13')	

# def further_14_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_14_model, model_name = 'further_14')	

# def further_15_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_15_model, model_name = 'further_15')	

#

# def further_16_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_16_model, model_name = 'further_16')	

# def further_17_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_17_model, model_name = 'further_17')	

# def further_18_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_18_model, model_name = 'further_18')	

# def further_19_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_19_model, model_name = 'further_19')	

# def further_20_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(49, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = further_20_model, model_name = 'further_20')	

####################################################################################
# further run notes
# 
# further_1: 0.00005827 (0.00004054) MSE
# further_2: 0.00005918 (0.00004335) MSE
# further_3: 0.00004620 (0.00003626) MSE
# further_4: 0.00004990 (0.00003224) MSE
# further_5: 0.00005037 (0.00003172) MSE
# further_6: 0.00004856 (0.00003030) MSE
# further_7: 0.00004461 (0.00002520) MSE
# further_8: 0.00005212 (0.00003230) MSE
# further_9: 0.00005318 (0.00003150) MSE
# further_10: 0.00005495 (0.00002763) MSE
# further_11: 0.00005311 (0.00003360) MSE
# further_12: 0.00006273 (0.00004262) MSE
# further_13: 0.00005212 (0.00003509) MSE
# further_14: 0.00005006 (0.00003205) MSE
# further_15: 0.00005037 (0.00003172) MSE
# further_16: 0.00006445 (0.00004245) MSE
# further_17: 0.00006445 (0.00004245) MSE
# further_18: 0.00006445 (0.00004245) MSE
# further_19: 0.00005829 (0.00003338) MSE
# further_20: 0.00006305 (0.00004293) MSE
# 
# results are still disappointing, probably because our target is too hard to predict (depends on much much more than our data)
# 
# try a last ditch effort, 'exploder' models : utilizing layers sized as powers of the number of inputs(did this a little already)
####################################################################################

def exploder_1_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(343, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = exploder_1_model, model_name = 'exploder_1')	

def exploder_2_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(343, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(49, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = exploder_2_model, model_name = 'exploder_2')	

def exploder_3_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(343, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(49, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = exploder_3_model, model_name = 'exploder_3')	

def exploder_4_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(343, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = exploder_4_model, model_name = 'exploder_4')	

def exploder_5_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(2401, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = exploder_5_model, model_name = 'exploder_5')	

def exploder_6_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(2401, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(343, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = exploder_6_model, model_name = 'exploder_6')	

def exploder_7_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(2401, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(343, kernel_initializer='normal', activation='relu'))
	model.add(Dense(49, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = exploder_7_model, model_name = 'exploder_7')	

def exploder_8_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(2401, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(343, kernel_initializer='normal', activation='relu'))
	model.add(Dense(49, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = exploder_8_model, model_name = 'exploder_8')	

def exploder_9_model(): 		#define model 
	model = Sequential()	# create model
	model.add(Dense(2401, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(49, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	return model
	
run_model(model_run = exploder_9_model, model_name = 'exploder_9')	


####################################################################################
# exploder run notes
# 

# 
# 
####################################################################################
