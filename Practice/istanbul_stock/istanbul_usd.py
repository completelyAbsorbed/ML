# istanbul_usd.py
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



# flag declaration
model_record_exists = True


# load dataset
dataframe = pandas.read_csv("istanbul_diff.csv")
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 3:10]   # date is represented in subsequent columns, so we skip column 0
Y = dataset[:, 2]		# ISE_usd

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
	
def run_model(model_run, model_name, model_record_exists = model_record_exists):
	numpy.random.seed(seed)
	estimators = []
	estimators.append(('standardize', StandardScaler()))
	estimators.append(('mlp', KerasRegressor(build_fn=model_run, epochs=100, batch_size=5, verbose=0)))
	pipeline = Pipeline(estimators)
	kfold = KFold(n_splits=10, random_state=seed)
	results = cross_val_score(pipeline, X, Y, cv=kfold)
	print(model_name + ": %.8f (%.8f) MSE" % (results.mean(), results.std()))
	new_row = [model_name, results.mean(), results.std()]
	# if model_record_exists:
		# model_dataframe = pandas.read_csv("model_record_usd.csv")
		# model_dataframe.loc[model_dataframe.shape[0]] = new_row
	# else:
		# model_dataframe = pandas.DataFrame(columns = ['model_name','MSE_mean', 'MSE_std'])
		# model_dataframe.loc[0] = new_row
	###
	model_dataframe = pandas.DataFrame(columns = ['model_name','MSE_mean', 'MSE_std'])
	model_dataframe.loc[0] = new_row
	###	
	#model_dataframe = model_dataframe.drop_duplicates
	with open('model_record_usd.csv', 'a') as f:
		model_dataframe.to_csv(f, header=False)
	#model_dataframe.to_csv('model_record_usd.csv')
		
####################################################################################

# def initial_1_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = initial_1_model, model_name = 'initial_1')	

# model_record_exists = True

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

# take a note from new_shapes_1 and use a first hidden layer width of 49

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

# def exploder_1_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(343, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = exploder_1_model, model_name = 'exploder_1')	

# def exploder_2_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(343, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(49, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = exploder_2_model, model_name = 'exploder_2')	

# def exploder_3_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(343, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(49, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = exploder_3_model, model_name = 'exploder_3')	

# def exploder_4_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(343, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = exploder_4_model, model_name = 'exploder_4')	

# def exploder_5_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(2401, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = exploder_5_model, model_name = 'exploder_5')	

# def exploder_6_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(2401, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(343, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = exploder_6_model, model_name = 'exploder_6')	

# def exploder_7_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(2401, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(343, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(49, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = exploder_7_model, model_name = 'exploder_7')	

# def exploder_8_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(2401, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(343, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(49, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = exploder_8_model, model_name = 'exploder_8')	

# def exploder_9_model(): 		#define model 
	# model = Sequential()	# create model
	# model.add(Dense(2401, input_dim=7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(49, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model
	
# run_model(model_run = exploder_9_model, model_name = 'exploder_9')	


####################################################################################
# big run notes
# 
# stack_1: 0.00024477 (0.00008999) MSE
# stack_2: 0.00028156 (0.00019726) MSE
# stack_3: 0.00033351 (0.00017288) MSE
# stack_4: 0.00039010 (0.00024451) MSE
# stack_5: 0.00045534 (0.00024827) MSE
# new_shapes_1: 0.00029080 (0.00014021) MSE
# new_shapes_2: 0.00030169 (0.00013971) MSE
# new_shapes_3: 0.00027357 (0.00011051) MSE
# new_shapes_4: 0.00030982 (0.00015250) MSE
# new_shapes_5: 0.00034121 (0.00022693) MSE
# pyramids_1: 0.00045274 (0.00025131) MSE
# pyramids_2: 0.00045534 (0.00024827) MSE
# pyramids_3: 0.00045220 (0.00026008) MSE
# pyramids_4: 0.00027707 (0.00007520) MSE
# pyramids_5: 0.00023204 (0.00009153) MSE
# pyramids_6: 0.00024914 (0.00012834) MSE
# derivative_1: 0.00035202 (0.00023781) MSE
# derivative_2: 0.00033806 (0.00026519) MSE
# derivative_3: 0.00043957 (0.00026585) MSE
# derivative_4: 0.00042674 (0.00027507) MSE
# derivative_5: 0.00045534 (0.00024827) MSE
# derivative_6: 0.00045534 (0.00024827) MSE
# derivative_7: 0.00029680 (0.00019465) MSE
# derivative_8: 0.00039153 (0.00021811) MSE
# derivative_9: 0.00040337 (0.00020686) MSE
# derivative_10: 0.00043858 (0.00026685) MSE
# secondary_1: 0.00033467 (0.00018097) MSE
# secondary_2: 0.00027415 (0.00020183) MSE
# secondary_3: 0.00042640 (0.00027931) MSE
# secondary_4: 0.00026538 (0.00019932) MSE
# secondary_5: 0.00034897 (0.00017544) MSE
# secondary_6: 0.00027678 (0.00020412) MSE
# secondary_7: 0.00027707 (0.00007520) MSE
# secondary_8: 0.00023032 (0.00009525) MSE
# secondary_9: 0.00036573 (0.00023368) MSE
# secondary_10: 0.00028732 (0.00019150) MSE
# secondary_11: 0.00031605 (0.00018704) MSE
# secondary_12: 0.00024183 (0.00008481) MSE
# secondary_13: 0.00043732 (0.00027362) MSE
# secondary_14: 0.00026504 (0.00012207) MSE
# secondary_15: 0.00033906 (0.00024367) MSE
# secondary_16: 0.00027868 (0.00010335) MSE
# further_1: 0.00045101 (0.00026775) MSE
# further_2: 0.00035236 (0.00024413) MSE
# further_3: 0.00022833 (0.00008609) MSE
# further_4: 0.00035214 (0.00018029) MSE
# further_5: 0.00034866 (0.00024028) MSE
# further_6: 0.00026554 (0.00010543) MSE
# further_7: 0.00025276 (0.00010641) MSE
# further_8: 0.00030047 (0.00019548) MSE
# further_9: 0.00029267 (0.00008563) MSE
# further_10: 0.00035878 (0.00016201) MSE
# further_11: 0.00036257 (0.00023361) MSE
# further_12: 0.00031036 (0.00019293) MSE
# further_13: 0.00032462 (0.00019015) MSE
# further_14: 0.00034444 (0.00023965) MSE
# further_15: 0.00034866 (0.00024028) MSE
# further_16: 0.00040028 (0.00020777) MSE
# further_17: 0.00040404 (0.00020672) MSE
# further_18: 0.00037545 (0.00022414) MSE
# further_19: 0.00041072 (0.00022922) MSE
# further_20: 0.00034621 (0.00017121) MSE
# exploder_1: 0.00032463 (0.00019046) MSE
# exploder_2: 0.00031488 (0.00013200) MSE
# exploder_3: 0.00033681 (0.00013801) MSE
# exploder_4: 0.00029327 (0.00013054) MSE
# exploder_5: 0.00037635 (0.00031316) MSE
# exploder_6: 0.00032622 (0.00015356) MSE
# exploder_7: 0.00033360 (0.00014374) MSE
# exploder_8: 0.00030544 (0.00008741) MSE
# exploder_9: 0.00027647 (0.00011004) MSE
# 
#
#
#
# 
####################################################################################
