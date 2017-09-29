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
X = dataset[:, 1:13]   # date is represented in subsequent columns, so we skip column 0
Y = dataset[:, 15]

# define base model_selection
def baseline_model(): # 18974.1539(10001.5383)
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim = 12, kernel_initializer = 'normal', activation = 'relu'))
	model.add(Dense(1, kernel_initializer = 'normal'))
	# Compile model
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')
	return model
	
seed = 99								# fix random seed for reproducibility
numpy.random.seed(seed)		# evaluate model with standardized dataset



# estimator = KerasRegressor(build_fn = baseline_model, nb_epoch=100, batch_size=5, verbose=0)

# kfold = KFold(n_splits = 10, random_state = seed)		# evaluate the baseline model with 10-fold cross validation
# results = cross_val_score(estimator, X, Y, cv = kfold)
# print("Results: %.4f (%.4f) MSE") % (results.mean(), results.std()) # 20164.0887 (11699.7947)

# numpy.random.seed(seed)	# create Pipeline to perform standardization avoiding data leak and  evaluate model with standardized dataset 
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

# def initial_1_model(): 		#define initial_1 model # 16997(9762)
	# model = Sequential()	# create model
	# model.add(Dense(12, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(12, kernel_initializer='normal', activation='relu'))
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

# def initial_2_model(): 		#define initial_2 model # 16993(6615)
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
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

# def initial_3_model(): 		#define initial_3 model # 19470(7934)
	# model = Sequential()	# create model
	# model.add(Dense(12, input_dim=12, kernel_initializer='normal', activation='relu'))
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

# def initial_4_model(): 		#define initial_4 model # 19302(7421)			# great improvement!
	# model = Sequential()	# create model
	# model.add(Dense(20, input_dim=12, kernel_initializer='normal', activation='relu'))
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

# def initial_5_model(): 		#define initial_5 model # 18611(8203)		# very good!!!
	# model = Sequential()	# create model
	# model.add(Dense(18, input_dim=12, kernel_initializer='normal', activation='relu'))
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
# 
#	
#	
# 
####################################################################################

# def hidden_1_model(): 		#define hidden_1 model # 16940(5708)
	# model = Sequential()	# create model
	# model.add(Dense(19, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=hidden_1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("hidden_1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def hidden_2_model(): 		#define hidden_2 model # 19248(8507)
	# model = Sequential()	# create model
	# model.add(Dense(17, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=hidden_2_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("hidden_2: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def hidden_3_model(): 		#define hidden_3 model # 20900(9185)
	# model = Sequential()	# create model
	# model.add(Dense(16, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=hidden_3_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("hidden_3: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def hidden_4_model(): 		#define hidden_4 model # 20877(9200)
	# model = Sequential()	# create model
	# model.add(Dense(15, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=hidden_4_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("hidden_4: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def hidden_5_model(): 		#define hidden_5 model # 20757(9514)
	# model = Sequential()	# create model
	# model.add(Dense(21, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=hidden_5_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("hidden_5: %.4f (%.4f) MSE" % (results.mean(), results.std()))

####################################################################################
# 
#	hidden_1, initial_1, initial_2 had best results. let's explore extensions from these models...
# 
####################################################################################

# def secondary_1_model(): 		#define secondary_1 model # 18783(8249)
	# model = Sequential()	# create model
	# model.add(Dense(45, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=secondary_1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("secondary_1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def secondary_2_model(): 		#define secondary_2 model # 19427(8184)
	# model = Sequential()	# create model
	# model.add(Dense(36, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=secondary_2_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("secondary_2: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def secondary_3_model(): 		#define secondary_3 model # 16997(9762)		# whoops! this is identical to initial_1
	# model = Sequential()	# create model
	# model.add(Dense(12, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=secondary_3_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("secondary_3: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def secondary_4_model(): 		#define secondary_4 model # 16788(8598) # marginal improvement in MSE, std went up a lot
	# model = Sequential()	# create model
	# model.add(Dense(12, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=secondary_4_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("secondary_4: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def secondary_5_model(): 		#define secondary_5 model # 11008(2907)            # whoah! drastic improvement!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=secondary_5_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("secondary_5: %.4f (%.4f) MSE" % (results.mean(), results.std()))

####################################################################################
# 
#	secondary_5 shows massive improvement. expand on that. 
# 
####################################################################################

# def tertiary_1_model(): 		#define tertiary_1 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_2_model(): 		#define tertiary_2 model # 10910(7167) 
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(19, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(10, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_2_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_2: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_3_model(): 		#define tertiary_3 model # 9974(3432) 
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(21, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_3_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_3: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_4_model(): 		#define tertiary_4 model # 15537(12330) 
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(10, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_4_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_4: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_model(): 		#define tertiary_5 model # 7531(2018)  # best yet!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5: %.4f (%.4f) MSE" % (results.mean(), results.std()))

####################################################################################
# 
#	tertiary_5 best yet, tertiary_1 very good, too. expand on these a lot for overnight runs. running 40 extensions
# 
####################################################################################

# def tertiary_5_alt_1_model(): 		#define tertiary_5_alt_1 model # 12559(6343)  
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(19, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_2_model(): 		#define tertiary_5_alt_2 model # 9083(3636)  
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(17, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_2_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_2: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_3_model(): 		#define tertiary_5_alt_3 model # 6699(1496)  
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(20, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_3_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_3: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_4_model(): 		#define tertiary_5_alt_4 model # 15032(13207)  
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(22, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_4_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_4: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_5_model(): 		#define tertiary_5_alt_5 model # 8338(3280)  
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(24, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_5_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_5: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_6_model(): 		#define tertiary_5_alt_6 model # 8753(6441)  
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(26, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_6_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_6: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_7_model(): 		#define tertiary_5_alt_7 model # 5713(2402)  
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_7_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_7: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_8_model(): 		#define model # 9241(5760)  
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_8_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_8: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_9_model(): 		#define tertiary_5_alt_9 model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(19, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_9_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_9: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_10_model(): 		#define tertiary_5_alt_10 model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_10_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_10: %.4f (%.4f) MSE" % (results.mean(), results.std()))

##

# def tertiary_1_alt_1_model(): 		#define tertiary_1_alt_1 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_2_model(): 		#define tertiary_1_alt_2 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_2_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_2: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_3_model(): 		#define tertiary_1_alt_3 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_3_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_3: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_4_model(): 		#define tertiary_1_alt_4 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_4_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_4: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_5_model(): 		#define tertiary_1_alt_5 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_5_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_5: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_6_model(): 		#define tertiary_1_alt_6 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_6_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_6: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_7_model(): 		#define tertiary_1_alt_7 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_7_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_7: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_8_model(): 		#define tertiary_1_alt_8 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(9, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_8_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_8: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_9_model(): 		#define tertiary_1_alt_9 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_9_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_9: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_10_model(): 		#define tertiary_1_alt_10 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(28, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(17, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_10_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_10: %.4f (%.4f) MSE" % (results.mean(), results.std()))

#########

# def tertiary_5_alt_1_v1_model(): 		#define tertiary_5_alt_1 model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(19, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_1_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_1_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_2_v1_model(): 		#define tertiary_5_alt_2 model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(17, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_2_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_2_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_3_v1_model(): 		#define tertiary_5_alt_3 model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(20, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_3_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_3_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_4_v1_model(): 		#define tertiary_5_alt_4 model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(22, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_4_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_4_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_5_v1_model(): 		#define tertiary_5_alt_5 model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(24, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_5_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_5_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_6_v1_model(): 		#define tertiary_5_alt_6 model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(26, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_6_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_6_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_7_v1_model(): 		#define tertiary_5_alt_7 model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_7_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_7_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_8_v1_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_8_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_8_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_9_v1_model(): 		#define tertiary_5_alt_9 model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(19, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_9_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_9_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_5_alt_10_v1_model(): 		#define tertiary_5_alt_10 model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_10_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_10_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

##

# def tertiary_1_alt_1_v1_model(): 		#define tertiary_1_alt_1 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_1_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_1_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_2_v1_model(): 		#define tertiary_1_alt_2 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_2_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_2_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_3_v1_model(): 		#define tertiary_1_alt_3 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_3_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_3_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_4_v1_model(): 		#define tertiary_1_alt_4 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_4_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_4_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_5_v1_model(): 		#define tertiary_1_alt_5 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_5_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_5_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_6_v1_model(): 		#define tertiary_1_alt_6 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_6_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_6_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_7_v1_model(): 		#define tertiary_1_alt_7 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(7, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_7_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_7_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_8_v1_model(): 		#define tertiary_1_alt_8 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(9, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_8_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_8_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_9_v1_model(): 		#define tertiary_1_alt_9 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_9_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_9_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def tertiary_1_alt_10_v1_model(): 		#define tertiary_1_alt_10 model # 7845(3064)		# big improve!
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(17, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_1_alt_10_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_1_alt_10_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

##########
# tertiary_5_alt_1: 12559.9201 (6343.5722) MSE
# tertiary_5_alt_2: 9083.0533 (3635.9570) MSE
# tertiary_5_alt_3: 6698.7099 (1496.4358) MSE
# tertiary_5_alt_4: 15032.1349 (13206.9708) MSE
# tertiary_5_alt_5: 8337.7641 (3279.8210) MSE
# tertiary_5_alt_6: 8753.3249 (6440.5050) MSE
# tertiary_5_alt_7: 5713.3653 (2402.5405) MSE				# A 
# tertiary_5_alt_8: 9240.7840 (5759.5981) MSE
# tertiary_5_alt_9: 10170.9618 (5002.9874) MSE
# tertiary_5_alt_10: 6532.4723 (2210.9217) MSE
# tertiary_1_alt_1: 14934.2947 (8926.9461) MSE
# tertiary_1_alt_2: 17543.6376 (9622.4625) MSE
# tertiary_1_alt_3: 11878.7781 (3430.1698) MSE
# tertiary_1_alt_4: 12825.9241 (8702.1585) MSE
# tertiary_1_alt_5: 13202.8728 (7553.4016) MSE
# tertiary_1_alt_6: 7845.5975 (3064.9856) MSE
# tertiary_1_alt_7: 8782.6812 (1375.1374) MSE
# tertiary_1_alt_8: 7665.6011 (3578.2730) MSE
# tertiary_1_alt_9: 9559.3172 (5454.5706) MSE
# tertiary_1_alt_10: 8206.2840 (3722.9984) MSE
# tertiary_5_alt_1_v1: 11274.7287 (5001.6628) MSE
# tertiary_5_alt_2_v1: 9348.5114 (1116.5686) MSE
# tertiary_5_alt_3_v1: 7333.3944 (2646.2335) MSE
# tertiary_5_alt_4_v1: 5897.0174 (1838.6255) MSE			# B 
# tertiary_5_alt_5_v1: 6305.1817 (1447.2168) MSE
# tertiary_5_alt_6_v1: 10788.4687 (7809.6964) MSE
# tertiary_5_alt_7_v1: 7937.4837 (3672.7051) MSE
# tertiary_5_alt_8_v1: 5691.2505 (1559.7318) MSE			# C !!!! great result and very low std
# tertiary_5_alt_9_v1: 11614.5410 (3383.0192) MSE
# tertiary_5_alt_10_v1: 6757.9056 (2722.9458) MSE
# tertiary_1_alt_1_v1: 20746.0778 (8801.0268) MSE
# tertiary_1_alt_2_v1: 13169.8698 (7632.0453) MSE
# tertiary_1_alt_3_v1: 8619.8844 (3550.3324) MSE
# tertiary_1_alt_4_v1: 15439.8644 (11749.9562) MSE
# tertiary_1_alt_5_v1: 13067.4719 (8580.6133) MSE
# tertiary_1_alt_6_v1: 12886.1772 (3321.9299) MSE
# tertiary_1_alt_7_v1: 6822.2222 (2483.3408) MSE
# tertiary_1_alt_8_v1: 11225.1572 (4544.1746) MSE
# tertiary_1_alt_9_v1: 5798.6547 (2339.5162) MSE			# D
# tertiary_1_alt_10_v1: 7949.9455 (4882.9633) MSE

# Best to Worst of best four models (this run, also best four models of all runs) : C, A, B, D
#
# I think I'm happy with the reslts of C, and am going to spin up maybe 10 more models based on these 4,
# will move on to another project after this treasure-hunting round 
##########

# def treasure_hunt_1_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(38, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_1: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def treasure_hunt_2_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_2_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_2: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def treasure_hunt_3_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_3_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_3: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def treasure_hunt_4_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_4_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_4: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def treasure_hunt_5_a_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(33, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_5_a_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_5_a: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def treasure_hunt_5_b_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(33, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(23, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_5_b_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_5_b: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def treasure_hunt_5_c_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(33, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(23, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_5_c_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_5_c: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def treasure_hunt_6_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(48, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(38, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_6_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_6: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def treasure_hunt_7_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(48, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(38, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_7_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_7: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def treasure_hunt_8_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(38, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_8_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_8: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def treasure_hunt_9_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(22, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_9_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_9: %.4f (%.4f) MSE" % (results.mean(), results.std()))

# def treasure_hunt_10_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(22, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=treasure_hunt_10_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("treasure_hunt_10: %.4f (%.4f) MSE" % (results.mean(), results.std()))


##########
# post treasure-hunting notes
# 
# treasure_hunt_1:    7781.8471 (5970.7003) MSE
# treasure_hunt_2:    9377.0311 (4517.4399) MSE
# treasure_hunt_3:    6757.9056 (2722.9458) MSE
# treasure_hunt_4:    7154.1196 (3330.7417) MSE
# treasure_hunt_5_a: 6851.7413 (3014.2332) MSE
# treasure_hunt_5_b: 6455.2076 (1965.3980) MSE
# treasure_hunt_5_c: 6111.6082 (2708.8775) MSE
# treasure_hunt_6:    9784.4739 (4396.0060) MSE
# treasure_hunt_7:    6412.9884 (2738.2481) MSE
# treasure_hunt_8:    8742.1431 (6556.9743) MSE
# treasure_hunt_9:    6701.1283 (3139.9642) MSE
# treasure_hunt_10:  8685.4364 (3442.6768) MSE
# 
# no improvement over best model
##########
# best model :
# 
# def tertiary_5_alt_8_v1_model(): 		#define model # ()  
	# model = Sequential()	# create model
	# model.add(Dense(38, input_dim=12, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(28, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(18, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam') # Compile model
	# return model

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=tertiary_5_alt_8_v1_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("tertiary_5_alt_8_v1: %.4f (%.4f) MSE" % (results.mean(), results.std()))
# 
##########