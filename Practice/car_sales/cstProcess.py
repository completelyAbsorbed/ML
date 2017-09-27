# cstProcess.py : present a barebones approach to using ChronSeqTools, from beginning to end
# some code copied or adapted from https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/
# ARIMA "rules" to approach https://people.duke.edu/~rnau/arimrule.htm also in arimrule.txt in this repo
# How to Tune ARIMA Parameters in Python https://machinelearningmastery.com/tune-arima-parameters-python/
# ARIMA search on MLmastery https://machinelearningmastery.com/?s=arima&submit=Search
import ChronSeqTools as cst
from pandas import Series
from ChronSeqTools import makespace

################################################################################################
# define a series of flags so the program knows which steps we want to run
# this helps readability, save on compute cycles, and more
hot_open_flag = True
initial_exploratory_flag = False
initial_ARIMA_flag = False
grid_ARIMA_flag = False
review_residual_flag = True
validation_flag = False
# end of flag declarations
################################################################################################
# declare filepaths, names, and other variables up front so minimal editing of process required
filename = 'monthly_car_sales.csv'
prefix_split = 'car_sales'
training_filename = prefix_split + 'Training.csv'
validation_filename = prefix_split + 'Validation.csv'
stationary_filename = prefix_split + 'Stationary.csv'
n_last_periods = 12
months_in_year = 12
train_size_factor = 0.5
group_start = '1960'
group_end = '1967'   # set this for eliminating validation set
tg_param = 'A'
trend = 'nc'
p_1 = 1
d_1 = 1
q_1 = 1
p_grid = [0,1,2,3,12,15,17]
d_grid = range(0,3)
q_grid = [0,1,2,3,12,15,17]
p_2 = 1
d_2 = 0
q_2 = 1
bias = 165.90473
# end of non-flag declarations
################################################################################################

##### hot open data load script
if hot_open_flag:
	dataset = Series.from_csv(filename, header=0)
	makespace()

##### initial exploratory analysis
if initial_exploratory_flag:
	makespace()
	# load data and create champagneTraining.csv and champagneValidation.csv
	dataset, validation = cst.load(filename=filename, prefix_split=prefix_split, n_last_periods=n_last_periods)
	makespace()
	# split the dataset(training master set) into train, test
	train, test = cst.train_test_split(dataset = dataset, train_size_factor = train_size_factor)
	# persistence : establish performance baseline
	rmse_persistence = cst.persistence(train = train, test = test)
	makespace()
	# summary statistics & line, seasonal line, density, box-and-whisker plots
	cst.summaries(dataset = dataset, group_start = group_start, group_end = group_end, tg_param = tg_param)
	# make stationary, show plot 
	stationary = cst.make_stationary(dataset = dataset, months_in_year = months_in_year, filename = stationary_filename)
	makespace()
	# view ACF and PACF plots 
	cst.autocorrelation_plots(dataset = stationary) 
### notes on initial exploratory analysis :
# 
# initial line plot demonstrates some clear seasonality 
#		additionally, an increasing trend
# seasonal line plots show similar shapes, though significant variation is apparent
# 		January, February : tend to start out flat (1960-1961 are different, consider excluding)
#		March, April : increasing
#		May, June : up or down, not clear trend year to year
#		July, August : two step decreasing
#		September, October : tends to decrease, then sharp increase, or stay about same/slight increase, then sharp increase 
#		November, December : varies considerably. usually up then down, both slight. sometimes drastic
# density plot shows that the distribution is not Gaussian (would look more like a normal curve)
#		right tail is very fat. experiment with feature engineering and feature transformation
# bar plots show median increasing then decreasing 
# 		no outliers
#		some years tails are much longer than others
#		median skewed lower in each year
# stationary line plot seems to show some trend (up then down)
#		does this mean we shouldn't start with stationary? start with base? Feature Engineering?
#		gonna use d = 1 for initial model
#		use a grid on possible d values
# ACF & PACF plots 
#		Both plots show lag at 1,2, 12, 17 months
#		PACF shows lag at 15 months
#
# RMSE notes ~ see  rmse_notes.txt 
##### initial ARIMA(p, d, q), manual parameters	
if initial_ARIMA_flag:
	if not initial_exploratory_flag:
		dataset = Series.from_csv(filename, header=0)
		makespace()
	# do ARIMA process (using 1,1,1 parameters to start)
	rmse_arima_initial = cst.run_arima(p = p_1, d = d_1, q = q_1, dataset = dataset, train_size_factor = train_size_factor, trend = trend, print_flag = True)
	### notes on initial ARIMA
	#
	# using (p,d,q) = (1,1,1) achieved RMSE = 2046.045, a significant improvement over persistence!
	#
##### grid search ARIMA 		
if grid_ARIMA_flag:
	# search for and print best configuration and score among p,d,q grid specifications
	arima_params_best, rmse_arima_best = cst.run_grid_search_arima(p_grid = p_grid, d_grid = d_grid, q_grid = q_grid, dataset = dataset, train_size_factor = train_size_factor, trend = trend, months_in_year = months_in_year, full_print_flag = False)
### notes on grid search ARIMA
#
# best (p,d,q) = (1,0,1), best RMSE = 1874.408, nice slight improvement!
#
#### review residual errors an ARIMA(p_2, d_2, q_2)
if review_residual_flag:
	residuals = cst.examine_residual_error(p = p_2, d = d_2, q = q_2, dataset = dataset, train_size_factor = train_size_factor, trend = trend, months_in_year = months_in_year, print_flag = True)
	makespace()
	# mean of residuals : 165.90473, try a bias correction
	residuals_bias_correction = cst.examine_residual_error_bias_correction(p = p_2, d = d_2, q = q_2, dataset = dataset, train_size_factor = train_size_factor, trend = trend, months_in_year = months_in_year, print_flag = True, bias = bias)
	residuals_bias_correction_flip = cst.examine_residual_error_bias_correction(p = p_2, d = d_2, q = q_2, dataset = dataset, train_size_factor = train_size_factor, trend = trend, months_in_year = months_in_year, print_flag = True, bias = -bias)
	# look at autocorrelation plots for residuals
	cst.autocorrelation_plots(dataset = residuals)
	cst.autocorrelation_plots(dataset = residuals_bias_correction)	
##### Finalize Model, Make Prediction, Validate Model 
if validation_flag:
	# save validation model
	cst.save_validation_model(order = (p_2, d_2, q_2), filename = training_filename, bias = bias)
	makespace()
	# make prediction for first row of validation data
	cst.make_prediction(filename = training_filename) # predicted : 6794.773, actual : 6981, close!
	makespace()
	# validate model
	cst.validate_model(training_filename = training_filename, validation_filename = validation_filename) # RMSE = 361.110, very good!
			
################################################################################################
# editing notes
# 
# end editing notes 
################################################################################################