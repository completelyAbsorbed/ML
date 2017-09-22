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
initial_exploratory_flag = True
initial_ARIMA_flag = True
grid_ARIMA_flag = True
review_residual_flag = True
# end of flag declarations
################################################################################################
# declare filepaths, names, and other variables up front so minimal editing of process required
filename = 'champagne.csv'
prefix_split = 'champagne'
training_filename = prefix_split + 'Training.csv'
validation_filename = prefix_split + 'Validation.csv'
stationary_filename = prefix_split + 'Stationary.csv'
n_last_periods = 12
months_in_year = 12
train_size_factor = 0.5
group_start = '1964'
group_end = '1970'
tg_param = 'A'
trend = 'nc'
p_1 = 1
d_1 = 1
q_1 = 1
p_grid = range(0,7)
d_grid = range(0,3)
q_grid = range(0,7)
p_2 = 0
d_2 = 0
q_2 = 1
# end of non-flag declarations
################################################################################################

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
# initial line plot shows some clear seasonality, perhaps with increasing trend 
#		(barring most recent year included in the plot)
# 		there seem to be end-of-year spikes, middle-year humps, and steep decreases before 
#		end-of-year spikes.
#		it could be good to model the seasonal component and remove it,
#		or use differencing with one or two levels to make the series stationary.
# seasonal line plots show similar shapes
#		there is a drastic decrease each august, followed by a steep increase through the end
#		of the year.
#		The dips and subsequent increases vary somewhat from year to year.
#		most years have a less drastic decrease in July
#		the periods leading up to decrease vary somewhat, though appear mostly level 
# density plot shows that the distribution is not Gaussian (would look more like a normal curve)
#		additionally, the distribution has a long right tail, perhaps suggesting an exponential distribution
#		the median values for each year may indicate an increasing trend, though falling in 1970
#		the spread (body, middle 50%) appears reasonably stable
#		every year has outliers on the upper portion, with 1968 also having a lower outlier
# stationary line plot doesn't show obvious trend or seasonality,
#		therefore the stationary (seasonally differenced) dataset may be a good starting point
#		for our modeling.
#		no further distancing may be required, therefore parameter d may be set to 0
#		...
#	next step is to find our initial p and q for ARIMA, as our initial d will be 0
#	p is for Autoregression(AR)
#	q is for Moving Average(MA)
# 	...
# ACF & PACF plots 
#		Both plots seem to show significant lag at 1 and 12 months, with PACF also showing
#		notable lag at 13 months.
#		a good starting point for p and q are 1 (see literature for tuning ARIMA)
#
##### initial ARIMA(p, d, q), manual parameters	
if initial_ARIMA_flag:
	if not initial_exploratory_flag:
		dataset = Series.from_csv(filename, header=0)
		makespace()
	# do ARIMA process (using 1,1,1 parameters to start after some experimentation)
	rmse_arima_initial = cst.run_arima(p = p_1, d = d_1, q = q_1, dataset = dataset, train_size_factor = train_size_factor, trend = trend, print_flag = True)
	### notes on initial ARIMA
	#
	# using (p,d,q) = (1,1,1) achieved RMSE = 956.958, a significant improvement over persistence!
	#
##### grid search ARIMA 		
if grid_ARIMA_flag:
	# search for and print best configuration and score among p,d,q grid specifications
	arima_params_best, rmse_arima_best = cst.run_grid_search_arima(p_grid = p_grid, d_grid = d_grid, q_grid = q_grid, dataset = dataset, train_size_factor = train_size_factor, trend = trend, months_in_year = months_in_year, full_print_flag = False)
### notes on grid search ARIMA
#
# best (p,d,q) = (0,0,1), best RMSE = 939.464, may or may not be a statistically significant improvement
#
#### review residual errors an ARIMA(p_2, d_2, q_2)
if review_residual_flag:
	residuals = cst.examine_residual_error(p = p_2, d = d_2, q = q_2, dataset = dataset, train_size_factor = train_size_factor, trend = trend, months_in_year = months_in_year, print_flag = True)


		
################################################################################################
# editing notes
# 
# why are we doing train test split in initial exploratory analysis? should probably move this
# 
# end editing notes 
################################################################################################