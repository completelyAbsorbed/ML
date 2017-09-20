# cstProcess.py : present a barebones approach to using ChronSeqTools, from beginning to end
# some code copied or adapted from https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/
import ChronSeqTools as cst
from pandas import Series
from ChronSeqTools import makespace

################################################################################################
# define a series of flags so the program knows which steps we want to run
# this helps readability, save on compute cycles, and more
initial_exploratory_flag = True
initial_ARIMA_flag = False

# end of flag declarations
################################################################################################
# declare filepaths, names, and other variables up front so minimal editing of process required
filename = 'champagne.csv'
prefix_split = 'champagne'
training_filename = prefix_split + 'Training.csv'
validation_filename = prefix_split + 'Validation.csv'
n_last_periods = 12
train_size_factor = 0.5
group_start = '1964'
group_end = '1970'
tg_param = 'A'
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
	# view ACF and PACF plots 
	cst.autocorrelation_plots(dataset = dataset) # !!!!!!! need to make stationary
### notes on initial exploratory analysis :
# 
# 
# 
# 
# 
##### initial ARIMA(p, d, q), manual parameters	
if initial_ARIMA_flag:
	if not initial_exploratory_flag:
		dataset = Series.from_csv(filename, header=0)
		makespace()
	print 'hi!'
		
		
		
################################################################################################
# editing notes
# 
# !!! need to make stationary... we are in step 6.1 but skipped differencing. 
#		probably want to pass something besides dataset
#		anyway, pick back up here, and compare ACF/PACF plots to example before changing much else
# 
# why are we doing train test split in initial exploratory analysis? should probably move this
# 
# need to fill in good notes on initial exploratory analysis
# 
# end editing notes 
################################################################################################