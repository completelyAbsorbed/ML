# lstm_bach_chorales_a.py
from lisp_interface import read_lisp
from pandas import DataFrame
from pandas import Series
from random import shuffle
from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences


def makespace(lines=5):
	for line in range(0,lines):
		print

chorales_data = read_lisp(filename = 'chorales.lisp', split_char = '\n\n')

# chorales_data[0:99]													~ 100 different chorales
# chorales_data[chorale_num][0][0]								~ chorale_num
# chorales_data[chorale_num][0][1][slice_num]					~ slice-by-slice of individual chorale
# chorales_data[0chorale_num][0][1][slice_num][0][1]		~ slice start time, measured in 16th notes from beginning time 0
# chorales_data[0chorale_num][0][1][slice_num][1][1]		~ slice pitch, MIDI number (60 = C4, up through 75)
# chorales_data[0chorale_num][0][1][slice_num][2][1]		~ slice duration, measured in 16th notes
# chorales_data[0chorale_num][0][1][slice_num][3][1]		~ slice key signature, number of sharps or flats, positive for sharps, negative for flats
# chorales_data[0chorale_num][0][1][slice_num][4][1]		~ slice time signature, in 16th notes per bar
# chorales_data[0chorale_num][0][1][slice_num][5][1]		~ slice fermata, 0 or 1 depending whether note event is under a fermata
# 
# chorales_data[0chorale_num][0][slice_num][...][0]		~ should be 'st', 'pitch', 'dur', 'keysig', 'timesig', 'fermata'


###################################
# assemble the dataset to be used for this script
# 
# problem (b) we will start with a regression prediction, but may want to try this version as categorical as well
#
###################################


# use start time [0], duration [2], target : time signature [4]
keep_cols = [0, 2, 4]
col_names = ['start_time', 'duration', 'time_signature']

sub_chorales = []
sub_feature_names = []
time_signatures = []

for row in range(0, len(chorales_data)):
	sub_chorales.append(list([chorales_data[row][0][1:][0][i][1] for i in keep_cols]))
	sub_feature_names.append(list([chorales_data[row][0][1:][0][i][0] for i in keep_cols]))
	time_signatures.append(list([chorales_data[row][0][1:][0][4][1]]))


# sub_chorales = [chorales_data[:][0][1:][0][i] for i in [0, 2, 4]]
# sub_chorales = [chorales_data[:][0][1:][0][i][1] for i in [0, 2, 4]]
# sub_chorales = [chorales_data[:][0][1:][0][i][1] for i in [0, 2, 4]] # just field names, could do a check with this
flat_list = [item for sublist in sub_feature_names for item in sublist]
my_set = set(flat_list)
makespace()
print 'feature names extracted : ...'
print my_set
makespace()

flat_list = [item for sublist in time_signatures for item in sublist]
my_set = set(flat_list)
print 'unique time signatures : ...'
print my_set
makespace()

###################################
# make a DataFrame
#
# then, groom data for LSTM
# 
# note, a bit cheat-y maybe, but since time signature is only 12 or 16 (3/4 or 4/4), I will convert this to a binary column 
# 
###################################

# make list of df
chorales_df = DataFrame(sub_chorales, columns = col_names)
makespace()

entries_chorales = []	
	
for row in range(0, len(chorales_data)):
	sub_list = chorales_data[row][0][1:]
	df_list = []
	for sub_row in range(0, len(sub_list)):
		df_list.append(list([sub_list[sub_row][0][1], sub_list[sub_row][2][1], sub_list[sub_row][4][1]]))
	sub_df = DataFrame(df_list, columns = col_names)
	entries_chorales.append(sub_df)
	
# contruct binary target column
target = []
for row in range(0, 100):
	if(int(chorales_df.iloc[row, 2]) == 16):
		target.append(1)
	else:
		target.append(0)
		
chorales_df['target'] = Series(target, index = chorales_df.index)
chorales_df['id'] = Series(range(0, 100), index = chorales_df.index)
# convert all values to integer		
for row in range(0,100):
	for col in range(0,5):
		chorales_df.iloc[row,col] = int(chorales_df.iloc[row,col])
# some test splitting steps
indices = range(0,100)
shuffle(indices)
train_indices = indices[:50]
test_indices = indices[50:]
X_names = ['start_time', 'duration']
Y_names = ['time_signature']

X_train = []
Y_train = []
X_test = []
Y_test = []		
		
for row in range(0,100):
	sub_df = entries_chorales[row]
	sub_X = sub_df[X_names]
	sub_Y = sub_df[Y_names]
	if (row in train_indices):
		X_train.append(sub_X)
		Y_train.append(sub_Y)
	else:
		X_test.append(sub_X)
		Y_test.append(sub_Y)

# maximum sequence length is 86 consider padding a bit above or below this amount (100???, 25???)

###################################
# define and execute a vanilla LSTM
# 
###################################

# one hot encode sequence
def one_hot_encode(sequence, n_features):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_features)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# define model
length = 25           #also try 100
num_cols = 2
n_features = 2
out_index = 2
model = Sequential()
model.add(LSTM(25, input_shape=(length*num_cols, n_features)))
model.add(Dense(n_features, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

# fit model
for i in range(50):
	# X, y = generate_example(length, n_features, out_index)
	df_X = X_train[i]
	# print train_indices[i]
	# print target[train_indices[i]]
	# y = one_hot_encode(target[train_indices[i]], n_features)
	if (target[train_indices[i]] == 1):
		y = [1, 0]
	else:
		y = [0, 1]
	y = array(y).reshape(1, n_features)
	st = list(df_X['start_time'])
	st = pad_sequences([st], maxlen = length, truncating = 'post')
	dur = list(df_X['duration'])
	dur = pad_sequences([dur], maxlen = length, truncating = 'post')	
	X = [st, dur]
	X = array(X).reshape(1, length, n_features)
	model.fit(X, y, epochs=1, verbose=2)

# evaluate model
correct = 0
for i in range(50):
	X, y = generate_example(length, n_features, out_index)
	yhat = model.predict(X)
	if one_hot_decode(yhat) == one_hot_decode(y):
		correct += 1
print('Accuracy: %f' % ((correct/100)*100.0))

# prediction on new data
# X, y = generate_example(length, n_features, out_index)
# yhat = model.predict(X)
# print('Sequence:  %s' % [one_hot_decode(x) for x in X])
# print('Expected:  %s' % one_hot_decode(y))
# print('Predicted: %s' % one_hot_decode(yhat))




































	
