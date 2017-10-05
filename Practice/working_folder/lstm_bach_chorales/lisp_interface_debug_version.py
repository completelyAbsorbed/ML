# lisp_interface.py
# much of this code from answer by georg on stackoverflow : https://stackoverflow.com/questions/14058985/parsing-a-lisp-file-with-python

inputdata = '(1 ((st 8) (pitch 67) (dur 4) (keysig 1) (timesig 12) (fermata 0))((st 12) (pitch 67) (dur 8) (keysig 1) (timesig 12) (fermata 0)))'

from pyparsing import OneOrMore, nestedExpr

data = OneOrMore(nestedExpr()).parseString(inputdata)
print data

# [['1', [['st', '8'], ['pitch', '67'], ['dur', '4'], ['keysig', '1'], ['timesig', '12'], ['fermata', '0']], [['st', '12'], ['pitch', '67'], ['dur', '8'], ['keysig', '1'], ['timesig', '12'], ['fermata', '0']]]]



# data = open('chorales.lisp').read().split('\n\n')  
# print 
# print 
# print 
# print len(data)
# print 
# print 
# print 
# for step in range(0,len(data) - 1): 		# -1 because lisp formatting will encourage python to read a line beyond the last, causing error
	# data_sub = OneOrMore(nestedExpr()).parseString(data[step])
	# print data_sub

# read_lisp
def read_lisp(filename, split_char='\n\n'):
	data = open(filename).read().split(split_char)  
	top_list = []
	for step in range(0,len(data) - 1): 		# -1 because lisp formatting will encourage python to read a line beyond the last, causing error
		data_sub = OneOrMore(nestedExpr()).parseString(data[step])
		top_list.append(list(data_sub))
	return top_list

# write_lisp. have not tested
def write_lisp(x):
    return '(%s)' % ' '.join(lisp(y) for y in x) if isinstance(x, list) else x









######################################
print 
print 
print 
print 
print 
our_data = read_lisp(filename = 'chorales.lisp', split_char = '\n\n')
print len(our_data)
print 
print 
print 
print 
print 
print len(our_data[0])
print 
print 
print 
print 
print 
# print our_data[99][0]
print type(our_data[99][0][46])
print our_data[99][0][46]
print 
print 
print 
print 
print 
print our_data[99][0][46][0]
print 
print 
print 
print 
print 
print our_data[99][0][46][0][0]
print type(our_data[99][0][46][0][0])
print 
print 
print 
print 
print 
print (len(our_data[99][0]))