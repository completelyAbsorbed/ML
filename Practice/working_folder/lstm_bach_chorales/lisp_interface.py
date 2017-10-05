# lisp_interface.py
# much of this code from answer by georg on stackoverflow : https://stackoverflow.com/questions/14058985/parsing-a-lisp-file-with-python
from pyparsing import OneOrMore, nestedExpr

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


