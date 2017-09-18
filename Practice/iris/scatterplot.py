# Scatter Plot Matrix
import matplotlib
import pandas
from pandas.tools.plotting import scatter_matrix
csv_path = 'Wholesale customers data.csv'
data = pandas.read_csv(csv_path)
scatter_matrix(data)
matplotlib.pyplot.show()