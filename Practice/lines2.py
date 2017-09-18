import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import random
import pylab as pl


c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

line = [(0,0), (1,1)]
randColor = c[xrange(3),1]
lines = line


for x in range(0,200,1):
	rCol = c[random.sample(xrange(3),1)]
	rColA = rCol[0]
	randColor = randColor, rColA
	x1 = 0 + x*0.005
	y1 = 0
	lines = lines, [(x1,y1), (1,1)]
	
	
lc = mc.LineCollection(lines, colors = randColor, linewidths=5)
fig, ax = pl.subplots()
ax.add_collection(lc)

# ax.autoscale()

plt.axis('off') # turn off the axis

plt.show()