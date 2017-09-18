import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import pylab as pl




lines = [[(0, 0), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

lc = mc.LineCollection(lines, colors = c, linewidths=5)

fig, ax = pl.subplots()

ax.add_collection(lc)

# ax.autoscale()

plt.axis('off') # turn off the axis

plt.show()