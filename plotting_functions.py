from matplotlib import pyplot
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

data = [
	[1, 1, 17.25],
	[1, 5, 16.79],
	[3, 1, 17.11],
	[3, 5, 17.58],
	[32, 1, 17.11],
	[32, 5, 19.47],
	[32, 40, 23.45]
]

x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
y = x.copy().T # transpose
z = np.cos(x ** 2 + y ** 2)
data = np.array(data)
fig = plt.figure()
ax = plt.axes(projection='3d')
print(data.shape)
z_data = np.tile(np.expand_dims(data[:, 2], axis=1), (1, data.shape[0]))
z_data = np.diag(data[:, 2])
ax.plot_surface(data[:, 0], data[:, 1], z_data,cmap='viridis', edgecolor='none')
plt.show()