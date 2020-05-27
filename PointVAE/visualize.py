import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('augmented_data.pkl', 'rb') as handle:
    augmented_data = pickle.load(handle)

for data in augmented_data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0][:, 0], data[0][:, 1], data[0][:, 2])
    fig.show()