import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('augmented_data.pkl', 'rb') as handle:
    augmented_data = pickle.load(handle)

for data in augmented_data:
    print(len(data['true']))
    fig = plt.figure()
    ax_pred = fig.add_subplot(121, projection='3d')
    ax_true = fig.add_subplot(122, projection='3d')
    ax_pred.scatter(data['pred'][:, 0], data['pred'][:, 1], data['pred'][:, 2])
    ax_true.scatter(data['true'][:, 0], data['true'][:, 1], data['true'][:, 2])
    plt.show(block=True)
