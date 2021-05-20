import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
Q = np.load('./data/Q.npy')
Qindlist = np.nonzero(np.array(Q))
Qind = np.concatenate([Qindlist[0].reshape(-1, 1), Qindlist[1].reshape(-1, 1), Qindlist[2].reshape(-1, 1)], axis=1)
ax.plot(Qind[:, 0], Qind[:, 1], Qind[:, 2], marker="o", linestyle='None')
fig.savefig('./image/Qfig.pdf')
