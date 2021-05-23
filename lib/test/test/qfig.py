import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib as mpl
# from matplotlib import rc
# rc('text', usetex=True)
# rc('font', **{'family': "sans-serif"})
# params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
# mpl.rcParams['axes.xmargin'] = 0
# mpl.rcParams['axes.ymargin'] = 0


Q = np.load('./data/Q3.npy')
Qindlist = np.nonzero(np.array(Q))

Qind = np.concatenate([Qindlist[0].reshape(-1, 1),
                       Qindlist[1].reshape(-1, 1), Qindlist[2].reshape(-1, 1)], axis=1)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(0.02141806 * Qind[:, 0] - 0.535451462, 0.0225498 * Qind[:, 1] - 0.563745006,
             0.01 * Qind[:, 2] - 0.25, marker=".", linestyle='None')
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$Y$")
ax.set_zlabel(r"$\theta$")
plt.show()
fig.savefig('./image/Qfig3.pdf')
