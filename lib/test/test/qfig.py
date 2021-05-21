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


Q = np.load('./data/Q.npy')
Qindlist = np.nonzero(np.array(Q))

Qind = np.concatenate([Qindlist[0].reshape(-1, 1),
                       Qindlist[1].reshape(-1, 1), Qindlist[2].reshape(-1, 1)], axis=1)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(0.00836194 * Qind[:, 0] - 0.31357285, 0.008 * Qind[:, 1] - 0.3,
             0.00817288 * Qind[:, 2] - 0.30648284, marker=".", linestyle='None')
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$Y$")
ax.set_zlabel(r"$\theta$")
plt.show()
# fig.savefig('./image/Qfig.pdf')
