import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

X0 = np.arange(-1.5, 1.5 + 0.001, 0.1)
X1 = np.arange(-1.5, 1.5 + 0.001, 0.1)
X2 = np.arange(-1.5, 1.5 + 0.001, 0.1)

Qdata = np.load('Qdata_np2.npy')
Qsafe = np.load('Qsafe_np2.npy')
Qdataind = np.nonzero(Qdata)
Qsafeind = np.nonzero(Qsafe)

Udata_np = np.load('Udata_np2.npy')

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.plot(Qdataind[0], Qdataind[1],
#         Qdataind[2], marker="o", linestyle='None')
# fig.savefig('eee.png')
# plt.close()

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.plot(Qsafeind[0], Qsafeind[1],
#         Qsafeind[2], marker="o", linestyle='None')
# fig.savefig('fff.png')
# plt.close()

XX = np.zeros((1, 3))
for i in range(Qdataind[0].shape[0]):
    XX = np.concatenate([XX, np.array([
        X0[int(Qdataind[0][i])], X1[int(Qdataind[1][i])], X2[int(Qdataind[2][i])]]).reshape(1, -1)])

XX2 = np.zeros((1, 3))
for i in range(Qsafeind[0].shape[0]):
    XX2 = np.concatenate([XX2, np.array([
        X0[int(Qsafeind[0][i])], X1[int(Qsafeind[1][i])], X2[int(Qsafeind[2][i])]]).reshape(1, -1)])

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.plot(XX[:, 0], XX[:, 1], XX[:, 2], marker="o", linestyle='None')
fig.savefig('eee.png')
plt.close()

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.plot(XX2[:, 0], XX2[:, 1], XX2[:, 2], marker="o", linestyle='None')
fig.savefig('fff.png')
plt.close()

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.plot(Qsafeind[0], Qsafeind[1],
#         Qsafeind[2], marker="o", linestyle='None')
# fig.savefig('fff.png')
# plt.close()
