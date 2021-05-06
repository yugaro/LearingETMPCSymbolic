import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

Qdata = np.load('Qdata_np.npy')
Qsafe = np.load('Qsafe_np.npy')
Qdataind = np.nonzero(Qdata)
Qsafeind = np.nonzero(Qsafe)

Udata_np = np.load('Udata_np.npy')
print(Udata_np[:, :, 16, 0])
# print(np.nonzero(Udata_np))

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.plot(Qdataind[0], Qdataind[1],
        Qdataind[2], marker="o", linestyle='None')
fig.savefig('eee.png')
plt.close()

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.plot(Qsafeind[0], Qsafeind[1],
        Qsafeind[2], marker="o", linestyle='None')
fig.savefig('fff.png')
plt.close()
