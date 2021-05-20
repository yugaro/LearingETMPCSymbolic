import numpy as np
import matplotlib.pyplot as plt

eta = np.arange(0.0001, 1, 0.0001)
y1 = np.zeros(eta.shape[0])
y2 = np.zeros(eta.shape[0])
Lambda = np.diag([7, 7, 7])
alpha = 1.5503284900418595
etav = np.concatenate(
    [eta.reshape(-1, 1), eta.reshape(-1, 1), eta.reshape(-1, 1)], axis=1)
for i in range(eta.shape[0]):
    y1[i] = np.log(eta[i])
    y2[i] = np.log(np.sqrt(2 * (alpha ** 2) * (1 - np.exp(-0.5 * etav[i, :].T @ np.linalg.inv(Lambda) @ etav[i, :]))))

fig, ax = plt.subplots(1, 1)
ax.plot(eta, y1, label='a')
ax.plot(eta, y2, label='b')
ax.legend()
fig.savefig('eta.pdf')

# 1.5503284900418595
# 6.8229335, 2.96716047, 5.54688528
