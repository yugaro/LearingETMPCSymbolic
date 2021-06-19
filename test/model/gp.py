import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
np.random.seed(0)


class GP:
    def __init__(self, z_train, y_train, noise):
        self.z_train = z_train
        self.y_train = y_train
        self.noise = noise
        self.rbfk = [RBF(length_scale=np.ones(
            z_train.shape[1]), length_scale_bounds=(1e-20, 1e20)) for i in range(3)]
        self.whtk = [WhiteKernel(noise_level=self.noise,
                                 noise_level_bounds=(1e-20, 1e20)) for i in range(3)]
        self.csk = [ConstantKernel(
            constant_value_bounds=(1e-20, 1e20)) for i in range(3)]
        self.gpr = [GaussianProcessRegressor(
            alpha=1e-3,
            kernel=self.csk[i] * self.rbfk[i] + self.whtk[i],
            n_restarts_optimizer=10,
            random_state=0,
        ) for i in range(3)]
        for i in range(3):
            self.gpr[i].fit(self.z_train, self.y_train[:, i])

    def predict(self, z_test):
        means = np.zeros(3)
        stds = np.zeros(3)
        for i in range(3):
            means[i], stds[i] = self.gpr[i].predict(z_test, return_std=True)
        return means, stds
