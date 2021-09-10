import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
np.random.seed(0)


class GP:
    def __init__(self, z_train, y_train, noise):
        self.z_train = z_train
        self.y_train = y_train
        self.noise = noise
        self.rbfk = RBF(length_scale=np.ones(
            z_train.shape[1]))
        self.whtk = WhiteKernel(noise_level=self.noise)
        self.csk = ConstantKernel()
        self.gpr = GaussianProcessRegressor(
            alpha=0,
            kernel=self.csk * self.rbfk + self.whtk,
            normalize_y=True,
            random_state=0,
            n_restarts_optimizer=20
        )
        self.gpr.fit(self.z_train, self.y_train)

    def predict(self, z_test):
        means, stds = self.gpr.predict(
            z_test, return_std=True)
        return means, stds


# length_scale_bounds=(1e-3, 1e3)
# noise_level_bounds = (1e-3, 1e3)
# constant_value_bounds = (1e-3, 1e3)
