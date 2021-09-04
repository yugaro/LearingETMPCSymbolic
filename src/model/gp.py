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
            z_train.shape[1]), length_scale_bounds=(1e-40, 1e40))
        self.whtk = WhiteKernel(noise_level=self.noise,
                                noise_level_bounds=(1e-40, 1e40))
        self.csk = ConstantKernel(constant_value_bounds=(1e-40, 1e40))
        self.gpr = GaussianProcessRegressor(
            alpha=1e-3,
            kernel=self.csk * self.rbfk + self.whtk,
            # normalize_y=True,
            n_restarts_optimizer=100,
            random_state=0,
        )
        self.gpr.fit(self.z_train, self.y_train)

    def predict(self, z_test):
        means, stds = self.gpr.predict(
            z_test, return_std=True)
        return means, stds
