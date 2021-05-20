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
            z_train.shape[1]), length_scale_bounds=(1e-20, 1e20))
        self.whtk = WhiteKernel(noise_level=self.noise)
        self.csk = ConstantKernel(constant_value_bounds=(1e-20, 1e20))
        self.gpr = GaussianProcessRegressor(
            alpha=1e-6,
            kernel=self.csk * self.rbfk + self.whtk,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=0
        )
        self.gpr.fit(self.z_train, self.y_train)

    def predict(self, z_test):
        means, stds = self.gpr.predict(
            z_test, return_std=True)
        return means, stds

# self.rbfk = [RBF(length_scale=np.ones(
        #     z_train.shape[1]), length_scale_bounds=(1e-20, 1e20)) for i in range(3)]
        # self.whtk = [WhiteKernel(noise_level=self.noise) for i in range(3)]
        # self.gpr = [GaussianProcessRegressor(
        #     alpha=1e-6,
        #     kernel=1 * self.rbfk[i] + self.whtk[i],
        #     # normalize_y=True,
        #     n_restarts_optimizer=10,
        #     random_state=0
        # ) for i in range(3)]
        # for i in range(3):
        #     self.gpr[i].fit(self.z_train, self.y_train[:, i])

# means = np.zeros(3)
        # stds = np.zeros(3)
        # for i in range(3):
        #     means[i], stds[i] = self.gpr[i].predict(
        #         z_test, return_std=True)

# # class ExactGPModel(gpytorch.models.ExactGP):
# #     def __init__(self, z_train, y_train, likelihood):
# #         super(ExactGPModel, self).__init__(z_train, y_train, likelihood)
# #         self.mean_module = gpytorch.means.ConstantMean()
# #         outputscale_constraint = gpytorch.constraints.Positive(
# #             initial_value=torch.tensor(0.05))
# #         self.covar_module = gpytorch.kernels.ScaleKernel(
# #             outputscale_constraint=outputscale_constraint, base_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=z_train.size(1)))

# #     def forward(self, z):
# #         mean = self.mean_module(z)
# #         covar = self.covar_module(z)
# #         return gpytorch.distributions.MultivariateNormal(mean, covar)


# def train(args, z_train, y_train):
#     likelihood_list = [gpytorch.likelihoods.GaussianLikelihood()
#                        for i in range(3)]
#     model_list = [ExactGPModel(
#         z_train, y_train[:, i], likelihood_list[i]) for i in range(3)]
#     gpmodels = gpytorch.models.IndependentModelList(
#         model_list[0], model_list[1], model_list[2])
#     likelihoods = gpytorch.likelihoods.LikelihoodList(
#         model_list[0].likelihood, model_list[1].likelihood, model_list[2].likelihood)
#     mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihoods, gpmodels)

#     gpmodels.train()
#     likelihoods.train()
#     optimizer = torch.optim.Adam(gpmodels.parameters(), lr=0.2)
#     for i in range(args.gpudate_num):
#         optimizer.zero_grad()
#         output = gpmodels(*gpmodels.train_inputs)
#         loss = -mll(output, gpmodels.train_targets)
#         loss.backward()
#         optimizer.step()

#     covs = [gpmodels(*gpmodels.train_inputs)
#             [i].covariance_matrix.to('cpu').detach().numpy().astype(np.float64) for i in range(3)]
#     noises = [gpmodels.models[i].likelihood.noise.to(
#         'cpu').detach().numpy().reshape(-1, 1).astype(np.float64) for i in range(3)]
#     gpmodels.eval()
#     likelihoods.eval()
#     return gpmodels, likelihoods, covs, noises
