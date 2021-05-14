import torch
import gpytorch
import numpy as np
torch.manual_seed(1)
np.random.seed(3)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, z_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(z_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        outputscale_constraint = gpytorch.constraints.Positive(
            initial_value=torch.tensor(0.05))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            outputscale_constraint=outputscale_constraint, base_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=z_train.size(1)))

    def forward(self, z):
        mean = self.mean_module(z)
        covar = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def train(args, z_train, y_train):
    likelihood_list = [gpytorch.likelihoods.GaussianLikelihood()
                       for i in range(3)]
    model_list = [ExactGPModel(
        z_train, y_train[:, i], likelihood_list[i]) for i in range(3)]
    gpmodels = gpytorch.models.IndependentModelList(
        model_list[0], model_list[1], model_list[2])
    likelihoods = gpytorch.likelihoods.LikelihoodList(
        model_list[0].likelihood, model_list[1].likelihood, model_list[2].likelihood)
    mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihoods, gpmodels)

    gpmodels.train()
    likelihoods.train()
    optimizer = torch.optim.Adam(gpmodels.parameters(), lr=0.2)
    for i in range(args.gpudate_num):
        optimizer.zero_grad()
        output = gpmodels(*gpmodels.train_inputs)
        loss = -mll(output, gpmodels.train_targets)
        loss.backward()
        optimizer.step()

    covs = [gpmodels(*gpmodels.train_inputs)
            [i].covariance_matrix.to('cpu').detach().numpy().astype(np.float64) for i in range(3)]
    noises = [gpmodels.models[i].likelihood.noise.to(
        'cpu').detach().numpy().reshape(-1, 1).astype(np.float64) for i in range(3)]
    gpmodels.eval()
    likelihoods.eval()
    return gpmodels, likelihoods, covs, noises
