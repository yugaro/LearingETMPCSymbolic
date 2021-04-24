import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
torch.manual_seed(1)
train_num = 100
test_num = 30
x1_min = 18
x1_max = 25
x2_min = 15
x2_max = 25
x3_min = 30
x3_max = 100
M1 = 100
M2 = 100
Delta = 1
nu1 = [40, 1, 0.2]
nu2 = [50, 2, 0.1]
a_min = -0.02
a_max = 0.02
u_min = -1
u_max = 1
xinit = torch.tensor([20, 20, 60])

Time = 50
gp_updata_time = 50


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, z_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(z_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=z_train.size(1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

z_train = torch.zeros(1, 4)
y_train = Delta * torch.tensor([[-nu1[0] / M1, -nu2[0] / M2, 0]])
for t in range(Time):
    if t == 0:
        x = xinit
    u = (u_max - u_min) * torch.rand(1) + u_min
    a = (a_max - a_min) * torch.rand(1) + a_min

    f = torch.tensor([0, u, x[0] - x[1]])
    w = torch.tensor([a, 0, 0])
    d = torch.tensor([-(nu1[0] + nu1[1] * x[0] + nu1[2] * x[0] * x[0]) / M1, -(nu2[0] + nu2[1] * x[1] + nu2[2] * x[1] * x[1]) / M2, 0])
    x_next = x + Delta * (f + w + d)

    z = torch.cat([x, u], dim=0).reshape(1, -1)
    z_train = torch.cat([z_train, z], dim=0)
    y_train = torch.cat([y_train, x_next.reshape(1, -1)], dim=0)
    x = x_next

likelihood0 = gpytorch.likelihoods.GaussianLikelihood()
model0 = ExactGPModel(z_train, y_train[:, 0], likelihood0)
likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
model1 = ExactGPModel(z_train, y_train[:, 1], likelihood1)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
model2 = ExactGPModel(z_train, y_train[:, 2], likelihood2)

models = gpytorch.models.IndependentModelList(model0, model1, model2)
likelihoods = gpytorch.likelihoods.LikelihoodList(
    model0.likelihood, model1.likelihood, model2.likelihood)

mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihoods, models)

models.train()
likelihoods.train()
optimizer = torch.optim.Adam(models.parameters(), lr=0.1)

for k in range(gp_updata_time):
    optimizer.zero_grad()
    output = models(*models.train_inputs)
    loss = -mll(output, models.train_targets)
    loss.backward()
    optimizer.step()

models.eval()
likelihoods.eval()

z_test = torch.zeros(1, 4)
y_test = Delta * torch.tensor([[-nu1[0] / M1, -nu2[0] / M2, 0]])
lowerlist = torch.zeros(1, 3)
upperlist = torch.zeros(1, 3)
for t in range(Time):
    if t == 0:
        x = xinit
    u = (u_max - u_min) * torch.rand(1) + u_min

    z = torch.cat([x, u], dim=0).reshape(1, -1)
    z_train = torch.cat([z_test, z], dim=0)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihoods(*models(z, z, z))
    mean0 = predictions[0].mean
    mean1 = predictions[1].mean
    mean2 = predictions[2].mean
    # print(mean0 + 2 * torch.sqrt(predictions[0].variance))

    x_next = torch.cat([mean0, mean1, mean2], dim=0)
    y_test = torch.cat([y_test, x_next.reshape(1, -1)], dim=0)
    x = x_next

    lower0, upper0 = predictions[0].confidence_region()
    # print(upper0)
    lower1, upper1 = predictions[1].confidence_region()
    lower2, upper2 = predictions[2].confidence_region()
    lowers = torch.cat([lower0, lower1, lower2], dim=0)
    uppers = torch.cat([upper0, upper1, upper2], dim=0)
    lowerlist = torch.cat([lowerlist, lowers.reshape(1, -1)], dim=0)
    upperlist = torch.cat([upperlist, uppers.reshape(1, -1)], dim=0)

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.plot(y_train[1:, 0], c='r')
ax.plot(y_train[1:, 1], c='b')
fig.savefig('ccc.png')

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.plot(y_train[1:, 1], y_train[1:, 2], c='g')
fig.savefig('ddd.png')

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.plot(y_test[1:, 0], c='r')
ax.fill_between(np.arange(Time),
                lowerlist[1:, 0], upperlist[1:, 0], alpha=0.3, facecolor='r')
ax.plot(y_test[1:, 1], c='b')
ax.fill_between(np.arange(Time),
                lowerlist[1:, 1], upperlist[1:, 1], alpha=0.3, facecolor='b')
fig.savefig('eee.png')

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.plot(y_test[1:, 1], y_test[1:, 2], c='g')
fig.savefig('fff.png')
