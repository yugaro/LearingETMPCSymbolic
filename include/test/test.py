import torch
import gpytorch
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
xinit = [20, 20, 60]


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, z_train, y_train, likelihood):
        super(MultitaskGPModel, self).__init__(z_train, y_train, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            base_means=gpytorch.means.ConstantMean(), num_tasks=y_train.size(1)
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            data_covar_module=gpytorch.kernels.ScaleKernel(
                base_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=z_train.size(1))),
            num_tasks=y_train.size(1), rank=y_train.size(1)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        print(covar_x.evaluate())
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

x1 = (x1_max - x1_min) * torch.rand(train_num, 1) + x1_min
x2 = (x2_max - x2_min) * torch.rand(train_num, 1) + x2_min
x3 = (x3_max - x3_min) * torch.rand(train_num, 1) + x3_min
u = (u_max - u_min) * torch.rand(train_num, 1) + u_min
a = (a_max - a_min) * torch.rand(train_num, 1) + a_min

x = torch.cat([x1, x2, x3], dim=1)
f = torch.cat([torch.zeros(train_num, 1), u, x1 - x2], dim=1)
w = torch.cat([a, torch.zeros(train_num, 1),
               torch.zeros(train_num, 1)], dim=1)
d = torch.cat([-(nu1[0] + nu1[1] * x1 + nu1[2] * x1 * x1) / M1, -(nu2[0] +
                                                                  nu2[1] * x2 + nu2[2] * x2 * x2) / M2, torch.zeros(train_num, 1)], dim=1)

z_train = torch.cat([x, u], dim=1)
y_train = x + Delta * (f + w + d)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks=y_train.size(1))
model = MultitaskGPModel(z_train, y_train, likelihood)

model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    optimizer.zero_grad()
    output = model(z_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()


x1t = (x1_max - x1_min) * torch.rand(test_num, 1) + x1_min
x2t = (x2_max - x2_min) * torch.rand(test_num, 1) + x2_min
x3t = (x3_max - x3_min) * torch.rand(test_num, 1) + x3_min
ut = (u_max - u_min) * torch.rand(test_num, 1) + u_min
# at = (a_max - a_min) * torch.rand(test_num, 1) + a_min

xt = torch.cat([x1t, x2t, x3t], dim=1)

model.eval()
likelihood.eval()
z_test = torch.cat([xt, ut], dim=1)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(z_test))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()


for i in range(x.size(1)):
    with torch.no_grad():
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(z_train[:, 0].numpy(), y_train[:, i].numpy(), 'k*', c='r')
        ax.plot(z_test[:, 0].numpy(), mean[:, i].numpy(), marker='o', c='b')
        fig.savefig('aaa{}.png'.format(i))
