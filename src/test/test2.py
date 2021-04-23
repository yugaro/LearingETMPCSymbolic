import torch
import gpytorch
from matplotlib import pyplot as plt
torch.manual_seed(1)
train_num = 110
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

print(z_train.size())
print(y_train.size())


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
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for k in range(100):
    optimizer.zero_grad()
    output = models(*models.train_inputs)
    loss = -mll(output, models.train_targets)
    loss.backward()
    # print('Iter %d/%d - Loss: %.3f' %
    #       (k + 1, 100, loss.item()))
    # print('a')
    # print(models.models[0].covar_module.base_kernel.lengthscale)
    # print(models.models[1].covar_module.base_kernel.lengthscale)
    # print(models.models[2].covar_module.base_kernel.lengthscale)
    # print('b')
    optimizer.step()


x1t = (x1_max - x1_min) * torch.rand(test_num, 1) + x1_min
x2t = (x2_max - x2_min) * torch.rand(test_num, 1) + x2_min
x3t = (x3_max - x3_min) * torch.rand(test_num, 1) + x3_min
ut = (u_max - u_min) * torch.rand(test_num, 1) + u_min
# at = (a_max - a_min) * torch.rand(test_num, 1) + a_min

xt = torch.cat([x1t, x2t, x3t], dim=1)

models.eval()
likelihoods.eval()
z_test = torch.cat([xt, ut], dim=1)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihoods(*models(z_test, z_test, z_test))

s = 0
for model, prediction in zip(models.models, predictions):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    mean = prediction.mean
    lower, upper = prediction.confidence_region()

    tr_x = model.train_inputs[0][:, 0].detach().numpy()
    tr_y = model.train_targets.detach().numpy()

    # Plot training data as black stars
    ax.plot(tr_x, tr_y, 'k*')
    # Predictive mean as blue line
    ax.plot(z_test[:, 0].numpy(), mean.numpy(), 'b')
    # Shade in confidence
    ax.fill_between(z_test[:, 0].numpy(), lower.detach().numpy(),
                    upper.detach().numpy(), alpha=0.5)
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title('Observed Values (Likelihood)')
    fig.savefig('bbb{}.png'.format(s))
    s += 1
