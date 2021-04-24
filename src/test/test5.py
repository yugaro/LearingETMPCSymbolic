import torch
import gpytorch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# from matplotlib import pyplot as plt
torch.manual_seed(1)
np.random.seed(1)
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
xinit = torch.tensor([[20, 20, 60], [20, 25, 60], [
                     25, 20, 60], [25, 25, 60], [20, 22.5, 60], [22.5, 20, 60], [22.5, 22.5, 60], [25, 25, 60], [18, 15, 60], [18, 18, 60]])

Time = 15
epochs = 10
gp_updata_time = 100


etax = 0.5
etax_v = torch.tensor([etax, etax, etax])
etau = 0.2

gamma0 = 0.1
gamma1 = 0.1
gamma2 = 0.1

X0_min = 18
X0_max = 25
X1_min = 15
X1_max = 25
X2_min = 30
X2_max = 100

v = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                  [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])

L = 1
b0 = 1
b1 = 1
b2 = 1

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, z_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(z_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        outputscale_constraint = gpytorch.constraints.Positive(initial_value=torch.tensor(0.05))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            outputscale_constraint=outputscale_constraint, base_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=z_train.size(1)))

    def forward(self, z):
        mean = self.mean_module(z)
        covar = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def cal_epsilon(alpha, Lambda, etax_v):
    return torch.sqrt(2 * (alpha**2) * (1 - torch.exp(-0.5 * torch.matmul(torch.matmul(etax_v, torch.inverse(Lambda)), etax_v))))


def kernel_check(x, c, Lambda):
    flag = 1
    for i in range(v.shape[0]):
        xprime = x + c * torch.matmul(torch.sqrt(Lambda), v[i])
        if xprime[0] < X0_min or X0_max < xprime[0]:
            flag = 0
            break
        if xprime[1] < X1_min or X1_max < xprime[1]:
            flag = 0
            break
        if xprime[2] < X2_min or X2_max < xprime[2]:
            flag = 0
            break
    return flag


z_train = torch.zeros(1, 4)
y_train = Delta * torch.tensor([[-nu1[0] / M1, -nu2[0] / M2, 0]])
for epoch in range(epochs):
    for t in range(Time):
        if t == 0:
            x = xinit[epoch]
        u = (u_max - u_min) * torch.rand(1) + u_min
        a = (a_max - a_min) * torch.rand(1) + a_min

        f = torch.tensor([0, u, x[0] - x[1]])
        w = torch.tensor([a, 0, 0])
        d = torch.tensor([-(nu1[0] + nu1[1] * x[0] + nu1[2] * x[0] * x[0]) /
                          M1, -(nu2[0] + nu2[1] * x[1] + nu2[2] * x[1] * x[1]) / M2, 0])
        x_next = x + Delta * (f + w + d)
        z = torch.cat([x, u], dim=0).reshape(1, -1)
        z_train = torch.cat([z_train, z], dim=0)
        y_train = torch.cat([y_train, (x_next - (x + Delta * (f + w))).reshape(1, -1)], dim=0)
        x = x_next

likelihood0 = gpytorch.likelihoods.GaussianLikelihood()
model0 = ExactGPModel(z_train[1:], y_train[1:, 0], likelihood0)
likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
model1 = ExactGPModel(z_train[1:], y_train[1:, 1], likelihood1)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
model2 = ExactGPModel(z_train[1:], y_train[1:, 2], likelihood2)

# print(model0.covar_module.lazy_covariance_matrix)

models = gpytorch.models.IndependentModelList(model0, model1, model2)
likelihoods = gpytorch.likelihoods.LikelihoodList(
    model0.likelihood, model1.likelihood, model2.likelihood)

mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihoods, models)

models.train()
likelihoods.train()
optimizer = torch.optim.Adam(models.parameters(), lr=0.2)

for k in range(gp_updata_time):
    optimizer.zero_grad()
    output = models(*models.train_inputs)
    loss = -mll(output, models.train_targets)
    loss.backward()
    optimizer.step()

K0 = models(
    *models.train_inputs)[0].covariance_matrix + torch.eye(Time * epochs) * a_max
K1 = models(*models.train_inputs)[1].covariance_matrix + \
    torch.eye(Time * epochs) * a_max
K2 = models(*models.train_inputs)[2].covariance_matrix + \
    torch.eye(Time * epochs) * a_max

beta0 = torch.sqrt(b0 * b0 - torch.matmul(torch.matmul(
    models.train_targets[0], torch.inverse(K0)), models.train_targets[0]) + Time * epochs)
beta1 = torch.sqrt(b1 * b1 - torch.matmul(torch.matmul(
    models.train_targets[1], torch.inverse(K1)), models.train_targets[1]) + Time * epochs)
beta2 = torch.sqrt(b2 * b2 - torch.matmul(torch.matmul(
    models.train_targets[2], torch.inverse(K2)), models.train_targets[2]) + Time * epochs)

alpha0x = models.models[0].covar_module.outputscale
Lambda0x = torch.diag(
    models.models[0].covar_module.base_kernel.lengthscale.reshape(-1)[:3])
alpha1x = models.models[1].covar_module.outputscale
Lambda1x = torch.diag(
    models.models[1].covar_module.base_kernel.lengthscale.reshape(-1)[:3])
alpha2x = models.models[2].covar_module.outputscale
Lambda2x = torch.diag(
    models.models[2].covar_module.base_kernel.lengthscale.reshape(-1)[:3])


X0 = torch.arange(X0_min, X0_max, etax)
X1 = torch.arange(X1_min, X1_max, etax)
X2 = torch.arange(X2_min, X2_max, etax)
Uq = torch.arange(u_min, u_max, etau)

epsilon0 = cal_epsilon(alpha0x, Lambda0x, etax_v)
epsilon1 = cal_epsilon(alpha1x, Lambda1x, etax_v)
epsilon2 = cal_epsilon(alpha2x, Lambda2x, etax_v)
c0 = 2 * torch.log((2 * (alpha0x**2)) / (2 * (alpha0x**2) - (epsilon0**2)))
c1 = 2 * torch.log((2 * (alpha1x**2)) / (2 * (alpha1x**2) - (epsilon1**2)))
c2 = 2 * torch.log((2 * (alpha2x**2)) / (2 * (alpha2x**2) - (epsilon2**2)))

# print('a')
# print(beta0)
# print(beta1)
# print(beta2)
# print(alpha0x)
# print(alpha1x)
# print(alpha2x)
# print(epsilon0)
# print(epsilon1)
# print(epsilon2)

print('symbolic model')
# X = torch.zeros([1, 3])
# Q = torch.zeros([1, 3])
# for i in range(X0.shape[0]):
#     for j in range(X1.shape[0]):
#         for k in range(X2.shape[0]):
#             x_vec = torch.tensor([X0[i], X1[j], X2[k]])
#             X = torch.cat([X, x_vec.reshape(1, -1)], dim=0)
#             if kernel_check(x_vec, c0, Lambda0x) and kernel_check(x_vec, c1, Lambda1x) and kernel_check(x_vec, c2, Lambda2x):
#                 Q = torch.cat([Q, x_vec.reshape(1, -1)], dim=0)
# X = X[1:].clone()
# Q = Q[1:].clone()
# Q_np = Q[1:].to('cpu').detach().numpy()
# np.save('./Q_np.npy', Q_np)
Q = torch.from_numpy(np.load('./Q_np.npy'))

# print(X)
# print(Q)
# print(X.shape)
# print(Q.shape)

models.eval()
likelihoods.eval()
for i in range(Q.shape[0]):
    x_test = Q[i]
    for j in range(Uq.shape[0]):
        u_test = Uq[j]
        z_test = torch.cat([x_test, torch.tensor([u_test])], dim=0).reshape(1, -1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihoods(*models(z_test, z_test, z_test))
        f = torch.tensor([0, u_test, x_test[0] - x_test[1]])

        mean0 = x_test[0] + Delta * f[0] + predictions[0].mean
        mean1 = x_test[1] + Delta * f[1] + predictions[1].mean
        mean2 = x_test[2] + Delta * f[2] + predictions[2].mean
        sigma0 = predictions[0].variance
        sigma1 = predictions[1].variance
        sigma2 = predictions[2].variance

        xpre0l = mean0 - \
            (L * etax + b0 * epsilon0 + beta0 * torch.sqrt(sigma0) + etax)
        xpre0u = mean0 + \
            (L * etax + b0 * epsilon0 + beta0 * torch.sqrt(sigma0) + etax)
        xpre1l = mean1 - \
            (L * etax + b1 * epsilon1 + beta1 * torch.sqrt(sigma1) + etax)
        xpre1u = mean1 + \
            (L * etax + b1 * epsilon1 + beta1 * torch.sqrt(sigma1) + etax)
        xpre2l = mean2 - \
            (L * etax + b2 * epsilon2 + beta2 * torch.sqrt(sigma2) + etax)
        xpre2u = mean2 + \
            (L * etax + b2 * epsilon2 + beta2 * torch.sqrt(sigma2) + etax)
        # print('a')
        # print(x_test[1])
        # print(mean1)
        # print(xpre1l)
        # print(xpre1u)
