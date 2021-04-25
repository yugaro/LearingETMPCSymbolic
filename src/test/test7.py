import torch
import gpytorch
import numpy as np
# from matplotlib import pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
INF = 1e9
torch.manual_seed(1)
np.random.seed(1)

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

Time = 5
epochs = 6
gp_updata_time = 300

etax = 0.5
etax_v = torch.tensor([etax, etax, etax])
etau = 0.2

X0_min = 15
X0_max = 25
X1_min = 15
X1_max = 25
X2_min = 30
X2_max = 100

X_range_list = [[X0_min, X0_max], [X1_min, X1_max], [X2_min, X2_max]]

v = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                  [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])

L = 0.1
b0 = 0.1
b1 = 0.1
b2 = 0.1


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
        y_train = torch.cat(
            [y_train, (x_next - (x + Delta * (f + w))).reshape(1, -1)], dim=0)
        x = x_next

likelihood0 = gpytorch.likelihoods.GaussianLikelihood()
model0 = ExactGPModel(z_train[1:], y_train[1:, 0], likelihood0)
likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
model1 = ExactGPModel(z_train[1:], y_train[1:, 1], likelihood1)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
model2 = ExactGPModel(z_train[1:], y_train[1:, 2], likelihood2)


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
print(beta0)
print(beta1)
print(beta2)

alpha0x = models.models[0].covar_module.outputscale
Lambda0x = torch.diag(
    models.models[0].covar_module.base_kernel.lengthscale.reshape(-1)[:3])
alpha1x = models.models[1].covar_module.outputscale
Lambda1x = torch.diag(
    models.models[1].covar_module.base_kernel.lengthscale.reshape(-1)[:3])
alpha2x = models.models[2].covar_module.outputscale
Lambda2x = torch.diag(
    models.models[2].covar_module.base_kernel.lengthscale.reshape(-1)[:3])

Lambdax_list = [Lambda0x, Lambda1x, Lambda2x]

X0 = torch.arange(X0_min, X0_max + etax, etax)
X1 = torch.arange(X1_min, X1_max + etax, etax)
X2 = torch.arange(X2_min, X2_max + etax, etax)
# U = torch.arange(u_min, u_max + etau, etau)
U = torch.tensor([0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8, 1.0, -1.0])

epsilon0 = cal_epsilon(alpha0x, Lambda0x, etax_v)
epsilon1 = cal_epsilon(alpha1x, Lambda1x, etax_v)
epsilon2 = cal_epsilon(alpha2x, Lambda2x, etax_v)
c0 = 2 * torch.log((2 * (alpha0x**2)) / (2 * (alpha0x**2) - (epsilon0**2)))
c1 = 2 * torch.log((2 * (alpha1x**2)) / (2 * (alpha1x**2) - (epsilon1**2)))
c2 = 2 * torch.log((2 * (alpha2x**2)) / (2 * (alpha2x**2) - (epsilon2**2)))

c_list = [c0, c1, c2]

gamma0 = (1.4142 * alpha0x - 2 * epsilon0) / 2
gamma1 = (1.4142 * alpha1x - 2 * epsilon1) / 2
gamma2 = (1.4142 * alpha2x - 2 * epsilon2) / 2

cqin0 = 2 * torch.log((2 * (alpha0x**2)) /
                      (2 * (alpha0x**2) - ((2 * epsilon0 + gamma0)**2)))
cqin1 = 2 * torch.log((2 * (alpha1x**2)) /
                      (2 * (alpha1x**2) - ((2 * epsilon1 + gamma1)**2)))
cqin2 = 2 * torch.log((2 * (alpha2x**2)) /
                      (2 * (alpha2x**2) - ((2 * epsilon2 + gamma2)**2)))

cqin_list = [cqin0, cqin1, cqin2]

print('symbolic model')


def min_max_range_cal(X_range_list, c_list, Lambda_list, etax):
    Xq0_min = INF
    Xq0_max = -INF
    Xq1_min = INF
    Xq1_max = -INF
    Xq2_min = INF
    Xq2_max = -INF
    for i in range(len(c_list)):
        Xq0_min_tmp = X_range_list[0][0] + c_list[i] * \
            torch.matmul(torch.sqrt(Lambda_list[i]), v[0])[0]
        if Xq0_min > Xq0_min_tmp:
            Xq0_min_tmp_round = torch.round(Xq0_min_tmp)
            if Xq0_min_tmp_round - Xq0_min_tmp < 0:
                Xq0_min_tmp_round += etax
            Xq0_min = Xq0_min_tmp_round

        Xq0_max_tmp = X_range_list[0][1] + c_list[i] * \
            torch.matmul(torch.sqrt(Lambda_list[i]), v[1])[0]
        if Xq0_max < Xq0_max_tmp:
            Xq0_max_tmp_round = torch.round(Xq0_max_tmp)
            if Xq0_max_tmp_round - Xq0_max_tmp > 0:
                Xq0_max_tmp_round -= etax
            Xq0_max = Xq0_max_tmp_round

        Xq1_min_tmp = X_range_list[1][0] + c_list[i] * \
            torch.matmul(torch.sqrt(Lambda_list[i]), v[2])[1]
        if Xq1_min > Xq1_min_tmp:
            Xq1_min_tmp_round = torch.round(Xq1_min_tmp)
            if Xq1_min_tmp_round - Xq1_min_tmp < 0:
                Xq1_min_tmp_round += etax
            Xq1_min = Xq1_min_tmp_round

        Xq1_max_tmp = X_range_list[1][1] + c_list[i] * \
            torch.matmul(torch.sqrt(Lambda_list[i]), v[3])[1]
        if Xq1_max < Xq1_max_tmp:
            Xq1_max_tmp_round = torch.round(Xq1_max_tmp)
            if Xq1_max_tmp_round - Xq1_max_tmp > 0:
                Xq1_max_tmp_round -= etax
            Xq1_max = Xq0_max_tmp_round

        Xq2_min_tmp = X_range_list[2][0] + c_list[i] * \
            torch.matmul(torch.sqrt(Lambda_list[i]), v[4])[2]
        if Xq2_min > Xq2_min_tmp:
            Xq2_min_tmp_round = torch.round(Xq2_min_tmp)
            if Xq2_min_tmp_round - Xq2_min_tmp < 0:
                Xq2_min_tmp_round += etax
            Xq2_min = Xq2_min_tmp_round

        Xq2_max_tmp = X_range_list[2][1] + c_list[i] * \
            torch.matmul(torch.sqrt(Lambda_list[i]), v[5])[2]
        if Xq2_max < Xq2_max_tmp:
            Xq2_max_tmp_round = torch.round(Xq2_max_tmp)
            if Xq2_max_tmp_round - Xq2_max_tmp > 0:
                Xq2_max_tmp_round -= etax
            Xq2_max = Xq2_max_tmp_round

    Xq_range_list = [[Xq0_min.item(), Xq0_max.item()], [
        Xq1_min.item(), Xq1_max.item()], [Xq2_min.item(), Xq2_max.item()]]
    Xq_range_list_ind = [[int((Xq0_min.item() - X_range_list[0][0]) / etax), int((Xq0_max.item() - X_range_list[0][0]) / etax)], [
        int((Xq1_min.item() - X_range_list[1][0]) / etax), int((Xq1_max.item() - X_range_list[1][0]) / etax)], [
            int((Xq2_min.item() - X_range_list[2][0]) / etax), int((Xq2_max.item() - X_range_list[2][0]) / etax)]]
    return Xq_range_list, Xq_range_list_ind

Xq_range_list, Xq_range_list_ind = min_max_range_cal(
    X_range_list, c_list, Lambdax_list, etax)
Xqin_range_list, Xqin_range_list_ind = min_max_range_cal(
    X_range_list, cqin_list, Lambdax_list, etax)
print(Xq_range_list)
print(Xqin_range_list)
print(Xq_range_list_ind)
print(Xqin_range_list_ind)

Q = torch.zeros([X0.shape[0], X1.shape[0], X2.shape[0]])
Q[Xq_range_list_ind[0][0]:Xq_range_list_ind[0][1],
    Xq_range_list_ind[1][0]:Xq_range_list_ind[1][1], Xq_range_list_ind[2][0]:Xq_range_list_ind[2][1]] = 1
Qin = torch.zeros([X0.shape[0], X1.shape[0], X2.shape[0]])
Qin[Xqin_range_list_ind[0][0]:Xqin_range_list_ind[0][1],
    Xqin_range_list_ind[1][0]:Xqin_range_list_ind[1][1], Xqin_range_list_ind[2][0]:Xqin_range_list_ind[2][1]] = 1
Uq = torch.ones([X0.shape[0], X1.shape[0], X2.shape[0]]) * INF

models.eval()
likelihoods.eval()

# for i in range(Q.shape[0]):
#     for j in range(Q.shape[1]):
#         for k in range(Q.shape[2]):
#             if Q[i][j][k] == 0:
#                 continue
#             else:
#                 x_test = torch.tensor([X0[i], X1[j], X2[k]])
#                 u_flag = 0
#                 for l in range(U.shape[0]):
#                     u_test = U[l]
#                     z_test = torch.cat(
#                         [x_test, torch.tensor([u_test])], dim=0).reshape(1, -1)
#                     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                         predictions = likelihoods(
#                             *models(z_test, z_test, z_test))

#                     f = torch.tensor([0, u_test, x_test[0] - x_test[1]])
#                     mean0 = x_test[0] + Delta * f[0] + predictions[0].mean
#                     mean1 = x_test[1] + Delta * f[1] + predictions[1].mean
#                     mean2 = x_test[2] + Delta * f[2] + predictions[2].mean
#                     sigma0 = predictions[0].variance
#                     sigma1 = predictions[1].variance
#                     sigma2 = predictions[2].variance

#                     xpre0l = mean0 - \
#                         (L * etax + b0 * epsilon0 +
#                          beta0 * torch.sqrt(sigma0) + etax)
#                     xpre0u = mean0 + \
#                         (L * etax + b0 * epsilon0 +
#                          beta0 * torch.sqrt(sigma0) + etax)
#                     xpre1l = mean1 - \
#                         (L * etax + b1 * epsilon1 +
#                          beta1 * torch.sqrt(sigma1) + etax)
#                     xpre1u = mean1 + \
#                         (L * etax + b1 * epsilon1 +
#                          beta1 * torch.sqrt(sigma1) + etax)
#                     xpre2l = mean2 - \
#                         (L * etax + b2 * epsilon2 +
#                          beta2 * torch.sqrt(sigma2) + etax)
#                     xpre2u = mean2 + \
#                         (L * etax + b2 * epsilon2 +
#                          beta2 * torch.sqrt(sigma2) + etax)
#                     ind0_min = int(torch.ceil((xpre0l - X0_min) / etax).item())
#                     ind0_max = int(((X0_max - xpre0u) // etax).item())
#                     ind1_min = int(torch.ceil((xpre1l - X1_min) / etax).item())
#                     ind1_max = int(((X1_max - xpre1u) // etax).item())
#                     ind2_min = int(torch.ceil((xpre2l - X2_min) / etax).item())
#                     ind2_max = int(((X2_max - xpre2u) // etax).item())

#                     if ind0_min < Xqin_range_list_ind[0][0] or ind1_min < Xqin_range_list_ind[1][0] or ind2_min < Xqin_range_list_ind[2][0] or ind0_max > Xqin_range_list_ind[0][1] or ind1_max > Xqin_range_list_ind[1][1] or ind2_max > Xqin_range_list_ind[2][1]:
#                         continue
#                     else:
#                         ind_flag = 1
#                         for ind0 in range(ind0_max - ind0_min):
#                             for ind1 in range(ind1_max - ind1_min):
#                                 for ind2 in range(ind2_max - ind2_min):
#                                     if Qin[ind0_min + ind0][ind1_min + ind1][ind2_min + ind2] == 0:
#                                         ind_flag = 0
#                                         break
#                                 if ind_flag == 0:
#                                     break
#                             if ind_flag == 0:
#                                 break
#                         if ind_flag == 1:
#                             Uq[i][j][k] = U[l]
#                             break
#                         elif ind_flag == 0:
#                             Q[i][j][k] == 0

# Q_np = Q.to('cpu').detach().numpy()
# np.save('./Q_np.npy', Q_np)
# Qin_np = Qin.to('cpu').detach().numpy()
# np.save('./Qin_np.npy', Qin_np)
# Uq_np = Uq.to('cpu').detach().numpy()
# np.save('./Uq_np.npy', Uq_np)

Q = torch.from_numpy(np.load('./Q_np.npy'))
Qin = torch.from_numpy(np.load('./Qin_np.npy'))
Uq = torch.from_numpy(np.load('./Uq_np.npy'))

print(torch.nonzero(Qin))
# print(Q)
# print(Qin)
# print(Uq)
