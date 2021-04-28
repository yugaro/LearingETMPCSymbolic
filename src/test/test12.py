import torch
import gpytorch
import numpy as np
from casadi import vertcat, exp, SX, diag
import do_mpc
from matplotlib import rcParams
import cvxpy as cp
# import matplotlib.pyplot as plt

rcParams['axes.grid'] = True
rcParams['font.size'] = 18
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
INF = 1e9
torch.manual_seed(1)
np.random.seed(1)

M1 = 1000
M2 = 1000
Delta = 1
nu1 = [40, 1, 0.2]
nu2 = [50, 2, 0.1]
a_min = -0.02
a_max = 0.02
u_min = -1
u_max = 1
xinit = torch.tensor([[20, 20, 60], [20, 25, 60], [
                     25, 20, 60], [40, 20, 60], [20, 40, 80], [60, 30, 80], [30, 60, 60], [80, 40, 60], [40, 80, 60], [100, 100, 40]])

Time = 10
epochs = 10
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
b0 = 1
b1 = 1
b2 = 1


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

# def kernel_check(x, c, Lambda):
#     flag = 1
#     for i in range(v.shape[0]):
#         xprime = x + c * torch.matmul(torch.sqrt(Lambda), v[i])
#         if xprime[0] < X0_min or X0_max < xprime[0]:
#             flag = 0
#             break
#         if xprime[1] < X1_min or X1_max < xprime[1]:
#             flag = 0
#             break
#         if xprime[2] < X2_min or X2_max < xprime[2]:
#             flag = 0
#             break
#     return flag


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
            [y_train, (x_next - (x + Delta * (f))).reshape(1, -1)], dim=0)
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

K0_torch = models(*models.train_inputs)[0].covariance_matrix + \
    torch.eye(Time * epochs) * a_max
K1_torch = models(*models.train_inputs)[1].covariance_matrix + \
    torch.eye(Time * epochs) * a_max
K2_torch = models(*models.train_inputs)[2].covariance_matrix + \
    torch.eye(Time * epochs) * a_max

beta0 = torch.sqrt(b0 * b0 - torch.matmul(torch.matmul(
    models.train_targets[0], torch.inverse(K0_torch)), models.train_targets[0]) + Time * epochs)
beta1 = torch.sqrt(b1 * b1 - torch.matmul(torch.matmul(
    models.train_targets[1], torch.inverse(K1_torch)), models.train_targets[1]) + Time * epochs)
beta2 = torch.sqrt(b2 * b2 - torch.matmul(torch.matmul(
    models.train_targets[2], torch.inverse(K2_torch)), models.train_targets[2]) + Time * epochs)

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
U = torch.tensor([0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8, 1.0, -1.0])

epsilon0 = cal_epsilon(alpha0x, Lambda0x, etax_v)
epsilon1 = cal_epsilon(alpha1x, Lambda1x, etax_v)
epsilon2 = cal_epsilon(alpha2x, Lambda2x, etax_v)
c0 = 2 * torch.log((2 * (alpha0x**2)) / (2 * (alpha0x**2) - (epsilon0**2)))
c1 = 2 * torch.log((2 * (alpha1x**2)) / (2 * (alpha1x**2) - (epsilon1**2)))
c2 = 2 * torch.log((2 * (alpha2x**2)) / (2 * (alpha2x**2) - (epsilon2**2)))

c_list = [c0, c1, c2]
# gamma0 = (1.4142 * alpha0x - 2 * epsilon0) / 5
# gamma1 = (1.4142 * alpha1x - 2 * epsilon1) / 5
# gamma2 = (1.4142 * alpha2x - 2 * epsilon2) / 5

####################################################################################
alpha0 = np.sqrt(models.models[0].covar_module.outputscale.to(
    'cpu').detach().numpy())
alpha1 = np.sqrt(models.models[1].covar_module.outputscale.to(
    'cpu').detach().numpy())
alpha2 = np.sqrt(models.models[1].covar_module.outputscale.to(
    'cpu').detach().numpy())
Lambda0 = torch.diag(
    models.models[0].covar_module.base_kernel.lengthscale.reshape(-1)).to('cpu').detach().numpy() ** 2
Lambda1 = torch.diag(
    models.models[1].covar_module.base_kernel.lengthscale.reshape(-1)).to('cpu').detach().numpy() ** 2
Lambda2 = torch.diag(
    models.models[2].covar_module.base_kernel.lengthscale.reshape(-1)).to('cpu').detach().numpy() ** 2
ZT = models.train_inputs[0][0].to('cpu').detach().numpy()
K0 = models(
    *models.train_inputs)[0].covariance_matrix.to('cpu').detach().numpy().copy()
K1 = models(
    *models.train_inputs)[1].covariance_matrix.to('cpu').detach().numpy().copy()
K2 = models(
    *models.train_inputs)[2].covariance_matrix.to('cpu').detach().numpy().copy()
Y0 = models.train_targets[0].to('cpu').detach().numpy().copy()
Y1 = models.train_targets[1].to('cpu').detach().numpy().copy()
Y2 = models.train_targets[2].to('cpu').detach().numpy().copy()


################################################################################

def cal_kstar(zvar, ZT, alpha, Lambda):
    kstar = SX.zeros(ZT.shape[0])
    for i in range(ZT.shape[0]):
        kstar[i] = (SX(alpha) ** 2) * exp(-0.5 * (zvar - ZT[i]).T
                                          @ SX(np.linalg.inv(Lambda)) @ (zvar - ZT[i]))
    return kstar

model_type = 'discrete'
model = do_mpc.model.Model(model_type)

xvar = model.set_variable(var_type='_x', var_name='xvar', shape=(3, 1))
uvar = model.set_variable(var_type='_u', var_name='uvar', shape=(1, 1))
zvar = vertcat(xvar, uvar)

xvar_next = vertcat(
    xvar[0] + Delta * 0 + cal_kstar(zvar, SX(ZT), alpha0,
                                    Lambda0).T @ SX(np.linalg.inv(K0 + np.identity(K0.shape[0]) * a_max)) @ SX(Y0),
    xvar[1] + Delta * uvar + cal_kstar(zvar, SX(ZT), alpha1,
                                        Lambda0).T @ SX(np.linalg.inv(K1)) @ SX(Y1),
    xvar[2] + Delta * (xvar[0] - xvar[1]) + cal_kstar(zvar, SX(ZT),
                                                      alpha2, Lambda2).T @ SX(np.linalg.inv(K2)) @ SX(Y2)
)
model.set_rhs(var_name='xvar', expr=xvar_next)

xref = SX([22, 22, 50])
uref = SX([0])
weightx = diag(SX([1, 1, 1]))
weightu = diag(SX([1]))
costfunc = (xvar - xref).T @ weightx @ (xvar - xref)
model.set_expression(expr_name='costfunc', expr=costfunc)
model.setup()

mpc = do_mpc.controller.MPC(model)

horizon = 7
setup_mpc = {
    'n_robust': 0,
    'n_horizon': horizon,
    't_step': Delta,
    'state_discretization': 'discrete',
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

lterm = model.aux['costfunc']
mterm = model.aux['costfunc']
mpc.set_objective(lterm=lterm, mterm=mterm)
mpc.set_rterm(uvar=1)

xlower_bound = np.array([[15], [15], [30]])
xupper_bound = np.array([[25], [25], [100]])

mpc.bounds['lower', '_x', 'xvar'] = xlower_bound
mpc.bounds['upper', '_x', 'xvar'] = xupper_bound
mpc.bounds['lower', '_u', 'uvar'] = -1
mpc.bounds['upper', '_u', 'uvar'] = 1
mpc.terminal_bounds['lower', '_x', 'xvar'] = np.array([[17], [18], [50]])
mpc.terminal_bounds['upper', '_x', 'xvar'] = np.array([[22], [22], [80]])
mpc.setup()

estimator = do_mpc.estimator.StateFeedback(model)
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=0.1)
simulator.setup()

x0 = np.array([[20], [20], [80]])
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0
mpc.set_initial_guess()

u0 = mpc.make_step(x0)

########################################################################
# x_test = torch.from_numpy(np.array(mpc.opt_x_num['_x', 6, 0, 0]).reshape(-1))
# u_test = torch.from_numpy(
#     np.array(mpc.opt_x_num['_u', 6, 0]).reshape(-1))
# z_test = torch.cat([x_test, u_test], dim=0).reshape(1, -1).float()

# models.eval()
# likelihoods.eval()
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     predictions = likelihoods(*models(z_test, z_test, z_test))
# sigma0 = predictions[0].variance
# sigma1 = predictions[1].variance
# sigma2 = predictions[2].variance

# C0 = torch.sqrt(2 * torch.log((2 * (alpha0x**2)) /
#                               (2 * (alpha0x**2) - (gamma0**2))))
# C1 = torch.sqrt(2 * torch.log((2 * (alpha1x**2)) /
#                               (2 * (alpha1x**2) - (gamma1**2))))
# C2 = torch.sqrt(2 * torch.log((2 * (alpha2x**2)) /
#                               (2 * (alpha2x**2) - (gamma2**2))))

# DELTA = np.array([(beta0 * sigma0).to('cpu').detach().numpy(),
#                   (beta1 * sigma1).to('cpu').detach().numpy(),
#                   (beta2 * sigma2).to('cpu').detach().numpy()]).reshape(-1)
# ALPHA = np.array([alpha0x.to('cpu').detach().numpy(),
#                   alpha1x.to('cpu').detach().numpy(),
#                   alpha2x.to('cpu').detach().numpy()]).reshape(-1)
# psi = cp.Variable(3, pos=True)
# constranit = [cp.quad_form(psi + DELTA, np.linalg.inv(Lambda0x.to(
#     'cpu').detach().numpy())) <= (C0.to('cpu').detach().numpy())**2,
#     psi + 1e-12 <= ALPHA]

# trigger_func = cp.geo_mean(psi)
# prob_trigger = cp.Problem(cp.Maximize(trigger_func), constranit)
# prob_trigger.solve(solver=cp.MOSEK)

# print("Status:", prob_trigger.status)
# print(psi.value)
# print(ALPHA)
# print(ALPHA - psi.value)

gamma0 = (1.4142 * alpha0x - 2 * epsilon0) / 5
gamma1 = (1.4142 * alpha1x - 2 * epsilon1) / 5
gamma2 = (1.4142 * alpha2x - 2 * epsilon2) / 5

models.eval()
likelihoods.eval()
for i in reversed(range(horizon)):
    C0 = torch.sqrt(2 * torch.log((2 * (alpha0x**2)) /
                                  (2 * (alpha0x**2) - (gamma0**2))))
    C1 = torch.sqrt(2 * torch.log((2 * (alpha1x**2)) /
                                  (2 * (alpha1x**2) - (gamma1**2))))
    C2 = torch.sqrt(2 * torch.log((2 * (alpha2x**2)) /
                                  (2 * (alpha2x**2) - (gamma2**2))))
    x_test = torch.from_numpy(
        np.array(mpc.opt_x_num['_x', i, 0, 0]).reshape(-1))
    u_test = torch.from_numpy(
        np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1))
    z_test = torch.cat([x_test, u_test], dim=0).reshape(1, -1).float()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihoods(*models(z_test, z_test, z_test))
    sigma0 = torch.sqrt(predictions[0].variance) + a_max
    sigma1 = torch.sqrt(predictions[1].variance) + a_max
    sigma2 = torch.sqrt(predictions[2].variance) + a_max

    DELTA = np.array([(beta0 * sigma0).to('cpu').detach().numpy(),
                      (beta1 * sigma1).to('cpu').detach().numpy(),
                      (beta2 * sigma2).to('cpu').detach().numpy()]).reshape(-1)
    ALPHA = np.array([alpha0x.to('cpu').detach().numpy(),
                      alpha1x.to('cpu').detach().numpy(),
                      alpha2x.to('cpu').detach().numpy()]).reshape(-1)
    psi = cp.Variable(3, pos=True)
    constranit = [cp.quad_form(psi + DELTA, np.linalg.inv(Lambda0x.to(
        'cpu').detach().numpy())) <= (C0.to('cpu').detach().numpy())**2,
        cp.quad_form(psi + DELTA, np.linalg.inv(Lambda1x.to(
            'cpu').detach().numpy())) <= (C1.to('cpu').detach().numpy())**2,
        cp.quad_form(psi + DELTA, np.linalg.inv(Lambda2x.to(
            'cpu').detach().numpy())) <= (C2.to('cpu').detach().numpy())**2,
        psi <= 1.4 * ALPHA]

    trigger_func = cp.geo_mean(psi)
    prob_trigger = cp.Problem(cp.Maximize(trigger_func), constranit)
    prob_trigger.solve(solver=cp.MOSEK)
    print(psi.value)

    gamma0 = torch.from_numpy(np.array([psi.value[0]]))
    gamma1 = torch.from_numpy(np.array([psi.value[1]]))
    gamma2 = torch.from_numpy(np.array([psi.value[2]]))
