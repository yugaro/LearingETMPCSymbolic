import torch
import numpy as np
import gpytorch
import do_mpc
from casadi import vertcat, SX
# import matplotlib.pyplot as plt
import cvxpy as cp
torch.manual_seed(1)
np.random.seed(1)


class VEHICLE:
    def __init__(self, ts, noise, vr, omegar, Kx, Ky, Ktheta):
        self.ts = ts
        self.noise = noise
        self.vr = vr
        self.omegar = omegar
        self.Kx = Kx
        self.Ky = Ky
        self.Ktheta = Ktheta

    def getrealF(self, x, u):
        f0 = torch.tensor([torch.cos(x[2]) * (u[0])])
        f1 = torch.tensor([torch.sin(x[2]) * (u[0])])
        f2 = torch.tensor([(u[1])])
        return torch.tensor([f0, f1, f2])

    def realRK4(self, x, u):
        k1 = self.getrealF(x, u)
        k2 = self.getrealF(x + self.ts / 2 * k1[2], u)
        k3 = self.getrealF(x + self.ts / 2 * k2[2], u)
        k4 = self.getrealF(x + self.ts * k3[2], u)
        x_next = x + self.ts / 6 * \
            (k1 + 2 * k2 + 2 * k3 + k4) + 2 * \
            self.noise * torch.rand(3) - self.noise

        return x_next

    def getErrF(self, x, u):
        f0 = torch.tensor([u[1] * x[1] - u[0] + self.vr * torch.cos(x[2])])
        f1 = torch.tensor([-u[1] * x[0] + self.vr * torch.sin(x[2])])
        f2 = torch.tensor([self.omegar - u[1]])
        return torch.tensor([f0, f1, f2])

    def errRK4(self, x, u):
        k1 = self.getErrF(x, u)
        k2 = self.getErrF(x + self.ts / 2 * k1[2], u)
        k3 = self.getErrF(x + self.ts / 2 * k2[2], u)
        k4 = self.getErrF(x + self.ts * k3[2], u)
        x_next = x + self.ts / 6 * \
            (k1 + 2 * k2 + 2 * k3 + k4) + 2 * \
            self.noise * torch.rand(3) - self.noise
        return x_next

    def getPIDCon(self, x):
        v = self.vr * torch.cos(x[2]) + self.Kx * x[0]
        omega = self.omegar + self.vr * \
            (self.Ky * x[1] + self.Ktheta * torch.sin(x[2]))
        return v, omega


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


def make_data(vehicle, xinit, data_num, v_max, omega_max):
    z_train = torch.zeros(1, 5)
    y_train = torch.zeros(1, 3)
    for i in range(data_num):
        if i == 0:
            x = xinit
        if i == data_num / 2:
            x = -xinit
        if torch.rand(1) > 1:
            v = v_max * torch.rand(1)
            omega = 2 * omega_max * torch.rand(1) - omega_max
        else:
            v, omega = vehicle.getPIDCon(x)
        u = torch.tensor([v, omega])
        x_next = vehicle.errRK4(x, u)

        z = torch.cat([x, u], dim=0)
        z_train = torch.cat([z_train, z.reshape(1, -1)], dim=0)
        y_train = torch.cat(
            [y_train, (x_next - x).reshape(1, -1)], dim=0)
        x = x_next.clone()

    return z_train, y_train


def train(z_train, y_train, gpudate_num):
    likelihood_list = [gpytorch.likelihoods.GaussianLikelihood()
                       for i in range(y_train.shape[1])]
    model_list = [ExactGPModel(
        z_train[1:], y_train[1:, i], likelihood_list[i]) for i in range(y_train.shape[1])]
    gpmodels = gpytorch.models.IndependentModelList(
        model_list[0], model_list[1], model_list[2])
    likelihoods = gpytorch.likelihoods.LikelihoodList(
        model_list[0].likelihood, model_list[1].likelihood, model_list[2].likelihood)
    mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihoods, gpmodels)

    gpmodels.train()
    likelihoods.train()
    optimizer = torch.optim.Adam(gpmodels.parameters(), lr=0.2)
    for i in range(gpudate_num):
        optimizer.zero_grad()
        output = gpmodels(*gpmodels.train_inputs)
        loss = -mll(output, gpmodels.train_targets)
        loss.backward()
        optimizer.step()
    cov = [gpmodels(*gpmodels.train_inputs)
           [i].covariance_matrix for i in range(y_train.shape[1])]

    gpmodels.eval()
    likelihoods.eval()

    return gpmodels, likelihoods, cov


class safetyGame:
    def __init__(self, vehicle, gpmodels, likelihoods, cov, b, data_num, noise, etax, Xsafe, Uq, gamma_param):
        self.vehicle = vehicle
        self.gpmodels = gpmodels
        self.likelihoods = likelihoods
        self.cov = cov
        self.alpha = [torch.sqrt(
            self.gpmodels.models[i].covar_module.outputscale) for i in range(y_train.shape[1])]
        self.Lambdax = [torch.diag(
            self.gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(-1)[:3]) ** 2 for i in range(y_train.shape[1])]
        self.b = b
        self.data_num = data_num
        self.noise = noise
        self.etax = etax
        self.Xsafe = Xsafe
        self.Uq = Uq
        self.gamma_param = gamma_param

        self.etax_v = torch.tensor([etax, etax, etax])

        self.beta = torch.tensor([self.set_beta(b[i], gpmodels.train_targets[i], cov[i])
                                  for i in range(3)])
        self.epsilon = torch.tensor([self.set_epsilon(self.alpha[i], self.Lambdax[i])
                                     for i in range(3)])
        self.gamma = (torch.sqrt(torch.tensor(
            [2.])) * torch.tensor(self.alpha) - self.epsilon) / gamma_param
        self.cout = torch.tensor([self.set_c(self.alpha[i], self.epsilon[i])
                                  for i in range(3)])
        self.cin = torch.tensor([self.set_c(self.alpha[i], self.epsilon[i] + self.gamma[i])
                                 for i in range(3)])
        self.ellout = torch.cat(
            [torch.diag(self.cout[i] * torch.sqrt(self.Lambdax[i])).reshape(1, -1) for i in range(3)], dim=0)
        self.ellin = torch.cat(
            [torch.diag(self.cin[i] * torch.sqrt(self.Lambdax[i])).reshape(1, -1) for i in range(3)], dim=0)
        self.ellin_max = torch.tensor(
            [self.ellin[:, i].max() for i in range(3)])

    def gammaValue(self):
        return self.gamma

    def set_beta(self, b, y, cov):
        return torch.sqrt(b ** 2 - y @ torch.inverse(cov + torch.eye(self.data_num) * (self.noise ** 2)) @ y + self.data_num)

    def set_epsilon(self, alpha, Lambdax):
        return torch.sqrt(2 * (alpha**2) * (1 - torch.exp(-0.5 * self.etax_v @ torch.inverse(Lambdax) @ self.etax_v)))

    def set_c(self, alpha, epsilon):
        return torch.sqrt(2 * torch.log((2 * (alpha**2)) / (2 * (alpha**2) - (epsilon**2))))

    def min_max_check(self, x, xlist, dim):
        return torch.all(x + self.ellout[:, dim] <= torch.max(xlist)) and torch.all(torch.min(xlist) <= x - self.ellout[:, dim])

    def operation(self):
        X0 = torch.arange(self.Xsafe[0, 0],
                          self.Xsafe[0, 1] + 0.001, self.etax)
        X1 = torch.arange(self.Xsafe[1, 0],
                          self.Xsafe[1, 1] + 0.001, self.etax)
        X2 = torch.arange(self.Xsafe[2, 0],
                          self.Xsafe[2, 1] + 0.001, self.etax)
        X_range_min = torch.tensor([X0.min(), X1.min(), X2.min()])
        X_range_max = torch.tensor([X0.max(), X1.max(), X2.max()])

        Q = torch.zeros([X0.shape[0], X1.shape[0], X2.shape[0]])
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                for k in range(Q.shape[2]):
                    if self.min_max_check(X0[i], X0, dim=0) and self.min_max_check(X1[j], X1, dim=1) and self.min_max_check(X2[k], X2, dim=2):
                        Q[i][j][k] = 1
        Qind = torch.nonzero(Q).int()
        Qflag = 1
        print('Start safety game.')
        print(Qind.shape)
        while Qflag == 1:
            Qdata = torch.zeros([X0.shape[0], X1.shape[0], X2.shape[0]])
            Udata = torch.zeros([X0.shape[0], X1.shape[0], X2.shape[0], 2])
            Qsafe = Q.clone()
            Qsafeind = Qind.clone()
            for i in range(Qind.shape[0]):
                # i += 5000
                print(i)
                u_flag = 1
                for j in range(self.Uq.shape[0]):
                    if j == self.Uq.shape[0] - 1:
                        u_flag = 0
                    z_test = torch.tensor(
                        [X0[Qind[i, 0]], X1[Qind[i, 1]], X2[Qind[i, 2]], self.Uq[j, 0], self.Uq[j, 1]]).reshape(1, -1)
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        predictions = likelihoods(
                            *self.gpmodels(z_test, z_test, z_test))
                    means = torch.tensor(
                        [predictions[l].mean for l in range(3)])
                    variances = torch.tensor(
                        [predictions[l].variance for l in range(3)])
                    xpre_lower = torch.tensor(
                        [z_test[0, l] + means[l] - (self.b[l] * self.epsilon[l] + self.beta[l] * torch.sqrt(variances[l]) + self.noise + self.etax)
                         for l in range(3)])
                    xpre_upper = torch.tensor(
                        [z_test[0, l] + means[l] + (self.b[l] * self.epsilon[l] + self.beta[l] * torch.sqrt(variances[l]) + self.noise + self.etax)
                         for l in range(3)])

                    Qind_lower = torch.ceil(
                        (xpre_lower - X_range_min) / self.etax).int()
                    Qind_upper = ((xpre_upper - X_range_min) //
                                  self.etax).int()

                    if torch.all(X_range_min <= xpre_lower) and torch.all(xpre_upper <= X_range_max):
                        if torch.all(Qsafe[Qind_lower[0]:Qind_upper[0] + 1, Qind_lower[1]:Qind_upper[1] + 1, Qind_lower[2]:Qind_upper[2] + 1] == 1):
                            Qdataind = torch.ceil(
                                (means - X_range_min) / self.etax).int()
                            Qdata[Qdataind[0], Qdataind[1], Qdataind[2]] = 1
                            Udata[Qdataind[0], Qdataind[1],
                                  Qdataind[2], 0] = self.Uq[j, 0]
                            Udata[Qdataind[0], Qdataind[1],
                                  Qdataind[2], 1] = self.Uq[j, 1]
                            print(Qind[i, 0].item(),
                                  Qind[i, 1].item(), Qind[i, 2].item())
                            break
                        else:
                            if u_flag == 1:
                                continue
                            elif u_flag == 0:
                                Q[Qind[i, 0], Qind[i, 1], Qind[i, 2]] = 0
                    else:
                        if u_flag == 1:
                            continue
                        elif u_flag == 0:
                            Q[Qind[i, 0], Qind[i, 1], Qind[i, 2]] = 0
            Qind = torch.nonzero(Q).clone()
            if Qsafeind.shape[0] == Qind.shape[0]:
                Qflag = 0
                print('Safety game was completed.')
                print(Qsafeind.shape[0])
                print(self.gamma_param)
                return Qsafe, Qdata, Udata
                break
            else:
                print(Qsafeind.shape[0])
                print(Qind.shape[0])
                print('Continue...')
                continue


class ETMPC:
    def __init__(self, vehicle, gpmodels, likelihoods, mpctype, noise, gamma, horizon, weightx, ts, v_max, omega_max, cov, b, data_num):
        self.vehicle = vehicle
        self.gpmodels = gpmodels
        self.likelihoods = likelihoods
        self.mpctype = mpctype
        self.noise = noise
        self.gamma = gamma.to('cpu').detach().numpy()
        self.horizon = horizon
        self.weightx = weightx
        self.ts = ts
        self.v_max = v_max
        self.omega_max = omega_max
        self.cov = [cov[i].to('cpu').detach().numpy() for i in range(3)]
        self.alpha = [torch.sqrt(
            self.gpmodels.models[i].covar_module.outputscale).to('cpu').detach().numpy() for i in range(3)]
        self.Lambda = [(torch.diag(
            self.gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(-1)) ** 2).to('cpu').detach().numpy() for i in range(3)]
        self.Lambdax = [(torch.diag(
            self.gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(-1)[:3]) ** 2).to('cpu').detach().numpy() for i in range(3)]
        self.ZT = self.gpmodels.train_inputs[0][0].to('cpu').detach().numpy()
        self.Y = [self.gpmodels.train_targets[i].to(
            'cpu').detach().numpy() for i in range(3)]
        self.mpcmodel = do_mpc.model.Model(self.mpctype)
        self.setup_mpc = {
            'n_robust': 0,
            'n_horizon': self.horizon,
            't_step': self.ts,
            'state_discretization': self.mpctype,
            'store_full_solution': True,
        }
        self.b = b
        self.data_num = data_num
        self.beta = torch.tensor([self.set_beta(
            self.b[i], self.gpmodels.train_targets[i], cov[i]) for i in range(3)])

    def set_beta(self, b, y, cov):
        return torch.sqrt(b ** 2 - y @ torch.inverse(cov + (self.noise ** 2) * torch.eye(self.data_num)) @ y + self.data_num)

    def kstarF(self, zvar):
        kstar = SX.zeros(3, self.ZT.shape[0])
        for i in range(3):
            for j in range(self.ZT.shape[0]):
                kstar[i, j] = (self.alpha[i] ** 2) * np.exp(-0.5 * (zvar - self.ZT[j, :]).T @
                                                            np.linalg.inv(self.Lambda[i]) @ (zvar - self.ZT[j, :]))
        return kstar

    def muF(self, zvar):
        kstar = self.kstarF(zvar)
        mu = [kstar[i, :] @ np.linalg.inv(self.cov[i] + (self.noise ** 2) * np.identity(
            self.cov[i].shape[0])) @ self.Y[i] for i in range(3)]
        return mu

    def setup(self):
        # set variable
        xvar = self.mpcmodel.set_variable(
            var_type='_x', var_name='xvar', shape=(3, 1))
        uvar = self.mpcmodel.set_variable(
            var_type='_u', var_name='uvar', shape=(2, 1))
        zvar = vertcat(xvar, uvar)
        mu = self.muF(zvar)
        xvar_next = vertcat(xvar[0] + mu[0], xvar[1] + mu[1], xvar[2] + mu[2])
        self.mpcmodel.set_rhs(var_name='xvar', expr=xvar_next)

        # set cost function
        costfunc = xvar.T @ weightx @ xvar
        self.mpcmodel.set_expression(expr_name='costfunc', expr=costfunc)
        self.mpcmodel.setup()
        lterm = self.mpcmodel.aux['costfunc']
        mterm = self.mpcmodel.aux['costfunc']

        # set mpc
        mpc = do_mpc.controller.MPC(self.mpcmodel)
        mpc.set_param(**self.setup_mpc)
        mpc.set_objective(lterm=lterm, mterm=mterm)
        mpc.set_rterm(uvar=1)
        mpc.bounds['lower', '_u', 'uvar'] = - \
            np.array([[self.v_max], [self.omega_max]])
        mpc.bounds['upper', '_u', 'uvar'] = np.array(
            [[self.v_max], [self.omega_max]])
        mpc.terminal_bounds['lower', '_x', 'xvar'] = - \
            np.array([[0.1], [0.1], [0.1]])
        mpc.terminal_bounds['upper', '_x', 'xvar'] = np.array(
            [[0.1], [0.1], [0.1]])
        mpc.setup()

        # set simulator and estimator
        estimator = do_mpc.estimator.StateFeedback(self.mpcmodel)
        simulator = do_mpc.simulator.Simulator(self.mpcmodel)
        simulator.set_param(t_step=self.ts)
        simulator.setup()

        return mpc, simulator, estimator

    def set_initial(self, mpc, simulator, estimator, x0):
        mpc.x0 = x0
        simulator.x0 = x0
        estimator.x0 = x0
        mpc.set_initial_guess()

    def operation(self, mpc, simulator, estimator, x0):
        u0, solver_stats = mpc.make_step(x0)
        print(solver_stats['success'])
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

    def cF(self, pg):
        c = [np.sqrt(2 * np.log((2 * (self.alpha[i]**2)) /
                                (2 * (self.alpha[i]**2) - (pg[i]**2)))) for i in range(3)]
        return c

    def triggerValue(self, mpc):
        for i in reversed(range(self.horizon)):
            xsuc = torch.from_numpy(
                np.array(mpc.opt_x_num['_x', i, 0, 0]).reshape(-1))
            usuc = torch.from_numpy(
                np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1))
            zsuc = torch.cat([xsuc, usuc], dim=0).reshape(1, -1).float()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = likelihoods(*gpmodels(zsuc, zsuc, zsuc))
            varsuc = torch.tensor([predictions[j].variance for j in range(3)])

            psi = cp.Variable(3, pos=True)
            if i == self.horizon - 1:
                pg = self.gamma
                trigger_values = self.gamma.reshape(1, -1)
            c = self.cF(pg)

            constranits = [cp.quad_form(cp.multiply(self.b, psi) + cp.multiply(self.beta, torch.sqrt(
                varsuc)), np.linalg.inv(self.Lambdax[j])) <= c[j] ** 2 for j in range(3)]
            constranits += [psi[j] <= 1.4 * self.alpha[j] for j in range(3)]

            trigger_func = cp.geo_mean(psi)
            prob_trigger = cp.Problem(cp.Maximize(trigger_func), constranits)
            prob_trigger.solve(solver=cp.MOSEK)
            if prob_trigger.status == 'infeasible':
                return prob_trigger.status
            pg = psi.value
            trigger_values = np.concatenate(
                [trigger_values, psi.value.reshape(1, -1)], axis=0)
        return np.flip(trigger_values)

    def kernelValue(self, x, xprime, alpha, Lambdax):
        return (alpha ** 2) * np.exp(-0.5 * (x - xprime) @ np.linalg.inv(Lambdax) @ (x - xprime))

    def kernel_metric(self, xstar, xe):
        xe = xe.to('cpu').detach().numpy().copy()
        kmd = [np.sqrt(2 * (self.alpha[i] ** 2) - 2 * self.kernelValue(xstar,
                                                                       xe, self.alpha[i], self.Lambdax[i])) for i in range(3)]
        return kmd

    def triggerF(self, xstar, xe, trigger_value):
        kmd = self.kernel_metric(xstar, xe)
        return np.all(kmd <= trigger_value)

# set param
xinit = torch.tensor([1., 1., 1.])
vr = 2.
omegar = 1.
v_max = 2
omega_max = 2
ts = 0.1
noise = 0.001
data_num = 100
Kx = 1
Ky = 1
Ktheta = 1
gpudate_num = 100
Xsafe = torch.tensor([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
b = [1, 1, 1]

# set param safety game
etax = 0.1
etau = 0.2
Vq_upper = torch.arange(0., v_max + etau, etau)
Vq_lower = torch.arange(-v_max, 0., etau)
Omegaq_upper = torch.arange(0., omega_max + etau, etau)
Omegaq_lower = torch.arange(-omega_max, 0., etau)
Vq = torch.zeros(int(2 * omega_max / etau + 1))
Omegaq = torch.zeros(int(2 * omega_max / etau + 1))
for i in range(Omegaq.shape[0]):
    if i % 2 == 0:
        Vq[i] = Vq_upper[i // 2]
        Omegaq[i] = Omegaq_upper[i // 2]
    elif i % 2 == 1:
        Vq[i] = Vq_lower[Vq_lower.shape[0] - i // 2 - 1]
        Omegaq[i] = Omegaq_lower[Omegaq_lower.shape[0] - i // 2 - 1]
Uq = torch.zeros(Vq.shape[0] * Omegaq.shape[0], 2)
for i in range(Vq.shape[0]):
    for j in range(Omegaq.shape[0]):
        Uq[i * Omegaq.shape[0] + j, 0] = Vq[i]
        Uq[i * Omegaq.shape[0] + j, 1] = Omegaq[j]
gamma_param = 5

# set param
mpctype = 'discrete'
weightx = np.diag([1, 1, 1])
gamma = 0.01
horizon = 20
x0 = np.array([[3.], [2.], [1.]])

if __name__ == '__main__':
    # create gpmodel
    vehicle = VEHICLE(ts, noise, vr, omegar, Kx, Ky, Ktheta)
    z_train, y_train = make_data(vehicle, xinit, data_num, v_max, omega_max)
    gpmodels, likelihoods, cov = train(
        z_train, y_train, gpudate_num)

    # implement safety game
    safetygame = safetyGame(vehicle, gpmodels, likelihoods, cov,
                            b, data_num, noise, etax, Xsafe, Uq, gamma_param)
    gamma = safetygame.gammaValue()
    # Qsafe, Qdata, Udata = safetygame.operation()

    # set etmpc
    etmpc = ETMPC(vehicle, gpmodels, likelihoods, mpctype, noise, gamma,
                  horizon, weightx, ts, v_max, omega_max, cov, b, data_num)
    mpc, simulator, estimator = etmpc.setup()

    # implement mpc
    flag_mpc = 1
    while flag_mpc == 1:
        etmpc.set_initial(mpc, simulator, estimator, x0)
        etmpc.operation(mpc, simulator, estimator, x0)
        trigger_values = etmpc.triggerValue(mpc)
        if trigger_values == 'infeasible':
            break
        for j in range(horizon):
            if j == 0:
                xe = torch.from_numpy(x0).reshape(-1)
            u = torch.from_numpy(
                np.array(mpc.opt_x_num['_u', j, 0])).reshape(-1)
            xe_next = vehicle.errRK4(xe, u)

            xstar_next = np.array(mpc.opt_x_num['_x', j + 1, 0, 0]).reshape(-1)
            if j == horizon - 1:
                flag_mpc = 0
                break
            elif etmpc.triggerF(xstar_next, xe_next, trigger_values[j + 1]):
                xe = xe_next
                print('a')
            else:
                x0 = xe_next.to('cpu').detach().numpy().reshape(-1, 1)
                print('b')
                break

# fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(16, 9))
# graphics.plot_results()
# graphics.reset_axes()
# fig.savefig('mpc2.png')

# x0 = np.array([[4], [-0.3], [1]])
# mpc.x0 = x0
# simulator.x0 = x0
# estimator.x0 = x0
# mpc.set_initial_guess()
# pathr = torch.from_numpy(x0.astype(np.float64)).reshape(1, -1)
# pathc = torch.from_numpy(x0.astype(np.float64) -
#                             x0.astype(np.float64)).reshape(1, -1)
# pathn = torch.from_numpy(x0.astype(np.float64) -
#                             x0.astype(np.float64)).reshape(1, -1)
# for k in range(200):
#     if k == 0:
#         xr = torch.from_numpy(x0.astype(np.float64)).reshape(-1)
#         xe = torch.from_numpy(x0.astype(np.float64)).reshape(-1)
#     ur = torch.tensor([vr, omegar])
#     xr_next = vehicle.realRK4(xr, ur)
#     pathr = torch.cat([pathr, xr_next.reshape(1, -1)], dim=0)
#     xr = xr_next

#     u0 = mpc.make_step(x0)
#     xe_next = vehicle.errRK4(xe, torch.from_numpy(u0.reshape(-1)))
#     pathc = torch.cat([pathc, (xr_next - xe_next).reshape(1, -1)], dim=0)
#     xe = xe_next

#     y_next = simulator.make_step(u0)
#     x0 = estimator.make_step(y_next)
#     pathn = torch.cat(
#         [pathn, (xr_next - torch.from_numpy(x0.astype(np.float64)).reshape(-1)).reshape(1, -1)], dim=0)

# fig, ax = plt.subplots(1, 1)
# ax.plot(pathc[:, 0], pathc[:, 1], c='b', alpha=0.6)
# ax.plot(pathn[:, 0], pathn[:, 1], c='g', alpha=0.6)
# ax.plot(pathr[:, 0], pathr[:, 1], c='r', alpha=0.6)
# fig.savefig('paths.png')

# safetygame = safetyGame(vehicle, gpmodels, likelihoods, cov,
#                         alpha, Lambdax, b, data_num, noise, etax, Xsafe, Uq, gamma_param)
# Qsafe, Qdata, Udata = safetygame.operation()

# Qsafe_np = Qsafe.to('cpu').detach().numpy()
# Qdata_np = Qdata.to('cpu').detach().numpy()
# Udata_np = Udata.to('cpu').detach().numpy()
# np.save('Qsafe_np2', Qsafe_np)
# np.save('Qdata_np2', Qdata_np)
# np.save('Udata_np2', Udata_np)

# def K_check(z, alpha, Lambda):
#     K = torch.zeros([z.shape[0], z.shape[0]])
#     for i in range(z.shape[0]):
#         for j in range(z.shape[0]):
#             K[i][j] = (alpha**2) * torch.exp(-0.5 * (z[i] - z[j]) @ (torch.inverse(Lambda)) @ (z[i] - z[j]))
#     return K


# class OCP:
#     def __init__(self, ocpmodel, alpha, Lambda, cov, ZT, Y, noise, gamma, horizon):
#         self.ocpmodel = ocpmodel
#         self.alpha = [alpha[i].to('cpu').detach().numpy() for i in range(3)]
#         self.Lambda = [Lambda[i].to('cpu').detach().numpy() for i in range(3)]
#         self.cov = [cov[i].to('cpu').detach().numpy() for i in range(3)]
#         self.ZT = ZT[0][0].to('cpu').detach().numpy()
#         self.Y = [Y[i].to('cpu').detach().numpy() for i in range(3)]
#         self.noise = noise
#         self.gamma = gamma
#         self.horizon = horizon

#     def kstarF(self, zvar):
#         kstar = SX.zeros(3, self.ZT.shape[0])
#         for i in range(3):
#             for j in range(self.ZT.shape[0]):
#                 kstar[i, j] = (self.alpha[i] ** 2) * np.exp(-0.5 * (zvar - self.ZT[j, :]).T @
#                                                             np.linalg.inv(self.Lambda[i]) @ (zvar - self.ZT[j, :]))
#         return kstar

#     def muF(self, zvar):
#         kstar = self.kstarF(zvar)
#         mu = [kstar[i, :] @ np.linalg.inv(self.cov[i] + self.noise * np.identity(
#             self.cov[i].shape[0])) @ self.Y[i] for i in range(3)]
#         return mu

#     def cF(self, pg):
#         c = [np.sqrt(2 * np.log((2 * (self.alpha[i]**2)) /
#                                 (2 * (self.alpha[i]**2) - (pg**2)))) for i in range(3)]
#         return c

#     def trigger(self):
#         return 0

# ocpmodel_type = 'discrete'
# gamma = 0.01
# horizon = 20
# ocpmodel = do_mpc.model.Model(ocpmodel_type)
# ocp = OCP(ocpmodel, alpha, Lambda, cov,
#           gpmodels.train_inputs, gpmodels.train_targets, noise, gamma, horizon)

# xvar = ocpmodel.set_variable(var_type='_x', var_name='xvar', shape=(3, 1))
# uvar = ocpmodel.set_variable(var_type='_u', var_name='uvar', shape=(2, 1))
# zvar = vertcat(xvar, uvar)
# mu = ocp.muF(zvar)
# xvar_next = vertcat(xvar[0] + mu[0], xvar[1] + mu[1], xvar[2] + mu[2])
# ocpmodel.set_rhs(var_name='xvar', expr=xvar_next)

# weightx = np.diag([1, 1, 1])
# costfunc = xvar.T @ weightx @ xvar
# ocpmodel.set_expression(expr_name='costfunc', expr=costfunc)
# ocpmodel.setup()

# mpc = do_mpc.controller.MPC(ocpmodel)
# setup_mpc = {
#     'n_robust': 0,
#     'n_horizon': horizon,
#     't_step': ts,
#     'state_discretization': 'discrete',
#     'store_full_solution': True,
# }
# mpc.set_param(**setup_mpc)

# lterm = ocpmodel.aux['costfunc']
# mterm = ocpmodel.aux['costfunc']
# mpc.set_objective(lterm=lterm, mterm=mterm)
# mpc.set_rterm(uvar=1)

# # mpc.bounds['lower', '_x', 'xvar'] = xlower_bound
# # mpc.bounds['upper', '_x', 'xvar'] = xupper_bound
# mpc.bounds['lower', '_u', 'uvar'] = -np.array([[v_max], [omega_max]])
# mpc.bounds['upper', '_u', 'uvar'] = np.array([[v_max], [omega_max]])
# mpc.terminal_bounds['lower', '_x', 'xvar'] = - np.array([[0.1], [0.1], [0.1]])
# mpc.terminal_bounds['upper', '_x', 'xvar'] = np.array([[0.1], [0.1], [0.1]])
# mpc.setup()

# estimator = do_mpc.estimator.StateFeedback(ocpmodel)
# simulator = do_mpc.simulator.Simulator(ocpmodel)
# simulator.set_param(t_step=ts)
# simulator.setup()

# x0 = np.array([[4], [-0.3], [1]])
# mpc.x0 = x0
# simulator.x0 = x0
# estimator.x0 = x0
# mpc.set_initial_guess()
# for k in range(200):
#     u0 = mpc.make_step(x0)
#     for i in reversed(range(horizon)):
#         xsuc = torch.from_numpy(
#             np.array(mpc.opt_x_num['_x', i, 0, 0]).reshape(-1))
#         usuc = torch.from_numpy(
#             np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1))
#         zsuc = torch.cat([xsuc, usuc], dim=0).reshape(1, -1).float()
#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             predictions = likelihoods(*gpmodels(zsuc, zsuc, zsuc))
#         varsuc = [predictions[i].variance for i in range(3)]

#         psi = cp.Variable(3, pos=True)
#         constranit = [cp.quad_form(psi + DELTA, np.linalg.inv(Lambda0x.to(
#             'cpu').detach().numpy())) <= (C0.to('cpu').detach().numpy())**2,
#             cp.quad_form(psi + DELTA, np.linalg.inv(Lambda1x.to(
#                 'cpu').detach().numpy())) <= (C1.to('cpu').detach().numpy())**2,
#             cp.quad_form(psi + DELTA, np.linalg.inv(Lambda2x.to(
#                 'cpu').detach().numpy())) <= (C2.to('cpu').detach().numpy())**2,
#             psi <= 1.4 * ALPHA]
#          y_next = simulator.make_step(u0)
#          x0 = estimator.make_step(y_next)


# def plot():
#     x0 = np.array([[4], [-0.3], [1]])
#     mpc.x0 = x0
#     simulator.x0 = x0
#     estimator.x0 = x0
#     mpc.set_initial_guess()
#     pathr = torch.from_numpy(x0.astype(np.float64)).reshape(1, -1)
#     pathc = torch.from_numpy(x0.astype(np.float64) -
#                              x0.astype(np.float64)).reshape(1, -1)
#     pathn = torch.from_numpy(x0.astype(np.float64) -
#                              x0.astype(np.float64)).reshape(1, -1)
#     for k in range(200):
#         if k == 0:
#             xr = torch.from_numpy(x0.astype(np.float64)).reshape(-1)
#             xe = torch.from_numpy(x0.astype(np.float64)).reshape(-1)
#         ur = torch.tensor([vr, omegar])
#         xr_next = vehicle.realRK4(xr, ur)
#         pathr = torch.cat([pathr, xr_next.reshape(1, -1)], dim=0)
#         xr = xr_next

#         u0 = mpc.make_step(x0)
#         xe_next = vehicle.errRK4(xe, torch.from_numpy(u0.reshape(-1)))
#         pathc = torch.cat([pathc, (xr_next - xe_next).reshape(1, -1)], dim=0)
#         xe = xe_next

#         y_next = simulator.make_step(u0)
#         x0 = estimator.make_step(y_next)
#         pathn = torch.cat(
#             [pathn, (xr_next - torch.from_numpy(x0.astype(np.float64)).reshape(-1)).reshape(1, -1)], dim=0)

#     fig, ax = plt.subplots(1, 1)
#     ax.plot(pathc[:, 0], pathc[:, 1], c='b', alpha=0.6)
#     ax.plot(pathn[:, 0], pathn[:, 1], c='g', alpha=0.6)
#     ax.plot(pathr[:, 0], pathr[:, 1], c='r', alpha=0.6)
#     fig.savefig('paths.png')
