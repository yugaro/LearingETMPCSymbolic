import torch
import numpy as np
import gpytorch
import do_mpc
from casadi import vertcat, SX
import matplotlib.pyplot as plt
import cvxpy as cp
torch.manual_seed(1)
np.random.seed(3)


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
            (k1 + 2 * k2 + 2 * k3 + k4)

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

    return z_train[1:], y_train[1:]


def train(z_train, y_train, gpudate_num):
    likelihood_list = [gpytorch.likelihoods.GaussianLikelihood()
                       for i in range(y_train.shape[1])]
    model_list = [ExactGPModel(
        z_train, y_train[:, i], likelihood_list[i]) for i in range(y_train.shape[1])]
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
            [2.])) * torch.tensor(self.alpha) - self.epsilon) / self.gamma_param
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

    def set_beta(self, b, y, cov):
        return torch.sqrt(b ** 2 - y @ torch.inverse(cov + torch.eye(cov.shape[0]) * (self.noise ** 2)) @ y + cov.shape[0])

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
        return torch.sqrt(b ** 2 - y @ torch.inverse(cov + (self.noise ** 2) * torch.eye(cov.shape[0])) @ y + cov.shape[0])

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
        ulist = np.zeros((1, 2))
        for i in range(self.horizon):
            ulist = np.concatenate(
                [ulist, np.array(mpc.opt_x_num['_u', i, 0]).reshape(1, -1)])
        return solver_stats['success'], torch.from_numpy(ulist[1:, :])

    def cF(self, pg):
        c = [np.sqrt(2 * np.log((2 * (self.alpha[i]**2)) /
                                (2 * (self.alpha[i]**2) - (pg[i]**2)))) for i in range(3)]
        return c

    def triggerValue(self, mpc):
        trigger_values = self.gamma.reshape(1, -1)
        for i in reversed(range(self.horizon)):
            xsuc = torch.from_numpy(
                np.array(mpc.opt_x_num['_x', i, 0, 0]).reshape(-1))
            usuc = torch.from_numpy(
                np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1))
            zsuc = torch.cat([xsuc, usuc], dim=0).reshape(1, -1).float()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = self.likelihoods(
                    *self.gpmodels(zsuc, zsuc, zsuc))
            varsuc = torch.tensor([predictions[j].variance for j in range(3)])

            psi = cp.Variable(3, pos=True)
            if i == self.horizon - 1:
                pg = self.gamma
            c = self.cF(pg)
            constranits = [cp.quad_form(cp.multiply(self.b, psi) + cp.multiply(self.beta, torch.sqrt(
                varsuc)) + self.noise * np.ones(3), np.linalg.inv(self.Lambdax[j])) <= (c[j] ** 2) for j in range(3)]
            constranits += [psi[j] <= 1.4 * self.alpha[j] for j in range(3)]

            trigger_func = cp.geo_mean(psi)
            prob_trigger = cp.Problem(cp.Maximize(trigger_func), constranits)
            prob_trigger.solve(solver=cp.MOSEK)
            if prob_trigger.status == 'infeasible':
                return prob_trigger.status, 0
            pg = psi.value
            trigger_values = np.concatenate(
                [trigger_values, psi.value.reshape(1, -1)], axis=0)
        return prob_trigger.status, np.flip(trigger_values)

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

    def learnD(self, xe, u, xe_next, ze_train, ye_train):
        ze = torch.cat([xe, u], dim=0).reshape(1, -1).float()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihoods(
                *self.gpmodels(ze, ze, ze))
        vare = torch.tensor([predictions[j].variance for j in range(3)])

        if torch.any(vare >= 0.05):
            ze_train = torch.cat([ze_train, ze], dim=0)
            ye_train = torch.cat(
                [ye_train, (xe_next - xe).reshape(1, -1)], dim=0)
        return ze_train, ye_train

    def dataCat(self, ze_train, ye_train):
        z_train_sum = self.gpmodels.train_inputs[0][0]
        y_train_sum = torch.cat([self.gpmodels.train_targets[0].reshape(-1, 1),
                                 self.gpmodels.train_targets[1].reshape(-1, 1), self.gpmodels.train_targets[2].reshape(-1, 1)], dim=1)
        z_train_sum = torch.cat([z_train_sum, ze_train], dim=0)
        y_train_sum = torch.cat([y_train_sum, ye_train], dim=0)

        return z_train_sum, y_train_sum


# set param
xinit = torch.tensor([1., 1., 1.])
vr = 2.
omegar = 1.
ur = torch.tensor([vr, omegar])
v_max = 2
omega_max = 2
ts = 0.1
noise = 0.001
data_num = 80
Kx = 1
Ky = 1
Ktheta = 1
gpudate_num = 100
Xsafe = torch.tensor([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
b = [1.05, 1.05, 1.05]

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
gamma_param = 20

# set param
mpctype = 'discrete'
weightx = np.diag([1, 1, 1])
horizon = 25
xr_init = np.array([[0., 0., 0.]])

if __name__ == '__main__':
    vehicle = VEHICLE(ts, noise, vr, omegar, Kx, Ky, Ktheta)
    z_train, y_train = make_data(vehicle, xinit, data_num, v_max, omega_max)

    flag_etmpc = 0
    flag_trigger = 0
    iteration = 0
    pathe = [torch.zeros(1, 3) for i in range(100)]
    pathc = [torch.zeros(1, 3) for i in range(100)]
    trigger_list = [torch.zeros(1) for i in range(100)]

    while flag_etmpc == 0:
        print('Iteration:', iteration)
        gpmodels, likelihoods, cov = train(
            z_train, y_train, gpudate_num)

        safetygame = safetyGame(vehicle, gpmodels, likelihoods, cov,
                                b, data_num, noise, etax, Xsafe, Uq, gamma_param)
        gamma = safetygame.gamma

        etmpc = ETMPC(vehicle, gpmodels, likelihoods, mpctype, noise, gamma,
                      horizon, weightx, ts, v_max, omega_max, cov, b, data_num)
        mpc, simulator, estimator = etmpc.setup()

        flag_init = 0
        flag_asm = 0
        while flag_init == 0:
            ze_train = torch.zeros(1, 5)
            ye_train = torch.zeros(1, 3)
            x0 = np.random.rand(3, 1) + 2
            while 1:
                etmpc.set_initial(mpc, simulator, estimator, x0)
                mpc_status, ulist = etmpc.operation(
                    mpc, simulator, estimator, x0)
                trigger_status, trigger_values = etmpc.triggerValue(mpc)

                if mpc_status and trigger_status == 'optimal':
                    print('mpc status:', mpc_status)
                    print('trigger status:', trigger_status)
                    flag_asm = 1
                    ulist_pre = ulist
                    for j in range(horizon + 1):
                        if j == 0:
                            xe = torch.from_numpy(x0).reshape(-1)
                            xr = torch.from_numpy(xr_init).reshape(-1)
                            pathe[iteration] = torch.cat(
                                [pathe[iteration], xe.reshape(1, -1)])
                            pathc[iteration] = torch.cat(
                                [pathc[iteration], (xr - xe).reshape(1, -1)])
                        xstar = np.array(
                            mpc.opt_x_num['_x', j, 0, 0]).reshape(-1)

                        if etmpc.triggerF(xstar, xe, trigger_values[j]):
                            if j == horizon:
                                flag_trigger = 1
                                break
                            u = torch.from_numpy(
                                np.array(mpc.opt_x_num['_u', j, 0])).reshape(-1)

                            xe_next = vehicle.errRK4(xe, u)
                            xr_next = vehicle.realRK4(xr, ur)

                            pathe[iteration] = torch.cat(
                                [pathe[iteration], xe_next.reshape(1, -1)])
                            pathc[iteration] = torch.cat(
                                [pathc[iteration], (xr_next - xe_next).reshape(1, -1)])

                            ze_train, ye_train = etmpc.learnD(
                                xe, u, xe_next, ze_train, ye_train)

                            xe = xe_next
                            xr = xr_next
                        else:
                            trigger_time = j
                            x0 = xe.to('cpu').detach().numpy().reshape(-1, 1)
                            print('trigger:', trigger_time)
                            trigger_list[iteration] = torch.cat(
                                [trigger_list[iteration], torch.tensor([trigger_time])])
                            break
                    if flag_trigger == 1:
                        flag_init = 1
                        flag_etmpc = 1
                        print('Event-triggered mpc was completed')
                        break
                else:
                    if flag_asm == 0:
                        print('mpc status:', mpc_status)
                        print('trigger status:', trigger_status)
                        print('Assumption is not hold.')
                        break
                    elif flag_asm == 1:
                        flag_init = 1
                        for j in range(horizon - trigger_time):
                            if j == 0:
                                xe = torch.from_numpy(x0).reshape(-1)
                                pathe[iteration] = torch.cat(
                                    [pathe[iteration], xe.reshape(1, -1)])
                                pathc[iteration] = torch.cat(
                                    [pathc[iteration], (xr - xe).reshape(1, -1)])
                            xe_next = vehicle.errRK4(
                                xe, ulist_pre[trigger_time + j])
                            xr_next = vehicle.realRK4(xr, ur)

                            pathe[iteration] = torch.cat(
                                [pathe[iteration], xe_next.reshape(1, -1)])
                            pathc[iteration] = torch.cat(
                                [pathc[iteration], (xr_next - xe_next).reshape(1, -1)])

                            ze_train, ye_train = etmpc.learnD(
                                xe, ulist_pre[trigger_time + j], xe_next, ze_train, ye_train)
                            xe = xe_next
                        x0 = xe.to('cpu').detach().numpy().copy()
                        z_train_sum, y_train_sum = etmpc.dataCat(
                            ze_train[1:], ye_train[1:])
                        z_train = z_train_sum.float().clone()
                        y_train = y_train_sum.float().clone()
                        break
        iteration += 1

    cm = plt.cm.get_cmap('jet', iteration)
    fig, ax = plt.subplots(1, 1)
    for i in range(iteration):
        ax.scatter(pathe[i][1, 0], pathe[i][1, 1],
                   color=cm(i), marker='o', label='start')
        ax.scatter(pathe[i][-1, 0], pathe[i][-1, 1],
                   color=cm(i), marker='*', label='goal')
        ax.plot(pathe[i][1:, 0], pathe[i][1:, 1],
                color=cm(i), label='iter:{0}, len:{1}'.format(i + 1, pathe[i].shape[0]))
    ax.legend()
    fig.savefig('pathe3.pdf')

    pathr = torch.zeros(1, 3)
    for i in range(1000):
        if i == 0:
            xr = torch.from_numpy(xr_init).reshape(-1)
            pathr = torch.cat([pathr, xr.reshape(1, -1)])
        xr_next = vehicle.realRK4(xr, ur)
        pathr = torch.cat([pathr, xr.reshape(1, -1)])
        xr = xr_next

    fig, ax = plt.subplots(1, 1)
    for i in range(iteration):
        ax.scatter(pathc[i][1, 0], pathc[i][1, 1],
                   color=cm(i), marker='o', label='start')
        ax.scatter(pathc[i][-1, 0], pathc[i][-1, 1],
                   color=cm(i), marker='*', label='goal')
        ax.plot(pathc[i][1:, 0], pathc[i][1:, 1], color=cm(i),
                label='iter:{0}, len:{1}'.format(i + 1, pathc[i].shape[0] - 1))
        for j in range(trigger_list[i].shape[0] - 1):
            ax.scatter(pathc[i][trigger_list[i][j + 1].int(), 0], pathc[i][trigger_list[i][j + 1].int(), 1],
                       color=cm(i), marker='x', label='trigger:{0}'.format(trigger_list[i][j + 1].int()))
    ax.plot(pathr[1:, 0], pathr[1:, 1], color='slategrey', label='reference')
    ax.legend(bbox_to_anchor=(1, 1),
              loc='upper left', borderaxespad=0, ncol=2, fontsize=6)
    fig.tight_layout()
    fig.savefig('pathc23.pdf')
