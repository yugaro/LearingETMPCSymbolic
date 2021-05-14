import torch
import numpy as np
import gpytorch
import do_mpc
from casadi import vertcat, SX
import cvxpy as cp
torch.manual_seed(1)
np.random.seed(3)

class ETMPC:
    def __init__(self, args, gpmodels, likelihoods, covs, noises, gamma):
        self.gpmodels = gpmodels
        self.likelihoods = likelihoods
        self.covs = covs
        self.noises = noises
        self.gamma = gamma
        self.b = np.array(args.b)
        self.weightx = np.diag(args.weightx)
        self.noise = args.noise
        self.mpc_type = args.mpc_type
        self.horizon = args.horizon
        self.ts = args.ts
        self.v_max = args.v_max
        self.omega_max = args.omega_max
        self.ZT = self.gpmodels.train_inputs[0][0].to(
            'cpu').detach().numpy()
        self.Y = [self.gpmodels.train_targets[i].to(
            'cpu').detach().numpy() for i in range(3)]
        self.alpha = [np.sqrt(self.gpmodels.models[i].covar_module.outputscale.to(
            'cpu').detach().numpy()) for i in range(3)]
        self.Lambda = [np.diag(self.gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(
            -1).to('cpu').detach().numpy() ** 2) for i in range(3)]
        self.Lambdax = [np.diag(self.gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(
            -1)[:3].to('cpu').detach().numpy() ** 2) for i in range(3)]
        self.mpcmodel = do_mpc.model.Model(self.mpc_type)
        self.setup_mpc = {
            'n_robust': 0,
            'n_horizon': self.horizon,
            't_step': self.ts,
            'state_discretization': self.mpc_type,
            'store_full_solution': True,
        }
        self.beta = np.array([self.setBeta(
            self.b[i], self.Y[i], self.covs[i], self.noises[i]) for i in range(3)])

    def setBeta(self, b, y, cov, noise):
        return np.sqrt(b ** 2 - y @ np.linalg.inv(cov + noise * np.identity(cov.shape[0])) @ y + cov.shape[0])

    def kstarF(self, zvar):
        kstar = SX.zeros(3, self.ZT.shape[0])
        for i in range(3):
            for j in range(self.ZT.shape[0]):
                kstar[i, j] = (self.alpha[i] ** 2) * np.exp(-0.5 * (zvar - self.ZT[j, :]).T @
                                                            np.linalg.inv(self.Lambda[i]) @ (zvar - self.ZT[j, :]))
        return kstar

    def muF(self, zvar):
        kstar = self.kstarF(zvar)
        mu = [kstar[i, :] @ np.linalg.inv(self.covs[i] + self.noises[i] * np.identity(
            self.covs[i].shape[0])) @ self.Y[i] for i in range(3)]
        return mu

    def setUp(self):
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
        costfunc = xvar.T @ self.weightx @ xvar
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
            varsuc = torch.tensor([predictions[j].variance - self.noises[j]**2 for j in range(3)])

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

    def kernelValue(self, alpha, Lambdax, x, xprime):
        return (alpha ** 2) * np.exp(-0.5 * (x - xprime) @ np.linalg.inv(Lambdax) @ (x - xprime))

    def kernelMetric(self, xstar, xe):
        xe = xe.to('cpu').detach().numpy()
        kmd = np.array([np.sqrt(2 * (self.alpha[i] ** 2) - 2 * self.kernelValue(
            self.alpha[i], self.Lambdax[i], xstar, xe)) for i in range(3)])
        return kmd

    def triggerF(self, xstar, xe, trigger_value):
        kmd = self.kernelMetric(xstar, xe)
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
