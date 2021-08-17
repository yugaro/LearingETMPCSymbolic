import numpy as np
import do_mpc
from casadi import vertcat, SX
import cvxpy as cp
np.random.seed(0)


class ETMPC:
    def __init__(self, args, gpmodels, gamma, horizon):
        self.gpmodels = gpmodels
        self.ZT = self.gpmodels.gpr.X_train_
        self.Y = self.gpmodels.gpr.y_train_
        self.covs = self.gpmodels.gpr.L_ @ self.gpmodels.gpr.L_.T
        self.alpha = np.sqrt(
            np.exp(gpmodels.gpr.kernel_.theta[0]))
        self.Lambda = np.diag(
            np.exp(gpmodels.gpr.kernel_.theta[1:1 + 5]) ** 2)
        self.Lambdax = np.diag(
            np.exp(gpmodels.gpr.kernel_.theta[1:1 + 3]) ** 2)
        self.noises = np.exp(
            gpmodels.gpr.kernel_.theta[-1])
        self.gamma = gamma
        self.b = np.array(args.b)
        self.mpc_type = args.mpc_type
        self.horizon = horizon
        self.terminalset = args.terminalset
        self.ts = args.ts
        self.mpcmodel = do_mpc.model.Model(self.mpc_type)
        self.setup_mpc = {
            'n_robust': 0,
            'n_horizon': self.horizon,
            't_step': self.ts,
            'state_discretization': self.mpc_type,
            'store_full_solution': True,
        }
        self.weightx = np.diag(args.weightx)
        self.v_max = args.v_max
        self.omega_max = args.omega_max
        self.beta = np.array([self.setBeta(
            self.b[i], self.Y[:, i], self.covs) for i in range(3)])

    def setBeta(self, b, Y, cov):
        # print(b ** 2 - Y @ np.linalg.inv(cov) @ Y + cov.shape[0])
        if b ** 2 - Y @ np.linalg.inv(cov) @ Y + cov.shape[0] < 0:
            return 1
        return np.sqrt(b ** 2 - Y @ np.linalg.inv(cov) @ Y + cov.shape[0])

    def kstarF(self, zvar):
        kstar = SX.zeros(self.ZT.shape[0])
        for i in range(self.ZT.shape[0]):
            kstar[i] = (self.alpha ** 2) * np.exp(-0.5 * (zvar - self.ZT[i, :]
                                                          ).T @ np.linalg.inv(self.Lambda) @ (zvar - self.ZT[i, :]))
        return kstar

    def muF(self, zvar):
        kstar = self.kstarF(zvar)
        mu = [kstar.T @ np.linalg.inv(self.covs) @ self.Y[:, i]
              for i in range(3)]
        return mu

    def setUp(self):
        # set variable
        xvar = self.mpcmodel.set_variable(
            var_type='_x', var_name='xvar', shape=(3, 1))
        uvar = self.mpcmodel.set_variable(
            var_type='_u', var_name='uvar', shape=(2, 1))
        zvar = vertcat(xvar, uvar)
        mu = self.muF(zvar)

        xvar_next = vertcat(xvar[0] + mu[0],
                            xvar[1] + mu[1],
                            xvar[2] + mu[2])
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
        mpc.bounds['lower', '_u', 'uvar'] = -np.array(
            [[self.v_max], [self.omega_max]])
        mpc.bounds['upper', '_u', 'uvar'] = np.array(
            [[self.v_max], [self.omega_max]])
        mpc.terminal_bounds['lower', '_x', 'xvar'] = - \
            np.array([[self.terminalset[0]], [
                     self.terminalset[1]], [self.terminalset[2]]])
        mpc.terminal_bounds['upper', '_x', 'xvar'] = np.array(
            [[self.terminalset[0]], [self.terminalset[1]], [self.terminalset[2]]])
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
        jcost = 0
        for i in range(self.horizon):
            jcost += mpc.opt_aux_num['_aux', i, 0][1]
        ulist = np.zeros((1, 2))
        for i in range(self.horizon):
            ulist = np.concatenate(
                [ulist, np.array(mpc.opt_x_num['_u', i, 0]).reshape(1, -1)])
        return solver_stats['success'], ulist[1:, :], jcost

    def cF(self, pg):
        c = [np.sqrt(2 * np.log((2 * (self.alpha**2)) /
                                (2 * (self.alpha**2) - (pg[i]**2)))) for i in range(3)]
        return c

    def xiF(self, mpc):
        xi_values = np.array(
            [self.gamma, self.gamma, self.gamma]).reshape(1, -1)
        for i in reversed(range(self.horizon)):
            if i == self.horizon - 1:
                xg = np.array([self.gamma, self.gamma, self.gamma])
            c = self.cF(xg)

            xsuc = np.array(mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)
            usuc = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
            zsuc = np.concatenate([xsuc, usuc], axis=0).reshape(1, -1)
            _, stdsuc = self.gpmodels.predict(zsuc)

            xi = cp.Variable(3, pos=True)
            constranits = [cp.quad_form(cp.multiply(
                self.b, xi) + self.beta * stdsuc + self.noises, np.linalg.inv(self.Lambdax)) <= np.min(c) / 6]
            constranits += [xi[j] + (self.beta[j] * stdsuc + self.noises) /
                            self.b[j] <= 1.4142 * self.alpha for j in range(3)]

            xi_func = cp.geo_mean(xi)
            prob_xi = cp.Problem(cp.Maximize(xi_func), constranits)
            prob_xi.solve(solver=cp.MOSEK)

            if prob_xi.status == 'infeasible':
                return prob_xi.status, 0
            xi_values = np.concatenate(
                [xi_values, xi.value.reshape(1, -1)], axis=0)

            xg = xi.value + (self.beta * stdsuc + self.noises) / self.b

        return prob_xi.status, np.flip(xi_values)

    def kernelValue(self, alpha, Lambdax, x, xprime):
        return (alpha ** 2) * np.exp(-0.5 * (x - xprime) @ np.linalg.inv(Lambdax) @ (x - xprime))

    def kernelMetric(self, xstar, xe):
        kmd = np.sqrt(2 * (self.alpha ** 2) - 2 *
                      self.kernelValue(self.alpha, self.Lambdax, xstar, xe))
        return kmd

    def triggerF(self, xstar, xe, trigger_value):
        kmd = self.kernelMetric(xstar, xe)
        return np.all(kmd <= trigger_value)

    def stdbarF(self, xi1):
        cbar = self.cF(xi1)
        stdbar = np.diag(cbar * np.sqrt(self.Lambdax)) / self.beta
        return stdbar

    def learnD(self, xe, u, xe_next, ze_train, ye_train, xi_values, lflag):
        ze = np.concatenate([xe, u], axis=0).reshape(1, -1)
        _, stdsuc = self.gpmodels.predict(ze)
        if lflag:
            stdbar = self.stdbarF(xi_values[:, 1])
            # stdbar = self.stdbarF(xi_values[:, 1]) / 1000
            # stdbar = 0.015
        else:
            stdbar = 0.015
        if stdsuc > np.min(stdbar) / 4:
            ze_train = np.concatenate([ze_train, ze], axis=0)
            ye_train = np.concatenate(
                [ye_train, (xe_next - xe).reshape(1, -1)], axis=0)
        return ze_train, ye_train

    def dataCat(self, ze_train, ye_train):
        z_train_sum = np.concatenate(
            [self.gpmodels.gpr.X_train_, ze_train], axis=0)
        y_train_sum = np.concatenate(
            [self.gpmodels.gpr.y_train_, ye_train], axis=0)
        return z_train_sum, y_train_sum

# def xiF(self, mpc, psi_values):
    #     xi_values = np.zeros((1, 3))
    #     for i in range(self.horizon):
    #         xsuc = np.array(mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)
    #         usuc = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
    #         zsuc = np.concatenate([xsuc, usuc], axis=0).reshape(1, -1)
    #         _, stdsuc = self.gpmodels.predict(zsuc)
    #         c = self.cF(psi_values[i + 1, :])

    #         xi = cp.Variable(3, pos=True)
    #         constranits = [cp.quad_form(cp.multiply(self.b, xi) + self.beta * stdsuc, np.linalg.inv(self.Lambdax)) <= np.max(c) ** 2]
    #         constranits += [xi[j] <= 1.41213 * self.alpha for j in range(3)]

    #         xi_func = cp.geo_mean(xi)
    #         prob_xi = cp.Problem(cp.Maximize(xi_func), constranits)
    #         prob_xi.solve(solver=cp.MOSEK)

    #         if prob_xi.status == 'infeasible':
    #             return prob_xi.status, 0

    #         xi_values = np.concatenate(
    #             [xi_values, xi.value.reshape(1, -1)], axis=0)
    #     return prob_xi.status, xi_values


# def psiF(self, mpc):
#       psi_values = np.array(
#            [self.gamma, self.gamma, self.gamma]).reshape(1, -1)
#        for i in reversed(range(self.horizon)):
#             if i == self.horizon - 1:
#                 pg = np.array([self.gamma, self.gamma, self.gamma])
#             c = self.cF(pg)

#             psi = cp.Variable(3, pos=True)
#             constranits = [cp.quad_form(cp.multiply(
#                 self.b, psi), np.linalg.inv(self.Lambdax)) <= np.max(c) ** 2]
#             constranits += [psi[j] <= 1.41213 * self.alpha for j in range(3)]
#             psi_func = cp.geo_mean(psi)
#             prob_psi = cp.Problem(cp.Maximize(psi_func), constranits)
#             prob_psi.solve(solver=cp.MOSEK)

#             if prob_psi.status == 'infeasible':
#                 return prob_psi.status, 0
#             pg = psi.value
#             psi_values = np.concatenate(
#                 [psi_values, psi.value.reshape(1, -1)], axis=0)
#         return prob_psi.status, np.flip(psi_values)

# def xiF2(self, mpc):
#     xi_values = np.array(
#         [self.gamma, self.gamma, self.gamma]).reshape(1, -1)
#     for i in reversed(range(self.horizon)):
#         if i == self.horizon - 1:
#             xg = np.array([self.gamma, self.gamma, self.gamma])
#         c = self.cF(xg)

#         xsuc = np.array(mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)
#         usuc = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
#         zsuc = np.concatenate([xsuc, usuc], axis=0).reshape(1, -1)
#         _, stdsuc = self.gpmodels.predict(zsuc)

#         xi = cp.Variable(3, pos=True)
#         constranits = [cp.quad_form(cp.multiply(
#             self.b, xi) + self.beta * stdsuc + self.noises, np.linalg.inv(self.Lambdax)) <= np.max(c) ** 2]
#         constranits += [xi[j] + (self.beta * stdsuc + self.noises) /
#                         self.b <= 1.4142 * self.alpha for j in range(3)]

#         xi_func = cp.geo_mean(xi)
#         prob_xi = cp.Problem(cp.Maximize(xi_func), constranits)
#         prob_xi.solve(solver=cp.MOSEK)

#         if prob_xi.status == 'infeasible':
#             return prob_xi.status, 0
#         xi_values = np.concatenate(
#             [xi_values, xi.value.reshape(1, -1)], axis=0)

#         xg = xi.value + (self.beta * stdsuc + self.noises) / self.b

#     return prob_xi.status, np.flip(xi_values)

# def xiF(self, mpc):
    #     xi_values = np.array(
    #         [self.gamma, self.gamma, self.gamma]).reshape(1, -1)
    #     for i in reversed(range(self.horizon)):
    #         if i == self.horizon - 1:
    #             xg = np.array([self.gamma, self.gamma, self.gamma])
    #         c = self.cF(xg)
    #         # print(xg)

    #         xsuc = np.array(mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)
    #         usuc = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
    #         zsuc = np.concatenate([xsuc, usuc], axis=0).reshape(1, -1)
    #         _, stdsuc = self.gpmodels.predict(zsuc)

    #         psi = cp.Variable(3, pos=True)
    #         constranits = [cp.quad_form(cp.multiply(self.b, psi), np.linalg.inv(self.Lambdax)) <= np.max(c) ** 2]
    #         constranits += [psi[j] <= 1.41421 * self.alpha for j in range(3)]
    #         constranits += [psi[j] >= (self.beta[j] * stdsuc + self.noises) / self.b[j] for j in range(3)]

    #         psi_func = cp.geo_mean(psi)
    #         prob_psi = cp.Problem(cp.Maximize(psi_func), constranits)
    #         prob_psi.solve(solver=cp.MOSEK)

    #         print(psi.value)

    #         if prob_psi.status == 'infeasible':
    #             return prob_psi.status, 0
    #         xi_values = np.concatenate(
    #             [xi_values, (psi.value - (self.beta * stdsuc + self.noises) / self.b).reshape(1, -1)], axis=0)
    #         xg = psi.value.copy()

    #     return prob_psi.status, np.flip(xi_values)
