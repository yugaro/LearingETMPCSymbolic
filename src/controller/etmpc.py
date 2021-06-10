import numpy as np
import do_mpc
from casadi import vertcat, SX
import cvxpy as cp
np.random.seed(0)


class ETMPC:
    def __init__(self, args, gpmodels, y_train, gamma):
        self.gpmodels = gpmodels
        self.y_mean = np.mean(y_train, axis=0)
        self.y_std = np.std(y_train, axis=0)
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
        self.horizon = args.horizon
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

        # print(self.ZT.shape)
        # print(self.beta)
        # print(self.y_std)

    def setBeta(self, b, Y, cov):
        # if b ** 2 - Y @ np.linalg.inv(cov) @ Y + cov.shape[0] < 0:
        # return 1
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

        xvar_next = vertcat(xvar[0] + self.y_std[0] * mu[0] + self.y_mean[0],
                            xvar[1] + self.y_std[1] * mu[1] + self.y_mean[1],
                            xvar[2] + self.y_std[2] * mu[2] + self.y_mean[2])
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
        mpc.bounds['upper', '_u', 'uvar'] = np.array(
            [[self.v_max], [self.omega_max * 2]])
        mpc.terminal_bounds['lower', '_x', 'xvar'] = - \
            np.array([[0.05], [0.05], [0.05]])
        mpc.terminal_bounds['upper', '_x', 'xvar'] = np.array(
            [[0.05], [0.05], [0.05]])
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
        return solver_stats['success'], ulist[1:, :]

    def cF(self, pg):
        c = [np.sqrt(2 * np.log((2 * (self.alpha**2)) /
                                (2 * (self.alpha**2) - (pg[i]**2)))) for i in range(3)]
        return c

    def triggerValue(self, mpc):
        trigger_values = np.array(
            [self.gamma, self.gamma, self.gamma]).reshape(1, -1)
        for i in reversed(range(self.horizon)):
            xsuc = np.array(mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)
            usuc = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
            zsuc = np.concatenate([xsuc, usuc], axis=0).reshape(1, -1)

            _, stdsuc = self.gpmodels.predict(zsuc)

            psi = cp.Variable(3, pos=True)
            if i == self.horizon - 1:
                pg = np.array([self.gamma, self.gamma, self.gamma])
            c = self.cF(pg)

            constranits = [cp.quad_form(cp.multiply(self.b * self.y_std, psi) + self.beta *
                                        self.y_std * stdsuc, np.linalg.inv(self.Lambdax)) <= np.min(c) ** 2]
            constranits += [psi[j] <= 1.41213 * self.alpha for j in range(3)]

            trigger_func = cp.geo_mean(psi)
            prob_trigger = cp.Problem(cp.Maximize(trigger_func), constranits)
            prob_trigger.solve(solver=cp.CVXOPT)

            if prob_trigger.status == 'infeasible':
                return prob_trigger.status, 0
            pg = psi.value
            trigger_values = np.concatenate(
                [trigger_values, psi.value.reshape(1, -1)], axis=0)
        return prob_trigger.status, np.flip(trigger_values)

    def triggerValue2(self, mpc):
        trigger_values = np.array(
            [self.gamma, self.gamma, self.gamma]).reshape(1, -1)
        for i in reversed(range(self.horizon)):
            psi = cp.Variable(3, pos=True)
            if i == self.horizon - 1:
                pg = np.array([self.gamma, self.gamma, self.gamma])
            c = self.cF(pg)

            constranits = [cp.quad_form(cp.multiply(
                self.b * self.y_std, psi), np.linalg.inv(self.Lambdax)) <= np.min(c) ** 2]
            constranits += [psi[j] <= 1.41213 * self.alpha for j in range(3)]

            trigger_func = cp.geo_mean(psi)
            prob_trigger = cp.Problem(cp.Maximize(trigger_func), constranits)
            prob_trigger.solve(solver=cp.CVXOPT)

            if prob_trigger.status == 'infeasible':
                return prob_trigger.status, 0
            pg = psi.value
            trigger_values = np.concatenate(
                [trigger_values, psi.value.reshape(1, -1)], axis=0)

            print(psi.value)
        print(1.41213 * self.alpha)
        return prob_trigger.status, np.flip(trigger_values)

    def kernelValue(self, alpha, Lambdax, x, xprime):
        return (alpha ** 2) * np.exp(-0.5 * (x - xprime) @ np.linalg.inv(Lambdax) @ (x - xprime))

    def kernelMetric(self, xstar, xe):
        kmd = np.sqrt(2 * (self.alpha ** 2) - 2 *
                      self.kernelValue(self.alpha, self.Lambdax, xstar, xe))
        return kmd

    def triggerF(self, xstar, xe, trigger_value):
        kmd = self.kernelMetric(xstar, xe)
        return np.all(kmd <= trigger_value)

    def learnD(self, xe, u, xe_next, ze_train, ye_train):
        ze = np.concatenate([xe, u], axis=0).reshape(1, -1)
        _, stdsuc = self.gpmodels.predict(ze)

        if np.all(stdsuc <= 0.185):
            ze_train = np.concatenate([ze_train, ze], axis=0)
            ye_train = np.concatenate(
                [ye_train, (xe_next - xe).reshape(1, -1)], axis=0)
        return ze_train, ye_train

    def dataCat(self, ze_train, ye_train):
        z_train_sum = self.gpmodels.gpr.X_train_
        y_train_sum = self.y_std * self.gpmodels.gpr.y_train_ + self.y_mean
        z_train_sum = np.concatenate([z_train_sum, ze_train], axis=0)
        y_train_sum = np.concatenate([y_train_sum, ye_train], axis=0)
        return z_train_sum, y_train_sum

# if b ** 2 - Y @ np.linalg.inv(cov) @ Y + cov.shape[0] < 0:
        # return np.sqrt(b ** 2 - Y @ np.linalg.inv(cov) @ Y + cov.shape[0])
