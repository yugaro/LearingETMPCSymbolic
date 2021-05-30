import numpy as np
from controller import safetygame as sg
np.random.seed(3)


class Symbolic:
    def __init__(self, args, gpmodels, y_train):
        self.gpmodels = gpmodels
        self.y_mean = np.mean(y_train, axis=0).astype(np.float64)
        self.y_std = np.std(y_train, axis=0).astype(np.float64)
        self.ZT = self.gpmodels.gpr.X_train_.astype(np.float64)
        self.Y = self.gpmodels.gpr.y_train_.astype(np.float64)
        self.covs = self.gpmodels.gpr.L_ @ self.gpmodels.gpr.L_.T.astype(
            np.float64)
        self.alpha = np.sqrt(
            np.exp(gpmodels.gpr.kernel_.theta[0])).astype(np.float64)
        self.Lambda = np.diag(
            np.exp(gpmodels.gpr.kernel_.theta[1:1 + 5]) ** 2).astype(np.float64)
        self.Lambdax = np.diag(
            np.exp(gpmodels.gpr.kernel_.theta[1:1 + 3]) ** 2).astype(np.float64)
        self.noises = np.exp(
            gpmodels.gpr.kernel_.theta[-1]).astype(np.float64)

        self.b = np.array(args.b).astype(np.float64)
        self.etax_param = args.etax_param
        self.etau = args.etau
        self.Xsafe = np.array(args.Xsafe).astype(np.float64)
        self.v_max = args.v_max
        self.omega_max = args.omega_max

        self.xqparams = np.sqrt(np.diag(self.Lambdax)) / \
            np.min(np.sqrt(np.diag(self.Lambdax)))
        self.etax_v = (self.xqparams * self.etax_param).astype(np.float64)
        self.zlattice = args.zlattice

        self.Xqlist = self.setXqlist()
        self.Uq = self.setUq()
        self.epsilon = self.setEpsilon(
            self.alpha, self.Lambdax)
        self.gamma = self.setGamma(self.alpha, self.Lambdax, self.zlattice)
        self.cout = self.setC(self.alpha, self.epsilon)
        self.ellout = np.diag(self.cout * np.sqrt(self.Lambdax)).reshape(-1)

        self.cin = self.setC(self.alpha, self.gamma)
        self.ellin = np.diag(self.cin * np.sqrt(self.Lambdax)
                             ).reshape(-1).astype(np.float64)

    def setEpsilon(self, alpha, Lambdax):
        return np.sqrt(2 * (alpha**2) * (1 - np.exp(-0.5 * self.etax_v @ np.linalg.inv(Lambdax) @ self.etax_v)))

    def setGamma(self, alpha, Lambdax, zlattice):
        return np.sqrt(2 * (alpha**2) * (1 - np.exp(-2 * (zlattice ** 2) * self.etax_v @ np.linalg.inv(Lambdax) @ self.etax_v / 3)))

    def setC(self, alpha, epsilon):
        return np.sqrt(2 * np.log((2 * (alpha**2)) / (2 * (alpha**2) - (epsilon**2))))

    def setXqlist(self):
        for i in range(3):
            self.Xsafe[i, :] = self.Xsafe[i, :] * self.xqparams[i]
        return [np.arange(self.Xsafe[i, 0],
                          self.Xsafe[i, 1] + 0.000001, 2 / np.sqrt(3) * self.etax_v[i]).astype(np.float64) for i in range(3)]

    def setUq(self):
        Vq = np.arange(0., self.etau + 0.000001, self.etau)
        Omegaq = np.arange(0, self.omega_max +
                           self.etau + 0.000001, self.etau)
        Uq = np.zeros((Vq.shape[0] * Omegaq.shape[0], 2))
        for i in range(Vq.shape[0]):
            for j in range(Omegaq.shape[0]):
                Uq[i * Omegaq.shape[0] + j, :] = np.array([Vq[i], Omegaq[j]])
        return Uq

    def setQind_init(self):
        Qinit = np.zeros([self.Xqlist[0].shape[0],
                          self.Xqlist[1].shape[0], self.Xqlist[2].shape[0]]).astype(np.int)
        Qind_out = np.ceil(self.ellout / (2 / np.sqrt(3) *
                                          self.etax_v)).astype(np.int)
        Qinit[Qind_out[0]: -Qind_out[0] + 1,
              Qind_out[1]: -Qind_out[1] + 1,
              Qind_out[2]: -Qind_out[2] + 1] = 1
        Qind_init_list = np.nonzero(Qinit)
        return Qinit.tolist(), np.concatenate(
            [Qind_init_list[0].reshape(-1, 1), Qind_init_list[1].reshape(-1, 1), Qind_init_list[2].reshape(-1, 1)], axis=1).astype(np.float64)

    def safeyGame(self):
        Qinit, Qind_init = self.setQind_init()
        sgflag = 1
        while sgflag == 1:
            Q = sg.operation(Qinit, Qind_init, self.alpha, self.Lambda, self.Lambdax, self.covs,
                             self.noises, self.ZT, self.Y, self.b, self.Xqlist,
                             self.Uq, self.etax_v, self.epsilon, self.gamma,
                             self.y_mean, self.y_std, self.ellin)
            Qindlist = np.nonzero(np.array(Q))
            Qind = np.concatenate([Qindlist[0].reshape(-1, 1), Qindlist[1].reshape(-1, 1),
                                   Qindlist[2].reshape(-1, 1)], axis=1)
            if Qind_init.shape[0] == Qind.shape[0]:
                sgflag = 0
                print('complete.')
                return Q, Qind
            else:
                Qinit = Q
                Qind_init = Qind.astype(np.float64)
                print('continue..')
