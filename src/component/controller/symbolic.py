import numpy as np
from component.controller import safetygame as sg


class Symbolic:
    def __init__(self, args, gpmodels, covs, noises):
        self.gpmodels = gpmodels
        self.covs = covs
        self.noises = noises
        self.b = args.b
        self.etax = args.etax
        self.etau = args.etau
        self.Xsafe = args.Xsafe
        self.v_max = args.v_max
        self.omega_max = args.omega_max
        self.gamma_params = args.gamma_params
        self.etax_v = np.array([self.etax, self.etax, self.etax])
        self.ZT = gpmodels.train_inputs[0][0].to(
            'cpu').detach().numpy().astype(np.float64)
        self.Y = [gpmodels.train_targets[i].to('cpu').detach().numpy(
        ).reshape(-1, 1).astype(np.float64) for i in range(3)]
        self.alpha = np.array([np.sqrt(gpmodels.models[i].covar_module.outputscale.to(
            'cpu').detach().numpy()).astype(np.float64) for i in range(3)])
        self.Lambda = [np.diag(gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(
            -1).to('cpu').detach().numpy() ** 2).astype(np.float64) for i in range(3)]
        self.Lambdax = [np.diag(gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(
            -1)[:3].to('cpu').detach().numpy() ** 2).astype(np.float64) for i in range(3)]
        self.Xqlist = self.setXqlist()
        self.Uq = self.setUq()
        self.epsilon = np.array([self.setEpsilon(
            self.alpha[i], self.Lambdax[i]) for i in range(3)]).astype(np.float64)
        self.cout = [self.setC(self.alpha[i], self.epsilon[i])
                     for i in range(3)]
        self.ellout = np.concatenate(
            [np.diag(self.cout[i] * np.sqrt(self.Lambdax[i])).reshape(1, -1) for i in range(3)], axis=0)
        self.ellout_max = np.array([self.ellout[:, i].max() for i in range(3)])
        self.gamma = (np.sqrt(2) * self.alpha -
                      self.epsilon) / args.gamma_params

    def setEpsilon(self, alpha, Lambdax):
        return np.sqrt(2 * (alpha**2) * (1 - np.exp(-0.5 * self.etax_v @ np.linalg.inv(Lambdax) @ self.etax_v)))

    def setC(self, alpha, epsilon):
        return np.sqrt(2 * np.log((2 * (alpha**2)) / (2 * (alpha**2) - (epsilon**2))))

    def setXqlist(self):
        return [np.arange(self.Xsafe[i, 0],
                          self.Xsafe[i, 1] + 0.000001, self.etax).astype(np.float64)for i in range(3)]

    def setUq(self):
        Vq = np.arange(0., self.v_max + self.etau, self.etau)
        Omegaq = np.arange(0., self.omega_max + self.etau, self.etau)
        Uq = np.zeros((Vq.shape[0] * Omegaq.shape[0], 2))
        for i in range(Vq.shape[0]):
            for j in range(Omegaq.shape[0]):
                Uq[i * Omegaq.shape[0] + j, :] = np.array([Vq[i], Omegaq[j]])
        return Uq

    def setQind_init(self):
        Qinit = np.zeros([self.Xqlist[0].shape[0],
                          self.Xqlist[1].shape[0], self.Xqlist[2].shape[0]]).astype(np.int)
        Qind_out = np.ceil(self.ellout_max / self.etax).astype(np.int)
        Qinit[Qind_out[0]: -Qind_out[0], Qind_out[1]: -
              Qind_out[1], Qind_out[2]: -Qind_out[2]] = 1
        Qind_init_list = np.nonzero(Qinit)
        return Qinit.tolist(), np.concatenate(
            [Qind_init_list[0].reshape(-1, 1), Qind_init_list[1].reshape(-1, 1), Qind_init_list[2].reshape(-1, 1)], axis=1).astype(np.float64)

    def safeyGame(self):
        Qinit, Qind_init = self.setQind_init()
        sgflag = 1
        print(Qind_init.shape[0])
        while sgflag == 1:
            Q = sg.operation(Qinit, Qind_init, self.alpha, self.Lambda, self.Lambdax, self.covs,
                             self.noises, self.ZT, self.Y, self.b, self.Xqlist, self.Uq, self.etax, self.epsilon)
            Qindlist = np.nonzero(np.array(Q))
            Qind = np.concatenate([Qindlist[0].reshape(-1, 1), Qindlist[1].reshape(-1, 1),
                                   Qindlist[2].reshape(-1, 1)], axis=1).astype(np.float64)
            if Qind_init.shape[0] == Qind.shape[0]:
                sgflag = 0
                print('complete.')
            else:
                Qinit = Q
                Qind_init = Q
                print('continue..')

# X0 = np.arange(self.Xsafe[0, 0],
#                Xsafe[0, 1] + 0.000001, etax).astype(np.float64).reshape(-1, 1)
# X1 = np.arange(Xsafe[1, 0],
#             Xsafe[1, 1] + 0.000001, etax).astype(np.float64).reshape(-1, 1)
# X2 = np.arange(Xsafe[2, 0],
#             Xsafe[2, 1] + 0.000001, etax).astype(np.float64).reshape(-1, 1)

# self.etax_v = torch.tensor([self.etax, self.etax, self.etax])
# self.alpha = [torch.sqrt(
#     self.gpmodels.models[i].covar_module.outputscale) for i in range(3)]
# self.Lambdax = [torch.diag(
#     self.gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(-1)[:3]) ** 2 for i in range(3)]
# self.beta = torch.tensor([self.set_beta(b[i], self.gpmodels.train_targets[i], cov[i])
#                           for i in range(3)])
# self.epsilon = torch.tensor([self.set_epsilon(self.alpha[i], self.Lambdax[i])
#                              for i in range(3)])
# self.gamma = torch.tensor(
#     [(1.41421356 * self.alpha[i] - self.epsilon[i]) / self.gamma_param[i] for i in range(3)])
# self.cout = torch.tensor([self.set_c(self.alpha[i], self.epsilon[i])
#                           for i in range(3)])
# self.cin = torch.tensor([self.set_c(self.alpha[i], self.epsilon[i] + self.gamma[i])
#                          for i in range(3)])
# self.ellout = torch.cat(
#     [torch.diag(self.cout[i] * torch.sqrt(self.Lambdax[i])).reshape(1, -1) for i in range(3)], dim=0)
# self.ellin = torch.cat(
#     [torch.diag(self.cin[i] * torch.sqrt(self.Lambdax[i])).reshape(1, -1) for i in range(3)], dim=0)
# self.ellout_max = torch.tensor(
#     [self.ellout[:, i].max() for i in range(3)])
# self.ellin_max = torch.tensor(
#     [self.ellin[:, i].max() for i in range(3)])

# def set_beta(self, b, y, cov):
#     return torch.sqrt(b ** 2 - y @ torch.inverse(cov + torch.eye(cov.shape[0]) * (self.noise ** 2)) @ y + cov.shape[0])

# def set_epsilon(self, alpha, Lambdax):
#     return torch.sqrt(2 * (alpha**2) * (1 - torch.exp(-0.5 * self.etax_v @ torch.inverse(Lambdax) @ self.etax_v)))

# def set_c(self, alpha, epsilon):
#     return torch.sqrt(2 * torch.log((2 * (alpha**2)) / (2 * (alpha**2) - (epsilon**2))))

# def min_max_check(self, x, xlist, dim):
#     return torch.all(x + self.ellout[:, dim] <= torch.max(xlist)) and torch.all(torch.min(xlist) <= x - self.ellout[:, dim])
