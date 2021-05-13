import torch
import numpy as np
import gpytorch
import safetygame
torch.manual_seed(1)
np.random.seed(0)


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


def make_data(vehicle, xinits, data_num):
    z_train = torch.zeros(1, 5)
    y_train = torch.zeros(1, 3)
    for i in range(data_num):
        if i % 10 == 0:
            j = i // 10
            x = xinits[j, :]
        if torch.rand(1) > 0.9:
            v = v_max * torch.rand(1)
            omega = omega_max * torch.rand(1)
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
           [i].covariance_matrix for i in range(3)]
    noises = [gpmodels.models[i].likelihood.noise for i in range(3)]

    gpmodels.eval()
    likelihoods.eval()

    return gpmodels, likelihoods, cov, noises


class safetyGame:
    def __init__(self, vehicle, gpmodels, likelihoods, cov, b, data_num, etax, Xsafe, Uq, gamma_param, noises):
        self.vehicle = vehicle
        self.gpmodels = gpmodels
        self.likelihoods = likelihoods
        self.cov = cov
        self.b = b
        self.data_num = data_num
        self.etax = etax
        self.Xsafe = Xsafe
        self.Uq = Uq
        self.gamma_param = gamma_param
        self.noises = noises

        self.etax_v = torch.tensor([self.etax, self.etax, self.etax])
        self.alpha = [torch.sqrt(
            self.gpmodels.models[i].covar_module.outputscale) for i in range(3)]
        self.Lambdax = [torch.diag(
            self.gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(-1)[:3]) ** 2 for i in range(3)]
        self.beta = torch.tensor([self.set_beta(b[i], self.gpmodels.train_targets[i], cov[i], self.noises[i])
                                  for i in range(3)])
        self.epsilon = torch.tensor([self.set_epsilon(self.alpha[i], self.Lambdax[i])
                                     for i in range(3)])
        self.gamma = torch.tensor(
            [(1.41421356 * self.alpha[i] - self.epsilon[i]) / self.gamma_param[i] for i in range(3)])
        self.cout = torch.tensor([self.set_c(self.alpha[i], self.epsilon[i])
                                  for i in range(3)])
        self.cin = torch.tensor([self.set_c(self.alpha[i], self.epsilon[i] + self.gamma[i])
                                 for i in range(3)])
        self.ellout = torch.cat(
            [torch.diag(self.cout[i] * torch.sqrt(self.Lambdax[i])).reshape(1, -1) for i in range(3)], dim=0)
        self.ellin = torch.cat(
            [torch.diag(self.cin[i] * torch.sqrt(self.Lambdax[i])).reshape(1, -1) for i in range(3)], dim=0)
        self.ellout_max = torch.tensor(
            [self.ellout[:, i].max() for i in range(3)])
        self.ellin_max = torch.tensor(
            [self.ellin[:, i].max() for i in range(3)])

    def set_beta(self, b, y, cov, noise):
        return torch.sqrt(b ** 2 - y @ torch.inverse(cov + torch.eye(cov.shape[0]) * noise) @ y + cov.shape[0])

    def set_epsilon(self, alpha, Lambdax):
        return torch.sqrt(2 * (alpha**2) * (1 - torch.exp(-0.5 * self.etax_v @ torch.inverse(Lambdax) @ self.etax_v)))

    def set_c(self, alpha, epsilon):
        return torch.sqrt(2 * torch.log((2 * (alpha**2)) / (2 * (alpha**2) - (epsilon**2))))

    def min_max_check(self, x, xlist, dim):
        return torch.all(x + self.ellout[:, dim] <= torch.max(xlist)) and torch.all(torch.min(xlist) <= x - self.ellout[:, dim])

    def operation(self):
        X0 = torch.arange(self.Xsafe[0, 0],
                          self.Xsafe[0, 1] + 0.000001, self.etax)
        X1 = torch.arange(self.Xsafe[1, 0],
                          self.Xsafe[1, 1] + 0.000001, self.etax)
        X2 = torch.arange(self.Xsafe[2, 0],
                          self.Xsafe[2, 1] + 0.000001, self.etax)
        X_range_min = torch.tensor([X0.min(), X1.min(), X2.min()])
        X_range_max = torch.tensor([X0.max(), X1.max(), X2.max()])

        Q = torch.zeros([X0.shape[0], X1.shape[0], X2.shape[0]])

        Qind_init = torch.ceil(self.ellout_max / self.etax).int()
        Q[Qind_init[0]: -Qind_init[0], Qind_init[1]: -
            Qind_init[1], Qind_init[2]: -Qind_init[2]] = 1

        Qind = torch.nonzero(Q).int()
        Qflag = 1
        print('Start safety game.')
        while Qflag == 1:
            Qdata = torch.zeros([X0.shape[0], X1.shape[0], X2.shape[0]])
            Udata = torch.zeros([X0.shape[0], X1.shape[0], X2.shape[0], 2])
            Qsafe = Q.clone()
            Qsafeind = Qind.clone()
            for i in range(Qind.shape[0]):
                # print(i, Qind.shape[0], i / Qind.shape[0] * 100)
                u_flag = 1
                for j in range(self.Uq.shape[0]):
                    if j == self.Uq.shape[0] - 1:
                        u_flag = 0
                    z_test = torch.tensor(
                        [X0[Qind[i, 0]], X1[Qind[i, 1]], X2[Qind[i, 2]], self.Uq[j, 0], self.Uq[j, 1]]).reshape(1, -1)
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        predictions = self.likelihoods(
                            *self.gpmodels(z_test, z_test, z_test))
                    means = torch.tensor(
                        [predictions[l].mean for l in range(3)])
                    variances = torch.tensor(
                        [predictions[l].variance for l in range(3)])

                    if j == 1:
                        return

                    xpre_lower = torch.tensor(
                        [z_test[0, l] + means[l] - (self.b[l] * self.epsilon[l] + self.beta[l] * torch.sqrt(variances[l]) + self.etax)
                         for l in range(3)])
                    xpre_upper = torch.tensor(
                        [z_test[0, l] + means[l] + (self.b[l] * self.epsilon[l] + self.beta[l] * torch.sqrt(variances[l]) + self.etax)
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


# set param (all)
vr = 1.
omegar = 1.
ur = torch.tensor([vr, omegar])
v_max = 2
omega_max = 2
ts = 0.4
noise = 0.001
b = [1.0, 1.0, 1.0]

# set param (gp)
xinits = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
                       [0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [0., 0., 0.], [1., 1., 1.]])
data_num = 100
gpudate_num = 100
Kx = 1
Ky = 1
Ktheta = 1

# set param (safety game)
etax = 0.04
etau = 0.2
Vq = torch.arange(0., v_max + etau, etau)
Omegaq = torch.arange(0., omega_max + etau, etau)
Uq = torch.zeros(Vq.shape[0] * Omegaq.shape[0], 2)
for i in range(Vq.shape[0]):
    for j in range(Omegaq.shape[0]):
        Uq[i * Omegaq.shape[0] + j, :] = torch.tensor([Vq[i], Omegaq[j]])

print(Uq.shape)
gamma_param = [100, 100, 80]
Xsafe = torch.tensor([[-1.2, 1.2], [-1.2, 1.2], [-1.2, 1.2]])

# set param (etmpc)
mpctype = 'discrete'
weightx = np.diag([1, 1, 1])
horizon = 25
xr_init = np.array([[0., 0., 0.]])

if __name__ == '__main__':
    vehicle = VEHICLE(ts, noise, vr, omegar, Kx, Ky, Ktheta)
    z_train, y_train = make_data(vehicle, xinits, data_num)
    gpmodels, likelihoods, cov, noises = train(
        z_train, y_train, gpudate_num)

    SafetyGame = safetyGame(vehicle, gpmodels, likelihoods, cov,
                            b, data_num, etax, Xsafe, Uq, gamma_param, noises)
    # SafetyGame.operation()

    alpha = [torch.sqrt(gpmodels.models[i].covar_module.outputscale).to(
        'cpu').detach().numpy().astype(np.float64).reshape(-1, 1) for i in range(3)]

    Lambdax = [(torch.diag(gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(-1)
                           [:3]) ** 2).to('cpu').detach().numpy().astype(np.float64) for i in range(y_train.shape[1])]

    Lambda = [(torch.diag(gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(-1)) ** 2
               ).to('cpu').detach().numpy().astype(np.float64) for i in range(y_train.shape[1])]

    cov = [cov[i].to('cpu').detach().numpy().astype(np.float64)
           for i in range(3)]

    Y = [gpmodels.train_targets[i].to('cpu').detach().numpy().astype(
        np.float64).reshape(-1, 1) for i in range(3)]

    ZT = gpmodels.train_inputs[0][0].to('cpu').detach().numpy().astype(
        np.float64)

    X0 = np.arange(Xsafe[0, 0],
                   Xsafe[0, 1] + 0.000001, etax).astype(np.float64).reshape(-1, 1)
    X1 = np.arange(Xsafe[1, 0],
                   Xsafe[1, 1] + 0.000001, etax).astype(np.float64).reshape(-1, 1)
    X2 = np.arange(Xsafe[2, 0],
                   Xsafe[2, 1] + 0.000001, etax).astype(np.float64).reshape(-1, 1)

    Xlist = [X0, X1, X2]

    noises = [noises[i].to('cpu').detach().numpy().copy().astype(np.float64).reshape(-1, 1) for i in range(3)]

    safetygame.print_array(alpha, Lambda, Lambdax, cov, ZT, Y, np.array(b).astype(
        np.float64).reshape(-1, 1), Xlist, Uq.to('cpu').detach().numpy().astype(np.float64), etax, noises, noise)
