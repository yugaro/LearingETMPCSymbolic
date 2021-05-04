import torch
import numpy as np
from scipy import interpolate
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
torch.manual_seed(1)
np.random.seed(1)

class VEHICLE:
    def __init__(self, markers, marker_num, ts, noise):
        self.ts = ts
        self.noise = noise
        self.markers = markers
        self.tck, _ = interpolate.splprep(
            [self.markers[:, 0], self.markers[:, 1]], k=3, s=0)
        self.marker_num = marker_num

    def stateInputToVector(self, X, Y, theta, v, omega):
        x = torch.tensor([X, Y, theta])
        u = torch.tensor([v, omega])
        z = torch.tensor([X, Y, theta, v, omega])
        return x, u, z

    def getF(self, x, u):
        f0 = torch.tensor([torch.cos(x[2]) * u[0]])
        f1 = torch.tensor([torch.sin(x[2]) * u[0]])
        f2 = torch.tensor([u[1]])
        return torch.cat([f0, f1, f2], dim=0)

    def getF2(self, x, u):
        # f0 = torch.tensor([torch.cos(x[2] + 1 / 2 * self.ts * u[1]) * u[0]])
        # f1 = torch.tensor([torch.sin(x[2] + 1 / 2 * self.ts * u[1]) * u[0]])
        f0 = torch.tensor([0])
        f1 = torch.tensor([0])
        f2 = torch.tensor([u[1]])
        return torch.cat([f0, f1, f2], dim=0)

    def RK4(self, X, Y, theta, v, omega):
        x, u, z = self.stateInputToVector(X, Y, theta, v, omega)
        k1 = self.getF(x, u)
        k2 = self.getF(x + self.ts / 2 * k1[2], u)
        k3 = self.getF(x + self.ts / 2 * k2[2], u)
        k4 = self.getF(x + self.ts * k3[2], u)
        x_next = x + self.ts / 6 * \
            (k1 + 2 * k2 + 2 * k3 + k4) + 2 * \
            self.noise * torch.rand(3) - self.noise
        return x_next

    def RK1(self, X, Y, theta, v, omega):
        x, u, z = self.stateInputToVector(X, Y, theta, v, omega)
        k1 = self.getF(x, u)
        x_next = x + self.ts * k1
        return x_next

    def partF(self, X, Y, theta, v, omega):
        x, u, z = self.stateInputToVector(X, Y, theta, v, omega)
        k1 = self.getF2(x, u)
        x_next_part = x + self.ts * k1
        return x_next_part

    def getLine(self):
        arc = np.linspace(0, 1, num=self.marker_num)
        spline = interpolate.splev(arc, self.tck)
        yaw = np.zeros(self.marker_num)
        for i in range(self.marker_num - 1):
            yaw[i] = np.arctan2(spline[1][i + 1] - spline[1]
                                [i], spline[0][i + 1] - spline[0][i])
        yaw[-1] = np.arctan2(spline[1][0] - spline[1][-1],
                             spline[0][0] - spline[0][-1])
        return spline[0], spline[1], yaw


def getMap(markers, x, y):
    fig, ax = plt.subplots(1, 1)
    ax.plot(markers[:, 0], markers[:, 1], 'ro', label="controlpoint")
    ax.plot(x, y, label="splprep")
    for i in range(len(x)):
        c = patches.Rectangle(
            xy=(x[i] - 0.05, y[i] - 0.05), width=0.1, height=0.1, fc='g', ec='r')
        ax.add_patch(c)
    ax.set_title("spline")
    ax.legend(loc='lower right')
    ax.grid(which='both', color='black', linestyle='dotted')
    fig.savefig('map.png')


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


def make_data(vehicle, posx, posy, yaw, marker_num):
    z_train = torch.zeros(1, 5)
    y_train = torch.zeros(1, 3)
    for i in range(marker_num):
        for j in range(2):
            if j == 0:
                v = 0
                omega = 0
            else:
                v = v_max * torch.rand(1)
                omega = 2 * omega_max * torch.rand(1) - omega_max
            x_next_real = vehicle.RK4(posx[i], posy[i], yaw[i], v, omega)
            x_next_part = vehicle.partF(posx[i], posy[i], yaw[i], v, omega)
            z = torch.tensor([posx[i], posy[i], yaw[i], v, omega])
            z_train = torch.cat([z_train, z.reshape(1, -1)], dim=0)
            y_train = torch.cat(
                [y_train, (x_next_real - x_next_part).reshape(1, -1)], dim=0)
    return z_train, y_train

def train(z_train, y_train):
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
    for i in range(gp_updata_time):
        optimizer.zero_grad()
        output = gpmodels(*gpmodels.train_inputs)
        loss = -mll(output, gpmodels.train_targets)
        loss.backward()
        optimizer.step()
    cov = [gpmodels(*gpmodels.train_inputs)
           [i].covariance_matrix for i in range(y_train.shape[1])]

    gpmodels.eval()
    likelihoods.eval()

    alpha = [torch.sqrt(gpmodels.models[i].covar_module.outputscale)
             for i in range(y_train.shape[1])]
    Lambdax = [torch.inverse(torch.diag(
        gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(-1)[:3]) ** 2) for i in range(y_train.shape[1])]

    print(alpha)
    print(Lambdax)

    return gpmodels, likelihoods, cov, alpha, Lambdax

class safetyGame:
    def __init__(self, vehicle, gpmodels, likelihoods, cov, alpha, Lambdax, L, b, Time, noise, etax, posx, posy, yaw, marker_num):
        self.vehicle = vehicle
        self.gpmodels = gpmodels
        self.likelihoods = likelihoods
        self.cov = cov
        self.alpha = alpha
        self.Lambdax = Lambdax
        self.L = L
        self.b = b
        self.Time = Time
        self.noise = noise
        self.etax = etax
        self.posx = posx
        self.posy = posy
        self.yaw = yaw
        self.marker_num = marker_num
        self.etax_v = torch.tensor([etax, etax, etax])
        self.beta = [self.set_beta(b[i], gpmodels.train_targets[i], cov[i])
                     for i in range(len(b))]
        self.epsilon = [self.set_epsilon(
            alpha[i], Lambdax[i]) for i in range(len(b))]
        self.c = [self.set_c(alpha[i], self.epsilon[i]) for i in range(len(b))]
        self.ell_param = torch.cat(
            [torch.diag(self.c[i] * torch.sqrt(self.Lambdax[i])).reshape(1, -1) for i in range(len(b))], dim=0)

    def set_beta(self, b, y, cov):
        return torch.sqrt(b ** 2 - torch.matmul(torch.matmul(y, torch.inverse(cov + torch.eye(self.Time) * self.noise)), y) + self.Time)

    def set_epsilon(self, alpha, Lambda):
        return torch.sqrt(2 * (alpha**2) * (1 - torch.exp(-0.5 * torch.matmul(torch.matmul(self.etax_v, torch.inverse(Lambda)), self.etax_v))))

    def set_c(self, alpha, epsilon):
        return torch.sqrt(2 * torch.log((2 * (alpha**2)) / (2 * (alpha**2) - (epsilon**2))))

    def min_max_check(self, x, xlist, dim):
        return torch.all(x + self.ell_param[:, dim] <= torch.max(xlist)) and torch.all(torch.min(xlist) <= x - self.ell_param[:, dim])

    def operatioin(self, pos_reg, yaw_reg, V, Omega, L, b):
        X0 = torch.arange(self.posx[0] - pos_reg,
                          self.posx[0] + pos_reg + 0.0001, self.etax)
        X1 = torch.arange(self.posy[0] - pos_reg,
                          self.posy[0] + pos_reg + 0.0001, self.etax)
        X2 = torch.arange(self.yaw[0] - yaw_reg,
                          self.yaw[0] + yaw_reg + 0.0001, self.etax)
        X_range_min = torch.tensor([X0.min(), X1.min(), X2.min()])
        X_range_max = torch.tensor([X0.max(), X1.max(), X2.max()])
        print('a')
        print(self.ell_param)
        print(X_range_min)
        print(X_range_max)
        Q = torch.zeros([X0.shape[0], X1.shape[0], X2.shape[0]])
        Qsafe = Q.clone()
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                for k in range(Q.shape[2]):
                    if self.min_max_check(X0[i], X0, dim=0) and self.min_max_check(X1[j], X1, dim=1) and self.min_max_check(X2[k], X2, dim=2):
                        Q[i][j][k] = 1
        Qind = torch.nonzero(Q).int()
        print('b')
        print(Qind)
        print(Qind.shape)
        for i in range(Qind.shape[0]):
            print(i)
            u_flag = 0
            for j in range(V.shape[0]):
                for k in range(Omega.shape[0]):
                    z_test = torch.tensor(
                        [X0[Qind[i, 0]], X1[Qind[i, 1]], X2[Qind[i, 2]], V[j], Omega[k]]).reshape(1, -1).double()
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        predictions = likelihoods(*self.gpmodels(z_test, z_test, z_test))
                    x_next_part = self.vehicle.partF(
                        X0[Qind[i, 0]], X1[Qind[i, 1]], X2[Qind[i, 2]], V[j], Omega[k])
                    means = torch.tensor([predictions[l].mean for l in range(x_next_part.shape[0])])
                    variances = torch.tensor([predictions[l].variance for l in range(x_next_part.shape[0])])
                    xpre_lower = torch.tensor([x_next_part[l] + means[l] - (
                        self.L * self.etax + self.b[l] * self.epsilon[l] + self.beta[l] * torch.sqrt(variances[l]) + self.etax)
                        for l in range(x_next_part.shape[0])])
                    xpre_upper = torch.tensor([x_next_part[l] + means[l] + (
                        self.L * self.etax + self.b[l] * self.epsilon[l] + self.beta[l] * torch.sqrt(variances[l]) + self.etax)
                        for l in range(x_next_part.shape[0])])
                    Qind_lower = torch.ceil((xpre_lower - X_range_min) / self.etax).int()
                    Qind_upper = ((xpre_upper - X_range_min) // etax).int()
                    if torch.all(X_range_min <= xpre_lower) and torch.all(xpre_upper <= X_range_max):
                        if torch.all(Qsafe[Qind_lower[0]:Qind_upper[0] + 1, Qind_lower[1]:Qind_upper[1] + 1, Qind_lower[2]:Qind_upper[2] + 1] == 1):
                            print('c')
                            print(Qind[i])
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

if __name__ == '__main__':
    # set params
    markers = np.array([[0, 0], [0.05, -0.12], [0.2, -0.2], [0.4, -0.1], [0.6, 0.1], [0.8, 0.2], [0.8, 0.5],
                        [0.6, 0.6], [0.4, 0.55], [0.2, 0.45], [0.1, 0.3], [0.05, 0.2], [0, 0]])
    marker_num = 50
    ts = 0.1
    noise = 0.0001
    v_max = 1
    omega_max = 1
    gp_updata_time = 100
    L = 0
    b = [0.1, 0.1, 0.1]
    Time = marker_num * 2
    etax = 0.01
    etau = 0.1
    V = torch.arange(0., v_max + etau, etau)
    Omega_upper = torch.arange(0., omega_max + etau, etau)
    Omega_lower = torch.arange(-omega_max, 0., etau)
    Omega = torch.zeros(int(2 * omega_max / etau + 1))
    for i in range(Omega.shape[0]):
        if i % 2 == 0:
            Omega[i] = Omega_upper[i // 2]
        elif i % 2 == 1:
            Omega[i] = Omega_lower[Omega_lower.shape[0] - i // 2 - 1]
    pos_reg = 0.05
    yaw_reg = 0.6

    # define model
    vehicle = VEHICLE(markers, marker_num, ts, noise)
    posx, posy, yaw = vehicle.getLine()
    getMap(markers, posx, posy)

    # create training dataset
    z_train, y_train = make_data(vehicle, posx, posy, yaw, marker_num)

    # train model
    gpmodels, likelihoods, cov, alpha, Lambdax = train(z_train, y_train)

    # safety game
    safetygame = safetyGame(vehicle, gpmodels, likelihoods, cov,
                            alpha, Lambdax, L, b, Time, noise, etax, posx, posy, yaw, marker_num)
    safetygame.operatioin(pos_reg, yaw_reg, V, Omega, L, b)


# z_train = torch.zeros(1, 5)
# y_train = torch.zeros(1, 3)
# for i in range(maker_num):
#     for j in range(2):
#         if j == 0:
#             v = 0
#             omega = 0
#         else:
#             v = 2 * v_max * torch.rand(1) - v_max
#             omega = 2 * omega_max * torch.rand(1) - omega_max
#         x_next_real = vehicle.RK4(posx[i], posy[i], yaw[i], v, omega)
#         x_next_part = vehicle.partF(posx[i], posy[i], yaw[i], v, omega)
#         z = torch.tensor([posx[i], posy[i], yaw[i], v, omega])
#         z_train = torch.cat([z_train, z.reshape(1, -1)], dim=0)
#         y_train = torch.cat(
#             [y_train, (x_next_real - x_next_part).reshape(1, -1)], dim=0)

# likelihood_list = []
# model_list = []
# for i in range(y_train.shape[1]):
#     likelihood_list.append(gpytorch.likelihoods.GaussianLikelihood())
#     model_list.append(ExactGPModel(
#         z_train[1:], y_train[1:, 0], likelihood_list[i]))

# models = gpytorch.models.IndependentModelList(
#     model_list[0], model_list[1], model_list[2])
# likelihoods = gpytorch.likelihoods.LikelihoodList(
#     model_list[0].likelihood, model_list[1].likelihood, model_list[2].likelihood)
# mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihoods, models)

# models.train()
# likelihoods.train()
# optimizer = torch.optim.Adam(models.parameters(), lr=0.2)
# for i in range(gp_updata_time):
#     optimizer.zero_grad()
#     output = models(*models.train_inputs)
#     loss = -mll(output, models.train_targets)
#     loss.backward()
#     optimizer.step()
# models.eval()
# likelihoods.eval()

# for i in range(marker_num):
#     v = 2 * v_max * torch.rand(1) - v_max
#     omega = 2 * omega_max * torch.rand(1) - omega_max

#     x_next_real = vehicle.RK4(posx[i], posy[i], yaw[i], v, omega)
#     x_next_part = vehicle.partF(posx[i], posy[i], yaw[i], v, omega)

#     z_test = torch.tensor(
#         [posx[i], posy[i], yaw[i], v, omega]).reshape(1, -1)
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         predictions = likelihoods(*gpmodels(z_test, z_test, z_test))

#     predictions_mean = torch.tensor(
#         [predictions[0].mean, predictions[1].mean, predictions[2].mean])
#     predictions_variance = torch.tensor(
#         [predictions[0].variance, predictions[1].variance, predictions[2].variance])
#     print('aaaaa')
#     print(torch.tensor([posx[i], posy[i], yaw[i]]))
#     print(x_next_real)
#     print(x_next_part + predictions_mean)
#     print(predictions_mean)
#     print(predictions_variance)
