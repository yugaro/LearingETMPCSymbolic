import torch
import numpy as np
from scipy import interpolate
import gpytorch

class VEHICLE:
    def __init__(self, makers, maker_num, ts, noise_max):
        self.ts = ts
        self.noise_max = noise_max
        self.makers = makers
        self.tck, _ = interpolate.splprep(
            [self.makers[:, 0], self.makers[:, 1]], k=3, s=0)
        self.maker_num = maker_num

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

    def RK4(self, X, Y, theta, v, omega):
        x, u, z = self.stateInputToVector(X, Y, theta, v, omega)
        k1 = self.getF(x, u)
        k2 = self.getF(x + self.ts / 2 * k1[2], u)
        k3 = self.getF(x + self.ts / 2 * k2[2], u)
        k4 = self.getF(x + self.ts * k3[2], u)
        x_next = x + self.ts / 6 * \
            (k1 + 2 * k2 + 2 * k3 + k4) + 2 * \
            self.noise_max * torch.rand(3) - self.noise_max
        return x_next

    def RK1(self, X, Y, theta, v, omega):
        x, u, z = self.stateInputToVector(X, Y, theta, v, omega)
        k1 = self.getF(x, u)
        x_next = x + self.ts * k1
        return x_next

    def getLine(self):
        arc = np.linspace(0, 1, num=self.maker_num)
        spline = interpolate.splev(arc, self.tck)
        yaw = np.zeros(self.maker_num)
        for i in range(1, self.maker_num - 1):
            yaw[i] = np.arctan2(spline[1][i + 1] - spline[1][i - 1], spline[0][i + 1] - spline[0][i - 1])
        yaw[0] = yaw[1]
        yaw[-1] = yaw[-2]
        return spline[0], spline[1], yaw


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

makers = np.array([[0, 0], [1, -0.5], [2, 0], [3, 0.5], [4, 1.5], [4.8, 1.5], [5, 0.8], [6, 0.5], [6.5, 0], [7.5, 0.5],
                   [7, 2], [6, 3], [5, 4], [4., 2.5], [3, 3], [2., 3.5], [1.3, 2.2], [0.5, 2.], [-0.1, 1], [0, 0]])
maker_num = 100
ts = 0.1
noise_max = 0.01
v_max = 0.1
omega_max = 0.1
gp_updata_time = 100

vehicle = VEHICLE(makers, maker_num, ts, noise_max)
positionx, positiony, yaw = vehicle.getLine()

z_train = torch.zeros(1, 5)
y_train = torch.zeros(1, 3)
for i in range(maker_num):
    v = 2 * v_max * torch.rand(1) - v_max
    omega = 2 * omega_max * torch.rand(1) - omega_max

    z = torch.tensor([positionx[i], positiony[i], yaw[i], v, omega])
    x_next_real = vehicle.RK4(positionx[i], positionx[i], yaw[i], v, omega)
    x_next_nominal = vehicle.RK1(positionx[i], positionx[i], yaw[i], v, omega)

    z_train = torch.cat([z_train, z.reshape(1, -1)], dim=0)
    y_train = torch.cat([y_train, (x_next_real - x_next_nominal).reshape(1, -1)], dim=0)

likelihood_list = []
model_list = []
for i in range(y_train.shape[1]):
    likelihood_list.append(gpytorch.likelihoods.GaussianLikelihood())
    model_list.append(ExactGPModel(
        z_train[1:], y_train[1:, i], likelihood_list[i]))

models = gpytorch.models.IndependentModelList(
    model_list[0], model_list[1], model_list[2])
likelihoods = gpytorch.likelihoods.LikelihoodList(
    model_list[0].likelihood, model_list[1].likelihood, model_list[2].likelihood)
mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihoods, models)

models.train()
likelihoods.train()
optimizer = torch.optim.Adam(models.parameters(), lr=0.2)
for i in range(gp_updata_time):
    optimizer.zero_grad()
    output = models(*models.train_inputs)
    loss = -mll(output, models.train_targets)
    loss.backward()
    optimizer.step()
models.eval()
likelihoods.eval()

for i in range(maker_num):
    v = 2 * v_max * torch.rand(1) - v_max
    omega = 2 * omega_max * torch.rand(1) - omega_max

    x_next_real = vehicle.RK4(positionx[i], positionx[i], yaw[i], v, omega)
    x_next_nominal = vehicle.RK1(positionx[i], positionx[i], yaw[i], v, omega)

    z_test = torch.tensor(
        [positionx[i], positiony[i], yaw[i], v, omega]).reshape(1, -1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihoods(*models(z_test, z_test, z_test))

    predictions_mean = torch.tensor(
        [predictions[0].mean, predictions[1].mean, predictions[2].mean])
    predictions_variance = torch.tensor(
        [predictions[0].variance, predictions[1].variance, predictions[2].variance])
    print('aaaaa')
    print(x_next_real)
    print(x_next_nominal)
    print(x_next_nominal + predictions_mean)
    print(predictions_mean)
    print(predictions_variance)
