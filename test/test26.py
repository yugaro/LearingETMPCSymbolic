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

# set param (all)
vr = 1.
omegar = 1.
ur = torch.tensor([vr, omegar])
v_max = 2
omega_max = 2
ts = 0.1
noise = 0.001
b = [1.07, 1.07, 1.07]

# set param (gp)
xinits = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
                       [0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [0., 0., 0.], [1., 1., 1.]])
data_num = 100
gpudate_num = 100
Kx = 1
Ky = 1
Ktheta = 1

# set param (safety game)
etax = 0.05
etau = 0.2
Vq = torch.arange(0., v_max + etau, etau)
Omegaq = torch.arange(0., omega_max + etau, etau)
Uq = torch.zeros(Vq.shape[0] * Omegaq.shape[0], 2)
for i in range(Vq.shape[0]):
    for j in range(Omegaq.shape[0]):
        Uq[i * Omegaq.shape[0] + j, :] = torch.tensor([Vq[i], Omegaq[j]])
gamma_param = [60, 40, 40]
Xsafe = torch.tensor([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])

# set param (etmpc)
mpctype = 'discrete'
weightx = np.diag([1, 1, 1])
horizon = 25
xr_init = np.array([[0., 0., 0.]])

if __name__ == '__main__':
    vehicle = VEHICLE(ts, noise, vr, omegar, Kx, Ky, Ktheta)
    z_train, y_train = make_data(vehicle, xinits, data_num)
    gpmodels, likelihoods, cov = train(
        z_train, y_train, gpudate_num)

    alpha = np.array([torch.sqrt(gpmodels.models[i].covar_module.outputscale).to(
        'cpu').detach().numpy().astype(np.float64) for i in range(3)])

    Lambdax = [(torch.diag(gpmodels.models[i].covar_module.base_kernel.lengthscale.reshape(-1)
                           [:3]) ** 2).to('cpu').detach().numpy().astype(np.float64) for i in range(y_train.shape[1])]

    Y = [gpmodels.train_targets[i].to('cpu').detach().numpy().astype(np.float64) for i in range(3)]

    safetygame.print_array(alpha, Lambdax[0], Lambdax[1], Lambdax[2],
                           cov[0].to('cpu').detach().numpy().astype(np.float64), cov[1].to('cpu').detach().numpy().astype(np.float64),
                           cov[2].to('cpu').detach().numpy().astype(np.float64), Y[0], Y[1], Y[2])
