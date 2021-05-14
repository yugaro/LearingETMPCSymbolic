import torch
torch.manual_seed(1)


class Vehicle:
    def __init__(self, args):
        self.ts = args.ts
        self.noise = args.noise
        self.v_r = args.v_r
        self.omega_r = args.omega_r
        self.Kx = args.Kx
        self.Ky = args.Ky
        self.Ktheta = args.Ktheta

    def getRF(self, x, u):
        f0 = torch.tensor([torch.cos(x[2]) * (u[0])])
        f1 = torch.tensor([torch.sin(x[2]) * (u[0])])
        f2 = torch.tensor([(u[1])])
        return torch.tensor([f0, f1, f2])

    def realRK4(self, x, u):
        k1 = self.getRF(x, u)
        k2 = self.getRF(x + self.ts / 2 * k1[2], u)
        k3 = self.getRF(x + self.ts / 2 * k2[2], u)
        k4 = self.getRF(x + self.ts * k3[2], u)
        x_next = x + self.ts / 6 * \
            (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def getEF(self, x, u):
        f0 = torch.tensor([u[1] * x[1] - u[0] + self.v_r * torch.cos(x[2])])
        f1 = torch.tensor([-u[1] * x[0] + self.v_r * torch.sin(x[2])])
        f2 = torch.tensor([self.omega_r - u[1]])
        return torch.tensor([f0, f1, f2])

    def errRK4(self, x, u):
        k1 = self.getEF(x, u)
        k2 = self.getEF(x + self.ts / 2 * k1[2], u)
        k3 = self.getEF(x + self.ts / 2 * k2[2], u)
        k4 = self.getEF(x + self.ts * k3[2], u)
        x_next = x + self.ts / 6 * \
            (k1 + 2 * k2 + 2 * k3 + k4) + 2 * \
            self.noise * torch.rand(3) - self.noise
        return x_next

    def getPIDCon(self, x):
        v = self.v_r * torch.cos(x[2]) + self.Kx * x[0]
        omega = self.omega_r + self.v_r * \
            (self.Ky * x[1] + self.Ktheta * torch.sin(x[2]))
        return torch.tensor([v, omega])
