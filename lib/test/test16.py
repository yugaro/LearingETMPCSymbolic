import torch


class VEHICLE:
    def __init__(self, X, Y, theta, v, omega):
        self.X = X
        self.Y = Y
        self.theta = theta
        self.v = v
        self.omega = omega
        self.x = torch.tensor([self.X, self.Y, self.theta])
        self.u = torch.tensor([self.v, self.omega])
        self.z = torch.tensor([self.X, self.Y, self.theta, self.v, self.omega])
