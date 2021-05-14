import torch
from component.blueprint.set_args import set_args
from model.vehicle import Vehicle


def make_data(args, vehicle):
    xinits = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
                           [0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [0., 0., 0.], [1., 1., 1.]])
    z_train = torch.zeros(1, 5)
    y_train = torch.zeros(1, 3)
    for i in range(args.data_num):
        if i % 10 == 0:
            j = i // 10
            x = xinits[j, :]
        if torch.rand(1) > 0.8:
            u = torch.tensor([args.v_max * torch.rand(1), args.omega_max * torch.rand(1)])
        else:
            u = vehicle.getPIDCon(x)
        x_next = vehicle.errRK4(x, u)

        z = torch.cat([x, u], dim=0)
        z_train = torch.cat([z_train, z.reshape(1, -1)], dim=0)
        y_train = torch.cat(
            [y_train, (x_next - x).reshape(1, -1)], dim=0)
        x = x_next
    return z_train[1:], y_train[1:]

if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train, y_train = make_data(args, vehicle)
    torch.save(z_train, '../data/z_train.pt')
    torch.save(y_train, '../data/y_train.pt')
