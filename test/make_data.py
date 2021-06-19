import numpy as np
from blueprint.set_args import set_args
from model.vehicle import Vehicle
import matplotlib.pyplot as plt
np.random.seed(0)


def make_data(args, vehicle):
    xinits = np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, -0.5, 0.5],
                       [-0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [2, 2, 2]])
    xinits = xinits * 2
    z_train = np.zeros((1, 5))
    y_train = np.zeros((1, 3))
    for i in range(xinits.shape[0] * 5):
        if i % xinits.shape[0] == 0:
            j = i // xinits.shape[0]
            x = xinits[j, :]
        if np.random.rand(1) > 0.8:
            u = np.array([2, 2]) * \
                np.random.rand(1) - np.array([1, 1])
        else:
            u = vehicle.getPIDCon(x)
        x_next = vehicle.errRK4(x, u)

        z = np.concatenate([x, u], axis=0)
        z_train = np.concatenate([z_train, z.reshape(1, -1)], axis=0)
        y_train = np.concatenate(
            [y_train, (x_next - x).reshape(1, -1)], axis=0)
        x = x_next
    fig, ax = plt.subplots(1, 1)
    ax.plot(y_train[1:, 0], marker='*')
    ax.grid(linestyle='-')
    fig.savefig('./image/trajectoryx.pdf')

    fig, ax = plt.subplots(1, 1)
    ax.plot(y_train[1:, 1], marker='*')
    ax.grid(linestyle='-')
    fig.savefig('./image/trajectoryy.pdf')

    fig, ax = plt.subplots(1, 1)
    ax.plot(y_train[1:, 2], marker='*')
    ax.grid(linestyle='-')
    fig.savefig('./image/trajectorytheta.pdf')
    return z_train[1:], y_train[1:]


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train, y_train = make_data(args, vehicle)
    np.save(args.datafile_z, z_train)
    np.save(args.datafile_y, y_train)
