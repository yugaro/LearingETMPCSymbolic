import numpy as np
from blueprint.set_args import set_args
from model.vehicle import Vehicle
np.random.seed(0)


def make_data(args, vehicle):
    xinits = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
                       [0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5]])
    z_train = np.zeros((1, 5))
    y_train = np.zeros((1, 3))
    for i in range(xinits.shape[0] * 8):
        if i % 8 == 0:
            j = i // 8
            x = xinits[j, :]
        if np.random.rand(1) > 0.8:
            u = np.array([2, 2 * 1]) * \
                np.random.rand(1) - np.array([0, 1])
        else:
            u = vehicle.getPIDCon(x)
        x_next = vehicle.errRK4(x, u)

        z = np.concatenate([x, u], axis=0)
        z_train = np.concatenate([z_train, z.reshape(1, -1)], axis=0)
        y_train = np.concatenate(
            [y_train, (x_next - x).reshape(1, -1)], axis=0)
        x = x_next
    # np.set_printoptions(precision=3)
    # print(z_train)
    # print(np.mean(z_train, axis=0))
    # print(np.min(z_train, axis=0))
    # print(np.max(z_train, axis=0))
    # print(np.std(y_train, axis=0))
    # print(np.mean(y_train, axis=0))
    return z_train[1:], y_train[1:]

if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train, y_train = make_data(args, vehicle)
    np.save('./data/z_train.npy', z_train)
    np.save('./data/y_train.npy', y_train)
