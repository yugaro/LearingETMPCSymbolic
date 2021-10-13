import numpy as np
from blueprint.set_args import set_args
from model.vehicle import Vehicle
import matplotlib.pyplot as plt
np.random.seed(0)


def make_data(args, vehicle):
    xinits = np.array(
        [[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5],
         [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
         [0, 0, 0.5], [0, 0, -0.5], [0, 0.5, 0], [0, -0.5, 0], [0.5, 0, 0], [-0.5, 0, 0]])
    # [-0.5, 0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.5],
    # [[-2, -2, 2], [-1, 1, -1], [1, -1, -1], [2, 2, 2], [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]]
    # seed:0, [[2, 2, 2], [-0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-2, -2, -2], [-0.5, -0.5, -0.5]]
    # [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5],
    # [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
    # [1, 1, 1]
    xinits = xinits * 1
    z_train = np.zeros((1, 5))
    y_train = np.zeros((1, 3))

    p_num = 5

    for i in range(xinits.shape[0] * p_num):
        if i % p_num == 0:
            j = i // p_num
            x = xinits[j, :]
        if np.random.rand() > 1.9:
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
    print(z_train[:, [3, 4]])
    fig, ax = plt.subplots()
    ax.plot(y_train[:, 0], label=r'${\rm x}$')
    ax.plot(y_train[:, 1], label=r'${\rm y}$')
    ax.plot(y_train[:, 2], label=r'$\theta$')
    ax.legend()
    plt.show()

    return z_train[1:], y_train[1:]


def trajPID(args, vehicle):
    xinit = np.array([2, 2, 2])
    traj_data = np.zeros((1, 3))
    for i in range(90):
        if i == 0:
            xe = xinit
            xr = np.array(args.xinit_r).reshape(-1)

            theta = xr[2] - xe[2]
            rotation = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            pos = xr[: 2] - rotation @ xe[: 2]
            state = np.concatenate([pos, np.array([theta])])

            traj_data = np.concatenate(
                [traj_data, state.reshape(1, -1)], axis=0)
        u = vehicle.getPIDCon(xe)
        xe_next = vehicle.errRK4(xe, u)

        ur = np.array([args.v_r, args.omega_r])
        xr_next = vehicle.realRK4(xr, ur)

        theta_next = xr_next[2] - xe_next[2]
        rotation_next = np.array(
            [[np.cos(theta_next), -np.sin(theta_next)], [np.sin(theta_next), np.cos(theta_next)]])
        pos_next = xr_next[: 2] - rotation_next @ xe_next[: 2]
        state_next = np.concatenate(
            [pos_next, np.array([theta_next])])

        traj_data = np.concatenate(
            [traj_data, state_next.reshape(1, -1)], axis=0)

        xe = xe_next
        xr = xr_next

    fig, ax = plt.subplots(1, 1)
    ax.plot(traj_data[1:, 0], traj_data[1:, 1])
    ax.scatter(traj_data[1, 0], traj_data[1, 1])
    plt.show()


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train, y_train = make_data(args, vehicle)
    np.save('../data/z_train.npy', z_train)
    np.save('../data/y_train.npy', y_train)
    # trajPID(args, vehicle)

# trajPID(args, vehicle)
# def make_data(args, vehicle):
#     xinits = np.array([[2., 2., 2.], [2., 2., -2.], [-2., -2., -2.], [-2., -2., 2.],
#                        [-2., 2., 2.], [-2., 2., -2.], [2., -2., 2.], [2, -2, -2]])
#     z_train = np.zeros((1, 5))
#     y_train = np.zeros((1, 3))
#     for i in range(xinits.shape[0] * 10):
#         if i % 10 == 0:
#             j = i // 10
#             x = xinits[j, :]
#         if np.random.rand(1) > .5:
#             u = np.array([2, 2 * 1]) * \
#                 np.random.rand(1) - np.array([0, 1])
#         else:
#             u = vehicle.getPIDCon(x)
#         x_next = vehicle.errRK4(x, u)

#         z = np.concatenate([x, u], axis=0)
#         z_train = np.concatenate([z_train, z.reshape(1, -1)], axis=0)
#         y_train = np.concatenate(
#             [y_train, (x_next - x).reshape(1, -1)], axis=0)
#         x = x_next

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(z_train[1:, 0], marker='*')
    # ax.grid(linestyle='-')
    # fig.savefig('../image/trajectoryx.pdf')

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(z_train[1:, 1], marker='*')
    # ax.grid(linestyle='-')
    # fig.savefig('../image/trajectoryy.pdf')

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(z_train[1:, 2], marker='*')
    # ax.grid(linestyle='-')
    # fig.savefig('../image/trajectorytheta.pdf')

#     return z_train[1:], y_train[1:]
