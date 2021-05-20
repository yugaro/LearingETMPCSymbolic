import numpy as np
from blueprint.set_args import set_args
from model.vehicle import Vehicle
from model.gp import GP
from controller.symbolic import Symbolic
from controller.etmpc import ETMPC
np.random.seed(0)


def iterLearning(args, vehicle, z_train, y_train, traj_data, trigger_data, iter_num):
    # gp and safety game
    gpmodels = GP(z_train, y_train, args.noise)
    symmodel = Symbolic(args, gpmodels, y_train)
    return
    Q = symmodel.safeyGame()
    np.save('./data/Q.npy', Q)

    # etmpc
    etmpc = ETMPC(args, gpmodels, y_train, symmodel.gamma)
    mpc, simulator, estimator = etmpc.setUp()
    flag_asm = 0
    while 1:
        ze_train = np.zeros((1, 5))
        ye_train = np.zeros((1, 3))
        x0 = np.array([np.random.rand(1) + 2.5,
                       np.random.rand(1) + 2.5, 2 * np.random.rand(1) - 1])
        while 1:
            etmpc.set_initial(mpc, simulator, estimator, x0)
            mpc_status, ulist = etmpc.operation(
                mpc, simulator, estimator, x0)
            trigger_status, trigger_values = etmpc.triggerValue(mpc)

            print('mpc status:', mpc_status)
            print('trigger status:', trigger_status)
            if mpc_status and trigger_status == 'optimal':
                flag_asm = 1
                ulist_pre = ulist
                for i in range(args.horizon + 1):
                    if i == 0:
                        xe = x0.reshape(-1)
                        xr = np.array(args.xinit_r).reshape(-1)
                        traj_data[iter_num] = np.concatenate(
                            [traj_data[iter_num], (xr - xe).reshape(1, -1)], axis=0)
                    xstar = np.array(
                        mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)

                    if etmpc.triggerF(xstar, xe, trigger_values[i, :]):
                        if i == args.horizon:
                            return [1, traj_data, trigger_data]
                        u = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
                        ur = np.array([args.v_r, args.omega_r])

                        xe_next = vehicle.errRK4(xe, u)
                        xr_next = vehicle.realRK4(xr, ur)

                        traj_data[iter_num] = np.concatenate(
                            [traj_data[iter_num], (xr_next - xe_next).reshape(1, -1)], axis=0)

                        ze_train, ye_train = etmpc.learnD(
                            xe, u, xe_next, ze_train, ye_train)

                        xe = xe_next
                        xr = xr_next
                    else:
                        trigger_time = i
                        x0 = xe
                        trigger_data[iter_num] = np.concatenate(
                            [trigger_data[iter_num], np.array([trigger_time])])
                        print('trigger:', trigger_time)
                        break
            else:
                if flag_asm == 0:
                    print('assumption is not hold.')
                    break
                elif flag_asm == 1:
                    for i in range(args.horizon - trigger_time):
                        if i == 0:
                            xe = x0.reshape(-1)
                            traj_data[iter_num] = np.concatenate(
                                [traj_data[iter_num], (xr - xe).reshape(1, -1)], axis=0)
                        u = ulist_pre[trigger_time + i]
                        ur = np.array([args.v_r, args.omega_r])

                        xe_next = vehicle.errRK4(
                            xe, u)
                        xr_next = vehicle.realRK4(xr, ur)

                        traj_data[iter_num] = np.concatenate(
                            [traj_data[iter_num], (xr_next - xe_next).reshape(1, -1)], axis=0)

                        # ze_train, ye_train = etmpc.learnD(
                        #     xe, ulist_pre[trigger_time + i], xe_next, ze_train, ye_train)
                        xe = xe_next
                    x0 = xe
                    z_train_sum, y_train_sum = etmpc.dataCat(
                        ze_train[1:], ye_train[1:])
                    return [0, z_train_sum, y_train_sum]


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train = np.load(args.datafile_z)
    y_train = np.load(args.datafile_y)
    traj_data = [np.zeros((1, 3)) for i in range(100)]
    trigger_data = [np.zeros(1) for i in range(100)]
    iter_num = 0
    while 1:
        iterdata = iterLearning(args, vehicle, z_train,
                                y_train, traj_data, trigger_data, iter_num)
        iter_num += 1
        if iterdata[0] == 1:
            print('Event-triggered mpc was completed.')
            break
        else:
            z_train = iterdata[1].copy()
            y_train = iterdata[2].copy()
    for i in range(iter_num):
        np.save('./data/traj{}.npy'.format(i), iterdata[1][i])
        np.save('./data/trigger{}.npy'.format(i), iterdata[2][i])
    np.save('./data/iter_num.npy', np.array([iter_num]))
