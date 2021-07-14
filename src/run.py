import numpy as np
from blueprint.set_args import set_args
from model.vehicle import Vehicle
from model.gp import GP
from controller.symbolic import Symbolic
from controller.etmpc import ETMPC
np.random.seed(0)


def iterTask(args, vehicle, z_train, y_train, traj_data, trigger_data, iter_num):
    # gp and safety game
    gpmodels = GP(z_train, y_train, args.noise)
    symmodel = Symbolic(args, gpmodels)
    # return
    # Q, Qind, Cs = symmodel.safeyGame()
    # np.save('../data/Q3.npy', Q)
    # np.save('../data/Qind3.npy', Qind)
    # np.save('../data/Cs3.npy', Cs)
    # return

    while 1:
        ze_train = np.zeros((1, 5))
        ye_train = np.zeros((1, 3))

        # set initial state
        x0 = np.array([np.random.rand(1) + 4.,
                       np.random.rand(1) + 4., 6 * np.random.rand(1) - 3])

        # set initial horizon
        horizon = args.horizon
        etmpc = ETMPC(args, gpmodels, symmodel.gamma, horizon)
        mpc, simulator, estimator = etmpc.setUp()

        while 1:
            # solve OCP
            etmpc.set_initial(mpc, simulator, estimator, x0)
            mpc_status, ulist = etmpc.operation(
                mpc, simulator, estimator, x0)

            print('mpc status:', mpc_status)
            if mpc_status is False:
                print('Assumption is not hold. (MPC)')
                break

            # solve optimization problem
            xi_status, xi_values = etmpc.xiF(mpc)
            print('xi status:', xi_status)
            if xi_status == 'optimal':
                for i in range(horizon + 1):
                    if i == 0:
                        xe = x0.reshape(-1)
                        xr = np.array(args.xinit_r).reshape(-1)
                        traj_data[iter_num] = np.concatenate(
                            [traj_data[iter_num], (xr - xe).reshape(1, -1)], axis=0)
                    xstar = np.array(
                        mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)

                    if etmpc.triggerF(xstar, xe, xi_values[i, :]) and i < horizon:
                        u = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
                        ur = np.array([args.v_r, args.omega_r])

                        xe_next = vehicle.errRK4(xe, u)
                        xr_next = vehicle.realRK4(xr, ur)

                        traj_data[iter_num] = np.concatenate(
                            [traj_data[iter_num], (xr_next - xe_next).reshape(1, -1)], axis=0)

                        if horizon != 1:
                            ze_train, ye_train = etmpc.learnD(
                                xe, u, xe_next, ze_train, ye_train, xi_values, 1)

                        xe = xe_next
                        xr = xr_next
                    else:
                        trigger_time = i
                        horizon_pre = horizon
                        horizon -= (trigger_time - 1)
                        x0 = xe
                        trigger_data[iter_num] = np.concatenate(
                            [trigger_data[iter_num], np.array([trigger_time])])
                        print('trigger:', trigger_time)
                        print('updated horizon:', horizon)
                        break
                if horizon == 1 and horizon_pre == 1:
                    print('Horizon becomes 1.')
                    return [1, traj_data, trigger_data]
            else:
                print('xi status:', xi_status)
                print('Assumption is not hold. (xi)')
                for i in range(horizon):
                    if i == 0:
                        xe = x0.reshape(-1)
                        xr = np.array(args.xinit_r).reshape(-1)
                        traj_data[iter_num] = np.concatenate(
                            [traj_data[iter_num], (xr - xe).reshape(1, -1)], axis=0)

                    u = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
                    ur = np.array([args.v_r, args.omega_r])

                    xe_next = vehicle.errRK4(xe, u)
                    xr_next = vehicle.realRK4(xr, ur)

                    traj_data[iter_num] = np.concatenate(
                        [traj_data[iter_num], (xr_next - xe_next).reshape(1, -1)], axis=0)

                    ze_train, ye_train = etmpc.learnD(
                        xe, u, xe_next, ze_train, ye_train, xi_values, 0)

                    xe = xe_next
                    xr = xr_next

                z_train_sum, y_train_sum = etmpc.dataCat(ze_train[1:], ye_train[1:])
                print('data points num:', z_train_sum.shape[0])
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
        print('Iter:', iter_num + 1)
        iterdata = iterTask(args, vehicle, z_train,
                            y_train, traj_data, trigger_data, iter_num)
        iter_num += 1
        if iterdata[0] == 1:
            print('Event-triggered mpc was completed in the iter ', iter_num, '.')
            break
        else:
            z_train = iterdata[1].copy()
            y_train = iterdata[2].copy()
            print('Proceed to the next iteration.')
    for i in range(iter_num):
        np.save('../data/traj{}.npy'.format(i), iterdata[1][i])
        np.save('../data/trigger{}.npy'.format(i), iterdata[2][i])
    np.save('../data/iter_num.npy', np.array([iter_num]))

# if xi_status != 'optimal':
    #     print('xi status:', xi_status)
    #     print('Assumption is not hold. (xi)')
    #     break
    # else:
    #     print('horizon is decreased:', horizon_tmp)
    #     horizon = horizon_tmp
    # print('xi status:', xi_status)
    # if mpc_status is not True or xi_status != 'optimal':
    #     print('assumption is not hold.')
    #     break
    # as_tmp = 1
    # elif horizon == horizon_tmp:
    #     print('horizon is not decreased')
    #     z_train_sum, y_train_sum = etmpc.dataCat(
    #         ze_train[1:], ye_train[1:])
    #     return [0, z_train_sum, y_train_sum]
