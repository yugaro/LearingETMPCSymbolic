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
    # symmodel = Symbolic(args, gpmodels, y_train)
    symmodel = Symbolic(args, gpmodels)
    # return
    Q, Qind = symmodel.safeyGame()
    np.save('../data/Q2.npy', Q)
    np.save('../data/Qind2.npy', Qind)
    return

    while 1:
        ze_train = np.zeros((1, 5))
        ye_train = np.zeros((1, 3))

        # set initial state
        x0 = np.array([np.random.rand(1) + 2.5,
                       np.random.rand(1) + 2.5, 6 * np.random.rand(1) - 3])

        # set initial horizon
        horizon = args.horizon

        while 1:
            # solve OCP
            etmpc = ETMPC(args, gpmodels, symmodel.gamma, horizon)
            mpc, simulator, estimator = etmpc.setUp()
            etmpc.set_initial(mpc, simulator, estimator, x0)
            mpc_status, ulist = etmpc.operation(
                mpc, simulator, estimator, x0)

            # solve optimization problem
            psi_status, psi_values = etmpc.psiF(mpc)
            xi_status, xi_values = etmpc.xiF(
                mpc, psi_values)

            print('mpc status:', mpc_status)
            print('xi status:', xi_status)
            if mpc_status is not True or xi_status != 'optimal':
                print('assumption is not hold.')
                break
            ulist_pre = ulist
            for i in range(horizon + 1):
                if i == 0:
                    xe = x0.reshape(-1)
                    xr = np.array(args.xinit_r).reshape(-1)
                    traj_data[iter_num] = np.concatenate(
                        [traj_data[iter_num], (xr - xe).reshape(1, -1)], axis=0)
                xstar = np.array(
                    mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)

                if etmpc.triggerF(xstar, xe, xi_values[i, :]):
                    if i == horizon:
                        return [1, traj_data, trigger_data]
                    u = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
                    ur = np.array([args.v_r, args.omega_r])

                    xe_next = vehicle.errRK4(xe, u)
                    xr_next = vehicle.realRK4(xr, ur)

                    traj_data[iter_num] = np.concatenate(
                        [traj_data[iter_num], (xr_next - xe_next).reshape(1, -1)], axis=0)

                    ze_train, ye_train = etmpc.learnD(
                        xe, u, xe_next, ze_train, ye_train, xi_values)

                    xe = xe_next
                    xr = xr_next
                else:
                    trigger_time = i
                    horizon_tmp = horizon - (trigger_time - 1)
                    x0 = xe
                    trigger_data[iter_num] = np.concatenate(
                        [trigger_data[iter_num], np.array([trigger_time])])
                    print('trigger:', trigger_time)
                    break
            if horizon_tmp == 1:
                print('horizon becomes 1')
                return [1, traj_data, trigger_data]
            elif horizon == horizon_tmp:
                print('horizon is not decreased')
                z_train_sum, y_train_sum = etmpc.dataCat(
                    ze_train[1:], ye_train[1:])
                return [0, z_train_sum, y_train_sum]
            else:
                print('horizon is decreased:', horizon_tmp)
                horizon = horizon_tmp


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train = np.load(args.datafile_z)
    y_train = np.load(args.datafile_y)
    traj_data = [np.zeros((1, 3)) for i in range(100)]
    trigger_data = [np.zeros(1) for i in range(100)]
    iter_num = 0
    while 1:
        iterdata = iterTask(args, vehicle, z_train,
                            y_train, traj_data, trigger_data, iter_num)
        iter_num += 1
        if iterdata[0] == 1:
            print('Event-triggered mpc was completed.')
            break
        else:
            z_train = iterdata[1].copy()
            y_train = iterdata[2].copy()
    for i in range(iter_num):
        np.save('../data/traj{}.npy'.format(i), iterdata[1][i])
        np.save('../data/trigger{}.npy'.format(i), iterdata[2][i])
    np.save('../data/iter_num.npy', np.array([iter_num]))
