import numpy as np
from blueprint.set_args import set_args
from model.vehicle import Vehicle
from model.gp import GP
from controller.symbolic import Symbolic
# from controller.etmpc import ETMPC
np.random.seed(1)


def iterLearning(args, vehicle, z_train, y_train, traj_data, trigger_data, iter_num):
    # gp and safety game
    gpmodels = GP(z_train, y_train, args.noise)
    z_test = np.array([[1, 1, 1, 1, 0]])
    print(gpmodels.predict(z_test))
    print(vehicle.errRK4(
        np.array([1, 1, 1]), np.array([1, 0])) - np.array([1, 1, 1]))
    # print(y_train)
    # print(gpmodels.gpr[0].y_train_)
    # print(gpmodels.gpr[1].X_train_)
    # print(gpmodels.gpr[2].X_train_)
    symmodel = Symbolic(args, gpmodels, y_train)
    # # return
    # # Q, Qind = symmodel.safeyGame()
    # # np.save('../data/Q2.npy', Q)
    # # np.save('../data/Qind2.npy', Qind)
    # # return

    # # etmpc
    # etmpc = ETMPC(args, gpmodels, y_train, symmodel.gamma)
    # mpc, simulator, estimator = etmpc.setUp()
    # trigger_status, trigger_values2 = etmpc.triggerValue2(mpc)

    # flag_asm = 0
    # while 1:
    #     ze_train = np.zeros((1, 5))
    #     ye_train = np.zeros((1, 3))
    #     x0 = np.array([np.random.rand(1) + 1,
    #                    np.random.rand(1) + 1, 1 * np.random.rand(1)])
    #     while 1:
    #         etmpc.set_initial(mpc, simulator, estimator, x0)
    #         mpc_status, ulist = etmpc.operation(
    #             mpc, simulator, estimator, x0)
    #         trigger_status, trigger_values3 = etmpc.triggerValue3(
    #             mpc, trigger_values2)
    #         # print(trigger_values3)
    #         # trigger_status, trigger_values = etmpc.triggerValue(mpc)

    #         print('mpc status:', mpc_status)
    #         print('trigger status:', trigger_status)
    #         if mpc_status and trigger_status == 'optimal':
    #             flag_asm = 1
    #             ulist_pre = ulist
    #             for i in range(args.horizon + 1):
    #                 if i == 0:
    #                     xe = x0.reshape(-1)
    #                     xr = np.array(args.xinit_r).reshape(-1)
    #                     traj_data[iter_num] = np.concatenate(
    #                         [traj_data[iter_num], (xr - xe).reshape(1, -1)], axis=0)
    #                 xstar = np.array(
    #                     mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)

    #                 if etmpc.triggerF(xstar, xe, trigger_values3[i, :]):
    #                     if i == args.horizon:
    #                         return [1, traj_data, trigger_data]
    #                     u = np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
    #                     ur = np.array([args.v_r, args.omega_r])

    #                     xe_next = vehicle.errRK4(xe, u)
    #                     xr_next = vehicle.realRK4(xr, ur)

    #                     traj_data[iter_num] = np.concatenate(
    #                         [traj_data[iter_num], (xr_next - xe_next).reshape(1, -1)], axis=0)

    #                     ze_train, ye_train = etmpc.learnD(
    #                         xe, u, xe_next, ze_train, ye_train)

    #                     xe = xe_next
    #                     xr = xr_next
    #                 else:
    #                     trigger_time = i
    #                     x0 = xe
    #                     trigger_data[iter_num] = np.concatenate(
    #                         [trigger_data[iter_num], np.array([trigger_time])])
    #                     print('trigger:', trigger_time)
    #                     break
    #         else:
    #             if flag_asm == 0:
    #                 print('assumption is not hold.')
    #                 break
    #             elif flag_asm == 1:
    #                 for i in range(args.horizon - trigger_time):
    #                     if i == 0:
    #                         xe = x0.reshape(-1)
    #                         traj_data[iter_num] = np.concatenate(
    #                             [traj_data[iter_num], (xr - xe).reshape(1, -1)], axis=0)
    #                     u = ulist_pre[trigger_time + i]
    #                     ur = np.array([args.v_r, args.omega_r])

    #                     xe_next = vehicle.errRK4(
    #                         xe, u)
    #                     xr_next = vehicle.realRK4(xr, ur)

    #                     traj_data[iter_num] = np.concatenate(
    #                         [traj_data[iter_num], (xr_next - xe_next).reshape(1, -1)], axis=0)

    #                     xe = xe_next
    #                 x0 = xe
    #                 z_train_sum, y_train_sum = etmpc.dataCat(
    #                     ze_train[1:], ye_train[1:])
    #                 return [0, z_train_sum, y_train_sum]


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train = np.load(args.datafile_z)
    y_train = np.load(args.datafile_y)
    traj_data = [np.zeros((1, 3)) for i in range(100)]
    trigger_data = [np.zeros(1) for i in range(100)]
    iter_num = 0
    iterLearning(args, vehicle, z_train, y_train, traj_data, trigger_data, iter_num)
    # while 1:
    #     iterdata = iterLearning(args, vehicle, z_train,
    #                             y_train, traj_data, trigger_data, iter_num)
    #     iter_num += 1
    #     if iterdata[0] == 1:
    #         print('Event-triggered mpc was completed.')
    #         break
    #     else:
    #         z_train = iterdata[1].copy()
    #         y_train = iterdata[2].copy()
    # for i in range(iter_num):
    #     np.save('../data/traj{}.npy'.format(i), iterdata[1][i])
    #     np.save('../data/trigger{}.npy'.format(i), iterdata[2][i])
    # np.save('../data/iter_num.npy', np.array([iter_num]))

# print(gpmodels.multigpr[0].X_train_)
    # print(gpmodels.multigpr[1].X_train_)
    # print(gpmodels.multigpr[2].X_train_)
    # print(gpmodels.gpr[0].y_train_)
    # print(gpmodels.gpr[1].y_train_)
    # print(gpmodels.gpr[2].y_train_)
    # print(y_train[:, 0])
    # z_test = np.array([[1, 1, 1, 1, 0]])
    # print(gpmodels.X_train_)
    # print(gpmodels.multigpr)
    # print(gpmodels.gpr.predict(z_test))
    # z_test = np.array([[1, 1, 1, 1, 0]])
    # means, stds = gpmodels.predict(z_test)
    # print(means)
    # print(stds)
    # print(np.sqrt(
    #     np.exp(gpmodels.gpr[0].kernel_.theta[0])))
    # print(np.sqrt(
    #     np.exp(gpmodels.gpr[1].kernel_.theta[0])))
    # print(np.sqrt(
    #     np.exp(gpmodels.gpr[2].kernel_.theta[0])))
    # print(np.diag(
    #     np.exp(gpmodels.gpr[0].kernel_.theta[1:1 + 3]) ** 2))
    # print(np.diag(
    #     np.exp(gpmodels.gpr[1].kernel_.theta[1:1 + 3]) ** 2))
    # print(np.diag(
    #     np.exp(gpmodels.gpr[2].kernel_.theta[1:1 + 3]) ** 2))
    # print(gpmodels.gpr)
    # print(gpmodels)
    # print(gpmodels.estimators_)
    # print(gpmodels[0])
    # print(gpmodels[1])
    # print(gpmodels[2])
    # print(gpmodels[3])
