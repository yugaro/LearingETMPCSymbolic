import torch
import numpy as np
from blueprint.set_args import set_args
from model.vehicle import Vehicle
from model import gp
from controller.symbolic import Symbolic
from controller.etmpc import ETMPC
torch.manual_seed(1)
np.random.seed(3)


def iterLearning(args, vehicle, z_train, y_train, traj_data, trigger_data, iter_num):
    # gp and safety game
    gpmodels, likelihoods, covs, noises = gp.train(args, z_train, y_train)
    symmodel = Symbolic(args, gpmodels, covs, noises)
    # print(symmodel.ellout_max)
    # print(symmodel.ellin_max)
    # print(symmodel.epsilon)
    # print(symmodel.gamma)
    symmodel.safeyGame()

    # etmpc
    etmpc = ETMPC(args, gpmodels, likelihoods, covs, noises, symmodel.gamma)
    mpc, simulator, estimator = etmpc.setUp()
    flag_asm = 0
    while 1:
        ze_train = torch.zeros(1, 5)
        ye_train = torch.zeros(1, 3)
        x0 = np.array([np.random.rand(1) + 3.8, np.random.rand(1) + 3.8, 2 * np.random.rand(1) - 1])
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
                        xe = torch.from_numpy(x0).reshape(-1)
                        xr = torch.tensor(args.xinit_r).reshape(-1)
                        traj_data[iter_num] = torch.cat(
                            [traj_data[iter_num], (xr - xe).reshape(1, -1)])
                    xstar = np.array(
                        mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)

                    if etmpc.triggerF(xstar, xe, trigger_values[i, :]):
                        if i == args.horizon:
                            return [1, traj_data, trigger_data]
                        u = torch.from_numpy(
                            np.array(mpc.opt_x_num['_u', i, 0])).reshape(-1)
                        ur = torch.tensor([args.v_r, args.omega_r])

                        xe_next = vehicle.errRK4(xe, u)
                        xr_next = vehicle.realRK4(xr, ur)

                        traj_data[iter_num] = torch.cat(
                            [traj_data[iter_num], (xr_next - xe_next).reshape(1, -1)])

                        ze_train, ye_train = etmpc.learnD(
                            xe, u, xe_next, ze_train, ye_train)

                        xe = xe_next
                        xr = xr_next
                    else:
                        trigger_time = i
                        x0 = xe.to('cpu').detach().numpy().reshape(-1, 1)
                        print('trigger:', trigger_time)
                        trigger_data[iter_num] = torch.cat(
                            [trigger_data[iter_num], torch.tensor([trigger_time])])
                        break
            else:
                if flag_asm == 0:
                    print('assumption is not hold.')
                    break
                elif flag_asm == 1:
                    for i in range(args.horizon - trigger_time):
                        if i == 0:
                            xe = torch.from_numpy(x0).reshape(-1)
                            traj_data[iter_num] = torch.cat(
                                [traj_data[iter_num], (xr - xe).reshape(1, -1)])
                        u = ulist_pre[trigger_time + i]
                        ur = torch.tensor([args.v_r, args.omega_r])

                        xe_next = vehicle.errRK4(
                            xe, u)
                        xr_next = vehicle.realRK4(xr, ur)

                        traj_data[iter_num] = torch.cat(
                            [traj_data[iter_num], (xr_next - xe_next).reshape(1, -1)])

                        ze_train, ye_train = etmpc.learnD(
                            xe, ulist_pre[trigger_time + i], xe_next, ze_train, ye_train)
                        xe = xe_next
                    x0 = xe.to('cpu').detach().numpy()
                    z_train_sum, y_train_sum = etmpc.dataCat(
                        ze_train[1:], ye_train[1:])
                    return [0, z_train_sum.float(), y_train_sum.float()]


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train = torch.load(args.datafile_z)
    y_train = torch.load(args.datafile_y)
    traj_data = [torch.zeros(1, 3) for i in range(100)]
    trigger_data = [torch.zeros(1) for i in range(100)]
    iter_num = 0
    while 1:
        iterdata = iterLearning(args, vehicle, z_train,
                                y_train, traj_data, trigger_data, iter_num)
        iter_num += 1
        if iterdata[0] == 1:
            print('Event-triggered mpc was completed.')
            break
        else:
            z_train = iterdata[1].clone()
            y_train = iterdata[2].clone()

    for i in range(iter_num):
        torch.save(iterdata[1][i], '../data/traj{}.pt'.format(i))
        torch.save(iterdata[2][i], '../data/trigger{}.pt'.format(i))
    torch.save(torch.tensor([iter_num]), '../data/iter_num.pt')
