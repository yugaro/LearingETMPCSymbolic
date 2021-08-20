import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def traj_safety_controller(args, vehicle, Qind, Cs):
    # xe = etax * Qind[-1, :] + np.min(Xqlist, axis=1)
    Xqlist = np.load('../data/Xqlist.npy')
    etax = np.load('../data/etax.npy')
    xe = np.array([0.05, 0.05, 0.05])
    xe_traj = xe.reshape(1, -1)
    xr = np.array(args.xinit_r)
    ur = np.array([args.v_r, args.omega_r])
    xr_traj = (xr - xe).reshape(1, -1)
    for i in range(1000):
        xpoint = (np.round((xe - np.min(Xqlist, axis=1)) / etax)).astype(np.int)
        indcs = np.where(np.all(Qind == xpoint, axis=1))[0][0]
        u = Cs[indcs, :]

        xe_next = vehicle.errRK4(xe, u)
        xr_next = vehicle.realRK4(xr, ur)

        xe_traj = np.concatenate(
            [xe_traj, xe_next.reshape(1, -1)], axis=0)
        xr_traj = np.concatenate(
            [xr_traj, (xr_next - xe_next).reshape(1, -1)], axis=0)

        xe = xe_next
        xr = xr_next
    return xe_traj, xr_traj


def plot_traj_safe(args, vehicle):
    iter_num = np.load('../data/iter_num2.npy').item()

    for i in range(iter_num):
        Q = np.load('../data/Q5{}.npy'.format(i))
        Qind = np.load('../data/Qind5{}.npy'.format(i))
        Cs = np.load('../data/Cs5{}.npy'.format(i))

        xe_traj, xr_traj = traj_safety_controller(args, vehicle, Qind, Cs)

        # fig, ax = plt.subplots(1, 1)
        # ax.plot(xe_traj[:, 0], xe_traj[:, 1])
        # fig.savefig('../image/traj_safe.pdf')

        fig, ax = plt.subplots(1, 1)
        ax.plot(xr_traj[:, 0], xr_traj[:, 1])
        fig.savefig('../image/traj_safer5{}.pdf'.format(i))


def plot_contractive_set(args, vehicle):
    Q = np.load('../data/Q3.npy')
    Qind = np.load('../data/Qind3.npy')
    Cs = np.load('../data/Cs.npy')
    Xqlist = np.load('../data/Xqlist.npy')
    etax = np.load('../data/etax.npy')
    gamma = np.load('../data/gamma.npy')

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(etax[0] * Qind[:, 0] + np.min(Xqlist, axis=1)[0], etax[0] * Qind[:, 1] + np.min(Xqlist, axis=1)[1],
                 etax[2] * Qind[:, 2] + np.min(Xqlist, axis=1)[2], marker=".", linestyle='None')
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$Y$")
    ax.set_zlabel(r"$\theta$")
    fig.savefig('../image/contractive_set.pdf')
    plt.show()


def plot_u_data(args, vehicle):
    iter_num = np.load('../data/iter_num2.npy').item()
    for i in range(iter_num):
        fig, ax = plt.subplots()
        u = np.load('../data/u2{}.npy'.format(i))
        trigger = np.load('../data/trigger2{}.npy'.format(i))
        ax.plot(u[:, 0] / args.v_max + u[:, 1] / args.omega_max, linewidth=2)

        trigger_value = 0
        for j in range(trigger.shape[0] - 1):
            trigger_value += int(trigger[j])
            ax.plot(trigger_value, u[trigger_value, 0] /
                    args.v_max + u[trigger_value, 1] / args.omega_max, marker='x', color='c', markersize=10)
            # ax.plot(trigger_value, u[trigger_value, 1], marker='x')
        ax.set_xlabel(r'Time', fontsize=15)
        ax.set_ylabel(
            r'$\frac{|v(t)|}{v_{\rm max}} + \frac{|\Omega(t)|}{\Omega_{\rm max}} $', fontsize=15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.grid(linestyle='dotted')
        fig.savefig('../image/utraj2{}.pdf'.format(i), bbox_inches='tight')


def plot_horizon(args, vehicle):
    iter_num = np.load('../data/iter_num3.npy').item()

    fig, ax = plt.subplots()
    linestyle_list = ['solid', 'dashed', 'dotted']
    k = 0
    for i in range(iter_num):
        # if i == 1 or i == 2 or i == 5:
        #     step_horizon_data = np.zeros((1, 2))
        #     horizon = np.load('../data/horizon2{}.npy'.format(i))
        #     trigger = np.load('../data/trigger2{}.npy'.format(i))

        #     step_pre = 0
        #     for j in range(trigger.shape[0]):
        #         step_pos = 0
        #         # ax.plot(step_pre, horizon[j], marker='*')
        #         step_horizon_data = np.concatenate(
        #             [step_horizon_data, np.array([[step_pre, horizon[j]]])], axis=0)
        #         step_pos += step_pre + int(trigger[j])
        #         # ax.plot(step_pos, horizon[j], marker='*')
        #         step_horizon_data = np.concatenate(
        #             [step_horizon_data, np.array([[step_pos, horizon[j]]])], axis=0)
        #         step_pre = step_pos
        #     step_horizon_data = np.concatenate(
        #         [step_horizon_data, np.array([[step_pre, horizon[-1]]])], axis=0)
        #     ax.plot(step_horizon_data[1:, 0],
        #             step_horizon_data[1:, 1], linewidth=2, label=r'iter{}'.format(i), linestyle=linestyle_list[k])
        #     k += 1
        step_horizon_data = np.zeros((1, 2))
        horizon = np.load('../data/horizon3{}.npy'.format(i))
        trigger = np.load('../data/trigger3{}.npy'.format(i))

        step_pre = 0
        for j in range(trigger.shape[0]):
            step_pos = 0
            # ax.plot(step_pre, horizon[j], marker='*')
            step_horizon_data = np.concatenate(
                [step_horizon_data, np.array([[step_pre, horizon[j]]])], axis=0)
            step_pos += step_pre + int(trigger[j])
            # ax.plot(step_pos, horizon[j], marker='*')
            step_horizon_data = np.concatenate(
                [step_horizon_data, np.array([[step_pos, horizon[j]]])], axis=0)
            step_pre = step_pos
        step_horizon_data = np.concatenate(
            [step_horizon_data, np.array([[step_pre, horizon[-1]]])], axis=0)
        ax.plot(step_horizon_data[1:, 0],
                step_horizon_data[1:, 1], linewidth=2, label=r'iter{}'.format(i))
    ax.set_xlabel(r'Time [step]', fontsize=15)
    ax.set_ylabel(r'Prediction horizon ($H_k$)', fontsize=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.hlines([1], 0, 40, color='magenta',
              linestyles='dashdot', label=r'$H_k$ = 1')
    ax.vlines([40], 0, 35, color='red', linestyles='dashed', label=r'deadline')
    ax.legend(bbox_to_anchor=(0, 0.5), loc='upper left', borderaxespad=0, ncol=1, fontsize=15)
    ax.grid(which='major', alpha=0.5, linestyle='dotted')
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30])
    # ax.set_xticklabels([0, 1, 5, 10, 15, 20, 25, 30])
    ax.set_xlim(0, 42)
    ax.set_ylim(0, 32)
    plt.show()
    fig.savefig('../image/horizon_traj3.pdf', bbox_inches='tight')


def plot_jcost(args, vehicle):
    iter_num = np.load('../data/iter_num2.npy').item()

    fig, ax = plt.subplots()

    for i in range(iter_num):
        step_jcost_data = np.zeros((1, 2))
        jcost = np.load('../data/jcost2{}.npy'.format(i))
        trigger = np.load('../data/trigger2{}.npy'.format(i))

        step_pre = 0
        for j in range(trigger.shape[0]):
            step_pos = 0
            # ax.plot(step_pre, horizon[j], marker='*')
            step_jcost_data = np.concatenate(
                [step_jcost_data, np.array([[step_pre, jcost[j]]])], axis=0)
            step_pos += step_pre + int(trigger[j])
            # ax.plot(step_pos, horizon[j], marker='*')
            step_jcost_data = np.concatenate(
                [step_jcost_data, np.array([[step_pos, jcost[j]]])], axis=0)
            step_pre = step_pos

        step_jcost_data = np.concatenate(
            [step_jcost_data, np.array([[step_pre, jcost[-1]]])], axis=0)
        ax.plot(step_jcost_data[1:, 0], step_jcost_data[1:, 1], linewidth=2)
        ax.set_xlabel(r'Time', fontsize=15)
        ax.set_ylabel(r'Cost', fontsize=15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.grid(linestyle='dotted')
    plt.show()
    # fig.savefig('../image/horizon.pdf', bbox_inches='tight')


def plot_traj_trigger(args, vehicle):
    iter_num = np.load('../data/iter_num2.npy').item()

    pathr = np.zeros((1, 3))
    for i in range(100):
        if i == 0:
            xr = np.array(args.xinit_r).reshape(-1)
            pathr = np.concatenate([pathr, xr.reshape(1, -1)], axis=0)
        ur = np.array([args.v_r, args.omega_r])
        xr_next = vehicle.realRK4(xr, ur)
        pathr = np.concatenate([pathr, xr.reshape(1, -1)])
        xr = xr_next

    for i in range(iter_num):
        fig, ax = plt.subplots(figsize=(6.0, 8.0))
        ax.plot(pathr[1:, 0], pathr[1:, 1], color='r',
                label='reference', linewidth=2)
        traj = np.load('../data/traj2{}.npy'.format(i))
        trigger = np.load('../data/trigger2{}.npy'.format(i))
        ax.scatter(traj[1, 0], traj[1, 1], marker='o',
                   label='start', s=100, c='b')
        ax.plot(traj[1:, 0], traj[1:, 1],
                label='iter:{0}, len:{1}'.format(i + 1, traj.shape[0] - 1), linewidth=2)
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   marker='*', label='goal', s=100, color='orange')

        trigger_value = 0
        for j in range(trigger.shape[0] - 1):
            trigger_value += int(trigger[j]) + 1
            ax.scatter(traj[trigger_value, 0], traj[trigger_value, 1],
                       color='c', marker='x', label='trigger:{0}'.format(int(trigger[j])), s=100)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left',
                  borderaxespad=0, ncol=1, fontsize=15)
        # plt.show()
        fig.savefig('../image/traj_trigger5{}.pdf'.format(i + 1),
                    bbox_inches='tight')


def plt_traj_all(args, vehicle):
    iter_num = np.load('../data/iter_num3.npy').item()

    pathr = np.zeros((1, 3))
    for i in range(22):
        if i == 0:
            xr = np.array(args.xinit_r).reshape(-1)
            pathr = np.concatenate([pathr, xr.reshape(1, -1)], axis=0)
        ur = np.array([args.v_r, args.omega_r])
        xr_next = vehicle.realRK4(xr, ur)
        pathr = np.concatenate([pathr, xr.reshape(1, -1)])
        xr = xr_next

    pathr2 = np.zeros((1, 3))
    for i in range(100):
        if i == 0:
            xr = np.array(args.xinit_r).reshape(-1)
            pathr2 = np.concatenate([pathr2, xr.reshape(1, -1)], axis=0)
        ur = np.array([args.v_r, args.omega_r])
        xr_next = vehicle.realRK4(xr, ur)
        pathr2 = np.concatenate([pathr2, xr.reshape(1, -1)])
        xr = xr_next

    for i in range(iter_num):
        traj = np.load('../data/traj3{}.npy'.format(i))
        trigger = np.load('../data/trigger3{}.npy'.format(i))
        if traj.shape[0] != 1:
            fig, ax = plt.subplots(figsize=(7.0, 8.0))
            ax.plot(pathr[1:, 0], pathr[1:, 1], color='red',
                    label=r'leader', linewidth=2, linestyle='dashdot', zorder=1)

            ax.scatter(traj[1, 0], traj[1, 1], marker='o',
                       label=r'start', s=100, color='navy', zorder=100)
            ax.plot(traj[1:args.horizon + trigger.shape[0] * 2 + 2, 0], traj[1:args.horizon + trigger.shape[0] * 2 + 2, 1],
                    label=r'ETMPC', linewidth=2, color='royalblue', zorder=1)
            if traj.shape[0] != args.horizon + trigger.shape[0] * 2 + 1:
                ax.plot(traj[args.horizon + trigger.shape[0] * 2 + 1:, 0], traj[args.horizon + trigger.shape[0] * 2 + 1:, 1],
                        label=r'symbolic control', linewidth=2, color='lime', linestyle='dashed', zorder=1)
            # ax.plot(traj[1:args.horizon + trigger.shape[0] * 2 + 2, 0], traj[1:args.horizon + trigger.shape[0] * 2 + 2, 1],
            #         label='iter:{0}, len:{1}'.format(i + 1, traj.shape[0] - 1), linewidth=2, color='blue')
            # ax.plot(traj[args.horizon + trigger.shape[0] * 2 + 1:, 0], traj[args.horizon + trigger.shape[0] * 2 + 1:, 1],
            #         label='iter:{0}, len:{1}'.format(i + 1, traj.shape[0] - 1), linewidth=2, color='green')
            ax.scatter(traj[-1, 0], traj[-1, 1],
                       marker='*', label='end', s=200, color='gold', zorder=100)

            trigger_value = 0
            trigger_data = np.zeros((1, 2))
            for j in range(trigger.shape[0] - 1):
                trigger_value += int(trigger[j]) + 1
                trigger_data = np.concatenate([trigger_data, traj[trigger_value, :2].reshape(1, -1)])
            ax.scatter(trigger_data[1:, 0], trigger_data[1:, 1], color='coral',
                       marker='x', label=r'trigger', s=100, zorder=100)
            ax.add_patch(patches.Rectangle(xy=(-3, -3), width=1,
                                           height=1, color='forestgreen', fill=False, linewidth=2, hatch='\\', label=r'$\mathcal{X}_{\rm init}$',
                                           zorder=0))
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.legend(bbox_to_anchor=(1, 0), loc='lower right',
                      borderaxespad=0, ncol=1, fontsize=15)
            ax.set_xlim(-3.2, 1.5)
            ax.set_ylim(-3.2, 2.5)
            ax.set_xlabel(r'x', fontsize=20)
            ax.set_ylabel(r'y', fontsize=20)
            ax.grid(which='major', alpha=0.5, linestyle='dotted')
            fig.savefig('../image/traj_trigger7{}.pdf'.format(i),
                        bbox_inches='tight')

            # print('aaaaa')
            # print(pathr2[1:traj.shape[0], 2])
            # print('bbbbbbbbb')
            # print(traj[1:, 2])
            # plt.close()
            # fig, ax = plt.subplots()
            # ax.plot(np.abs(
            #     ((pathr2[1:traj.shape[0], 2] - traj[1:, 2]) % np.pi) * 360 / np.pi))
            # plt.show()
            # plt.close()


def plt_traj_xe(args, vehicle):
    iter_num = np.load('../data/iter_num3.npy').item()
    for i in range(iter_num):
        xe_traj = np.load('../data/xe_traj3{}.npy'.format(i))
        trigger = np.load('../data/trigger3{}.npy'.format(i))
        fig, ax = plt.subplots()
        ax.plot(np.abs(xe_traj[1:, 0]), linewidth=2)
        ax.plot(np.abs(xe_traj[1:, 1]), linewidth=2)
        ax.plot(np.abs(xe_traj[1:, 2]), linewidth=2)
        # plt.show()
        fig.savefig('../image/xe_traj7{}.pdf'.format(i))

        # trigger_value = 0
        # trigger_theta_data = np.zeros(1)
        # trigger_data = np.zeros(1)
        # for j in range(trigger.shape[0] - 1):
        #     trigger_value += int(trigger[j])
        #     trigger_theta_data = np.concatenate(
        #         [trigger_theta_data, np.array([xe_traj[trigger_value, 2]])])
        #     trigger_data = np.concatenate([trigger_data, np.array([trigger_value])])
        # ax.scatter(trigger_data[1:], np.abs(trigger_theta_data[1:]), color='coral',
        #            marker='x', label=r'trigger', s=100, zorder=100)
        # ax.set_xlabel(r'Time (step)', fontsize=15)
        # ax.set_ylabel(r'$|\theta - \theta_r|$ [rad]', fontsize=15)
        # ax.grid(which='major', alpha=0.5, linestyle='dotted')
        # fig.savefig('../image/rad_traj2{}.pdf'.format(i))

# ax.scatter(traj[trigger_value, 0], traj[trigger_value, 1],
#            color='magenta', marker='x', label='Trigger', s=100)
# ax.scatter(traj[trigger_value, 0], traj[trigger_value, 1],
#            color='c', marker='x', label='Trigger:{0}'.format(int(trigger[j])), s=100)
# np.random.seed(0)
# Z = np.random.rand(10, 10)
# Zm = Z * 0
# Z[3:5, 5:8] = 1
# fig, ax = plt.subplots(figsize=(9, 7))
# ax1 = ax.pcolormesh(Z)

# Zm = np.ma.masked_where(Z != 1, Z)
# print(Zm)
# ax2 = ax.pcolor(Zm, hatch='/', edgecolor='grey',
#                 facecolor='none', linewidth=0)
# plt.show()
