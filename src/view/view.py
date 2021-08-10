import numpy as np
import matplotlib.pyplot as plt


def traj_safety_controller(args, vehicle):
    Q = np.load('../data/Q3.npy')
    Qind = np.load('../data/Qind3.npy')
    Xqlist = np.load('../data/Xqlist.npy')
    etax = np.load('../data/etax.npy')
    Cs = np.load('../data/Cs3.npy')

    xe = etax * Qind[-1, :] + np.min(Xqlist, axis=1)
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
    xe_traj, xr_traj = traj_safety_controller(args, vehicle)
    gamma = np.load('../data/gamma.npy')

    fig, ax = plt.subplots(1, 1)
    ax.plot(xe_traj[:, 0], xe_traj[:, 1])
    fig.savefig('../image/traj_safe.pdf')

    fig, ax = plt.subplots(1, 1)
    ax.plot(xr_traj[:, 0], xr_traj[:, 1])
    fig.savefig('../image/traj_safer.pdf')


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
    iter_num = np.load('../data/iter_num.npy').item()
    fig, ax = plt.subplots()
    for i in range(iter_num):
        u = np.load('../data/u{}.npy'.format(i))
        trigger = np.load('../data/trigger{}.npy'.format(i))
        # ax.plot(u[:, 0])
        # ax.plot(u[:, 1])
        ax.plot(u[:, 0] / args.v_max + u[:, 1] / args.omega_max, linewidth=2)

        trigger_value = 0
        for j in range(trigger.shape[0] - 1):
            trigger_value += int(trigger[j])
            ax.plot(trigger_value, u[trigger_value, 0] /
                    args.v_max + u[trigger_value, 1] / args.omega_max, marker='x', color='c', markersize=10)
            # ax.plot(trigger_value, u[trigger_value, 1], marker='x')
        ax.set_xlabel(r'Time', fontsize=15)
        ax.set_ylabel(r'$\frac{|v(t)|}{v_{\rm max}} + \frac{|\Omega(t)|}{\Omega_{\rm max}} $', fontsize=15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.grid(linestyle='dotted')
        fig.savefig('../image/utraj.pdf', bbox_inches='tight')


def plot_horizon(args, vehicle):
    iter_num = np.load('../data/iter_num2.npy').item()

    fig, ax = plt.subplots()
    for i in range(iter_num):
        step_horizon_data = np.zeros((1, 2))
        horizon = np.load('../data/horizon2{}.npy'.format(i))
        trigger = np.load('../data/trigger2{}.npy'.format(i))

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
        ax.plot(step_horizon_data[1:, 0], step_horizon_data[1:, 1], linewidth=2)
        ax.hlines([1], 0, 40, color='magenta', linestyles='dashed')
        ax.set_xlabel(r'Time', fontsize=15)
        ax.set_ylabel(r'Prediction horizon', fontsize=15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.grid(linestyle='dotted')
    plt.show()
    # fig.savefig('../image/horizon.pdf', bbox_inches='tight')


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
        ax.scatter(traj[1, 0], traj[1, 1], marker='o', label='start', s=100, c='b')
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
        ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', borderaxespad=0, ncol=1, fontsize=15)
        # plt.show()
        fig.savefig('../image/traj_trigger5{}.pdf'.format(i + 1), bbox_inches='tight')
