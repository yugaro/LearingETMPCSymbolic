import numpy as np
import matplotlib.pyplot as plt


def plot_contractive_set(args):
    Q = np.load('../data/Q2.npy')
    Qind = np.load('../data/Qind2.npy')
    Xqlist = np.load('../data/Xqlist.npy')
    etax = np.load('../data/etax.npy')
    gamma = np.load('../data/gamma.npy')
    
    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(etax[0] * Qind[:, 0] + np.min(Xqlist, axis=1)[0], etax[0] * Qind[:, 1] + np.min(Xqlist, axis=1)[1],
    #              etax[2] * Qind[:, 2] + np.min(Xqlist, axis=1)[2], marker=".", linestyle='None')
    # ax.set_xlabel(r"$X$")
    # ax.set_ylabel(r"$Y$")
    # ax.set_zlabel(r"$\theta$")
    # fig.savefig('../image/contractive_set.pdf')
    # plt.show()


def plot_traj_trigger(args, vehicle):
    pathr = np.zeros((1, 3))
    for i in range(100):
        if i == 0:
            xr = np.array(args.xinit_r).reshape(-1)
            pathr = np.concatenate([pathr, xr.reshape(1, -1)], axis=0)
        ur = np.array([args.v_r, args.omega_r])
        xr_next = vehicle.realRK4(xr, ur)
        pathr = np.concatenate([pathr, xr.reshape(1, -1)])
        xr = xr_next

    iter_num = np.load('../data/iter_num.npy').item()
    cm = plt.cm.get_cmap('jet', iter_num)
    fig, ax = plt.subplots(1, 1)
    for i in range(iter_num):
        traj = np.load('../data/traj{}.npy'.format(i))
        trigger = np.load('../data/trigger{}.npy'.format(i))
        ax.scatter(traj[1, 0], traj[1, 1],
                   color=cm(i), marker='o', label='start')
        ax.plot(traj[1:, 0], traj[1:, 1], color=cm(i),
                label='iter:{0}, len:{1}'.format(i + 1, traj.shape[0] - 1))
        ax.scatter(traj[-1, 0], traj[-1, 1], color=cm(i), marker='*', label='goal')

        trigger_value = 0
        for j in range(trigger.shape[0] - 1):
            trigger_value += int(trigger[j + 1]) + 1
            ax.scatter(traj[trigger_value, 0], traj[trigger_value, 1],
                       color=cm(i), marker='x', label='trigger:{0}'.format(int(trigger[j + 1])))
    ax.plot(pathr[1:, 0], pathr[1:, 1], color='magenta', label='reference')
    ax.legend(bbox_to_anchor=(1.00, 1),
              loc='upper left', borderaxespad=0, ncol=2)
    fig.tight_layout()
    fig.savefig('../image/traj_trigger.pdf')

