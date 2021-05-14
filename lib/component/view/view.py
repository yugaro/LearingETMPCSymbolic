import torch
import matplotlib.pyplot as plt


def plot_traj_trigger(args, vehicle):
    pathr = torch.zeros(1, 3)
    for i in range(100):
        if i == 0:
            xr = torch.tensor(args.xinit_r).reshape(-1)
            pathr = torch.cat([pathr, xr.reshape(1, -1)])
        ur = torch.tensor([args.v_r, args.omega_r])
        xr_next = vehicle.realRK4(xr, ur)
        pathr = torch.cat([pathr, xr.reshape(1, -1)])
        xr = xr_next

    iter_num = torch.load('../data/iter_num.pt').item()
    cm = plt.cm.get_cmap('jet', iter_num)
    fig, ax = plt.subplots(1, 1)
    for i in range(iter_num):
        traj = torch.load('../data/traj{}.pt'.format(i))
        trigger = torch.load('../data/trigger{}.pt'.format(i))
        ax.plot(traj[1:, 0], traj[1:, 1], color=cm(i),
                label='iter:{0}, len:{1}'.format(i + 1, traj.shape[0] - 1))

        ax.scatter(traj[1, 0], traj[1, 1],
                   color=cm(i), marker='o', label='start')
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   color=cm(i), marker='*', label='goal')
        for j in range(trigger.shape[0] - 1):
            ax.scatter(traj[trigger[j + 1].int(), 0], traj[trigger[j + 1].int(), 1],
                       color=cm(i), marker='x', label='trigger:{0}'.format(trigger[j + 1].int()))
    ax.plot(pathr[1:, 0], pathr[1:, 1], color='orange', label='reference')
    ax.legend(bbox_to_anchor=(1.00, 1),
              loc='upper left', borderaxespad=0, ncol=2)
    fig.tight_layout()
    fig.savefig('../image/traj_trigger.pdf')
