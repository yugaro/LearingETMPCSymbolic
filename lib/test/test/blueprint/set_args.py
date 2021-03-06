import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts", type=float, default=0.3)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--xinit_r", type=float, default=[0., 0., 0.])
    parser.add_argument("--v_r", type=float, default=1.0)
    parser.add_argument("--omega_r", type=float, default=1.0)
    parser.add_argument("--v_max", type=float, default=3.0)
    parser.add_argument("--omega_max", type=float, default=3.25)
    parser.add_argument("--Kx", type=float, default=0.3)
    parser.add_argument("--Ky", type=float, default=0.1)
    parser.add_argument("--Ktheta", type=float, default=0.3)
    parser.add_argument("--etax_param", type=float, default=0.01)
    parser.add_argument("--etau", type=float, default=0.25)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--gpudate_num", type=int, default=500)
    parser.add_argument("--b", type=float,
                        default=[1.0, 1.0, 1.0])
    parser.add_argument("--Xsafe", type=float,
                        default=[[-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25]])
    parser.add_argument("--gamma_params", type=float, default=600)
    parser.add_argument("--mpc_type", type=str, default='discrete')
    parser.add_argument("--weightx", type=float,
                        default=[10.0, 1.0, 10.0])
    parser.add_argument('--datafile_z', type=str,
                        default='./data/z_train.npy')
    parser.add_argument('--datafile_y', type=str,
                        default='./data/y_train.npy')
    return parser.parse_args()
