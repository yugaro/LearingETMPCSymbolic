import argparse
import numpy as np

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts", type=float, default=0.4)
    parser.add_argument("--noise", type=float, default=0.001)
    parser.add_argument("--v_r", type=float, default=1.0)
    parser.add_argument("--omega_r", type=float, default=1.0)
    parser.add_argument("--v_max", type=float, default=2.0)
    parser.add_argument("--omega_max", type=float, default=2.0)
    parser.add_argument("--Kx", type=float, default=1.0)
    parser.add_argument("--Ky", type=float, default=1.0)
    parser.add_argument("--Ktheta", type=float, default=1.0)
    parser.add_argument("--etax", type=float, default=0.1)
    parser.add_argument("--etau", type=float, default=0.25)
    parser.add_argument("--data_num", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--gpudate_num", type=int, default=100)
    parser.add_argument("--b", type=float,
                        default=np.array([1.0, 1.0, 1.0]))
    parser.add_argument("--Xsafe", type=float,
                        default=np.array([[-1.2, 1.2], [-1.2, 1.2], [-1.2, 1.2]]))
    parser.add_argument("--gamma_params", type=float, default=[80, 80, 100])
    parser.add_argument('--datafile_z', type=str,
                        default='../data/z_train.pt')
    parser.add_argument('--datafile_y', type=str,
                        default='../data/y_train.pt')
    return parser.parse_args()
