from blueprint.set_args import set_args
from model.vehicle import Vehicle
from view.view import plot_traj_trigger
from view.view import plot_contractive_set
from view.view import plot_traj_safe
from view.view import plot_u_data
from view.view import plot_horizon
from view.view import plot_jcost
from view.view import plt_traj_all
from view.view import plt_traj_xe
from view.view import plot_3d


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    # plot_traj_trigger(args, vehicle)
    # plot_u_data(args, vehicle)
    # plot_horizon(args, vehicle)
    # plot_jcost(args, vehicle)
    # plot_contractive_set(args, vehicle)
    # traj_safety_controller(args, vehicle)
    # plot_traj_safe(args, vehicle)
    # plt_traj_all(args, vehicle)
    # plt_traj_xe(args, vehicle)
    plot_3d(args)
