from blueprint.set_args import set_args
from model.vehicle import Vehicle
# from view.view import plot_traj_trigger
from view.view import plot_contractive_set
from view.view import plot_traj_safe
if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    # plot_traj_trigger(args, vehicle)
    plot_contractive_set(args, vehicle)
    # traj_safety_controller(args, vehicle)
    # plot_traj_safe(args, vehicle)
