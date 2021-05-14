from blueprint.set_args import set_args
from model.vehicle import Vehicle
from view.view import plot_traj_trigger

if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    plot_traj_trigger(args, vehicle)
