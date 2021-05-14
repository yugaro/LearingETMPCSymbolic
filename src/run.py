import torch
from component.blueprint.set_args import set_args
from model.vehicle import Vehicle
from model import gp
from component.controller.symbolic import Symbolic


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train = torch.load(args.datafile_z)
    y_train = torch.load(args.datafile_y)

    # gp and safety game
    gpmodels, likelihoods, covs, noises = gp.train(args, z_train, y_train)
    symmodel = Symbolic(args, gpmodels, covs, noises)
    symmodel.safeyGame()
