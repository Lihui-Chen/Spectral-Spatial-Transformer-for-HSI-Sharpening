from .SRSolver import SRSolver


def create_solver(opt):
    solver = SRSolver(opt)
    return solver