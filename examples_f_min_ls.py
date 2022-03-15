import numpy as np
from more.more_algo import fmin
from cma.bbobbenchmarks import nfreefunclasses
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MORE')
logger.setLevel("INFO")

dim = 10
max_iters = 1000

x_start = 0.5 * np.random.randn(dim)
init_sigma = 1

# borrowing objectives from the cma package
objective = nfreefunclasses[7](0, zerof=False, zerox=False)
objective.initwithsize(curshape=(1, dim), dim=dim)

f_val, x_opt = fmin(objective, x_start, init_sigma, max_iters, debug=True, minimize=True)

print(f_val, x_opt)
