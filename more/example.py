import numpy as np
from more.gauss_full_cov import GaussFullCov
from more.quad_model import QuadModelLS, QuadModelSubBLR
from more.more_algo import MORE
from more.sample_db import SimpleSampleDatabase
from cma.bbobbenchmarks import nfreefunclasses
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MORE')
logger.setLevel("INFO")

dim = 15
max_iters = 1000
max_samples = 150  # 500 for ls model
samples_per_iter = 15
kl_bound = 1
gamma = 0.99
entropy_loss_bound = 0.1
minimize = True

model_options_sub = {"normalize_features": True,
                     }

model_options_ls = {"whiten_input": False,
                    "normalize_features": True,
                    "normalize_output": False,
                    "unnormalize_output": False,
                    "output_weighting": False,
                    "ridge_factor": 1e-5}

more_config = {"epsilon": kl_bound,
               "gamma": gamma,
               "beta_0": entropy_loss_bound}

x_start = 0.5 * np.random.randn(dim)
init_sigma = 1

# borrowing Rosenbrock from the cma package
objective = nfreefunclasses[7](0, zerof=True, zerox=True)
objective.initwithsize(curshape=(1, dim), dim=dim)

sample_db = SimpleSampleDatabase(max_samples)

search_dist = GaussFullCov(x_start, init_sigma * np.eye(dim))
# surrogate = QuadModelLS(dim, model_options_ls)
surrogate = QuadModelSubBLR(dim, model_options_sub)

more = MORE(dim, more_config, logger=logger)

for i in range(max_iters):
    logger.info("Iteration {}".format(i))
    new_samples = search_dist.sample(samples_per_iter)

    new_rewards = objective(new_samples)
    if minimize:
        # negate, MORE maximizes, but we want to minimize
        new_rewards = -new_rewards

    sample_db.add_data(new_samples, new_rewards)

    samples, rewards = sample_db.get_data()

    success = surrogate.fit(samples, rewards, search_dist, )
    if not success:
        continue

    new_mean, new_cov, success = more.step(search_dist, surrogate)

    if success:
        try:
            search_dist.update_params(new_mean, new_cov)
        except Exception as e:
            print(e)

    lam = objective(search_dist.mean.T)
    logger.info("Loss at mean {}".format(lam))
    logger.info("Change KL {}, Entropy {}".format(more._kl, search_dist.entropy))
    logger.info("Dist to x_opt {}".format(np.linalg.norm(objective._xopt - search_dist.mean.flatten())))

    dist_to_opt = np.abs((objective._fopt - lam))
    logger.info("Dist to f_opt {}".format(dist_to_opt))
    logger.info("-------------------------------------------------------------------------------")

    if dist_to_opt < 1e-8:
        break
