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

if __name__ == "__main__":

    dim = 10
    max_iters = 2000
    minimize = True

    samples_per_iter = 300

    model_options_ls = QuadModelLS.get_default_config()

    more_config = MORE.get_default_config()

    x_start = 0.5 * np.random.randn(dim)
    init_sigma = 1

    # borrowing Rosenbrock from the cma package
    objective = nfreefunclasses[7](0, zerof=True, zerox=False)
    objective.initwithsize(curshape=(1, dim), dim=dim)

    search_dist = GaussFullCov(x_start, init_sigma * np.eye(dim))
    surrogate = QuadModelLS(dim, model_options_ls)

    more = MORE(dim, more_config, logger=logger)

    for i in range(max_iters):
        logger.info("Iteration {}".format(i))
        samples = search_dist.sample(samples_per_iter)

        rewards = objective(samples)
        if minimize:
            # negate, MORE maximizes, but we want to minimize
            rewards = -rewards

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
