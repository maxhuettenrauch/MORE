import numpy as np
from more.gauss_full_cov import GaussFullCov
from more.quad_model import QuadModelSubBLR
from more.more_algo import MORE
from more.sample_db import SimpleSampleDatabase
from cma.bbobbenchmarks import nfreefunclasses
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MORE')
logger.setLevel("INFO")

if __name__ == "__main__":

    dim = 15
    max_iters = 1000
    kl_bound = 1
    gamma = 0.99
    entropy_loss_bound = 0.01
    minimize = True

    max_samples = 150
    samples_per_iter = 15

    model_options_sub = {"normalize_features": True,
                         "normalize_output": None,  # "mean_std",  # "mean_std_clipped",  # "rank", "mean_std", "min_max",
                         }

    more_config = MORE.get_default_config()

    # borrowing Rosenbrock from the cma package
    objective = nfreefunclasses[7](0, zerof=True, zerox=True)
    objective.initwithsize(curshape=(1, dim), dim=dim)

    sample_db = SimpleSampleDatabase(max_samples)

    x_start = objective.x_opt + 0.1 * np.random.randn(dim)
    init_sigma = 1

    search_dist = GaussFullCov(x_start, init_sigma ** 2 * np.eye(dim))
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

        success = surrogate.update_quad_model(samples, rewards, search_dist, )
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
