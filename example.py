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

    # buffer_fac = 1.5
    # max_samples = int(np.ceil((buffer_fac * (1 + dim + int(dim * (dim + 1) / 2)))))
    # samples_per_iter = int(4 + np.floor(3 * np.log(dim)))

    # model_options_ls = {"max_samples": max_samples,
    #                     "output_weighting": "rank",  # "rank",
    #                     "whiten_input": True,
    #                     "normalize_features": True,
    #                     "normalize_output": "mean_std_clipped",  # "mean_std",  # "rank", "mean_std", "min_max",
    #                     "top_data_fraction": 0.5,
    #                     "min_clip_value": -3.,
    #                     "unnormalize_output": False,  # "rank",
    #                     "ridge_factor": 1e-12,
    #                     "limit_model_opt": True,
    #                     "refit": False,
    #                     "buffer_fac": buffer_fac,
    #                     "seed": None}

    more_config = {"epsilon": kl_bound,
                   "gamma": gamma,
                   "beta_0": entropy_loss_bound,
                   "n_samples": samples_per_iter,
                   "max_samples": max_samples,
                   # "buffer_fac": buffer_fac,
                   "min_data_fraction": 0.5
                   }

    x_start = 0.5 * np.random.randn(dim)
    init_sigma = 1

    # borrowing Rosenbrock from the cma package
    objective = nfreefunclasses[5](0, zerof=True, zerox=True)
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
