import numpy as np
import nlopt
from types import SimpleNamespace

from more.quad_model import QuadModelLS


class MORE:
    @classmethod
    def get_default_config(cls, dim=None):

        more_config = {"epsilon": 0.5,
                       "gamma": -0.99,
                       "beta_0": 0.1,
                       "h_0": -200,
                       "eta_0": 1,
                       "omega_0": 1,
                       }

        if dim is not None:
            samples_per_iter = int(4 + np.floor(3 * np.log(dim)))

            more_config.update({"samples_per_iter": samples_per_iter,
                                "h_0": -25 * dim
                                })

        return more_config

    def __init__(self, dim, config_dict, logger=None):

        self.debug = False
        self.logger = logger
        self.dim = dim
        self.options = SimpleNamespace(**config_dict)

        self.beta = None
        self.beta_0 = self.options.beta_0
        self.gamma = self.options.gamma
        self.epsilon = self.options.epsilon
        self.eta_0 = self.options.eta_0
        self.omega_0 = self.options.omega_0
        self.h_0 = self.options.h_0  # minimum entropy of the search distribution

        # Setting up optimizer
        opt = nlopt.opt(nlopt.LD_LBFGS, 2)

        opt.set_lower_bounds((1e-20, 1e-20))
        opt.set_upper_bounds((np.inf, np.inf))

        opt.set_ftol_abs(1e-12)
        opt.set_ftol_rel(1e-12)
        opt.set_xtol_abs(1e-12)
        opt.set_xtol_rel(1e-12)
        opt.set_maxeval(1000)
        opt.set_maxtime(5 * 60 * 60)

        def opt_func(x, grad):
            g = self.dual_and_grad(x, grad)
            if np.isinf(g):
                opt.set_lower_bounds((float(x[0]), 1e-20))
            return float(g.flatten())

        opt.set_min_objective(opt_func)

        self.opt = opt
        self._grad_bound = 1e-5

        # constant values
        self._log_2_pi_k = self.dim * (np.log(2 * np.pi))
        self._entropy_const = self.dim * (np.log(2 * np.pi) + 1)

        # cached values
        self._eta = 1
        self._omega = 1
        self._old_term = None
        self._old_dist = None
        self._current_model = None
        self._dual = np.inf
        self._grad = np.zeros(2)
        self._kl = np.inf
        self._kl_mean = np.inf
        self._kl_cov = np.inf
        self._new_entropy = np.inf
        self._new_mean = None
        self._new_cov = None

    def new_natural_params(self, eta, omega, old_dist, model):
        reward_quad, reward_lin = model.get_model_params()
        lin = (eta * old_dist.nat_mean + reward_lin) / (eta + omega)  # linear parameter of pi
        prec = (eta * old_dist.prec + reward_quad) / (eta + omega)  # precision of pi
        return lin, prec

    def get_beta(self, old_dist):
        if self.gamma > 0:
            beta = self.gamma * (old_dist.entropy - self.h_0) + self.h_0
        else:
            if old_dist.entropy > self.h_0:
                beta = old_dist.entropy - self.beta_0
            else:
                beta = self.h_0
        return beta

    def step(self, old_dist, surrogate):
        """
        Given an old distribution and a model object, perform one MORE iteration
        :param old_dist: Distribution object
        :param surrogate: quadratic model object
        :return: new distribution parameters and success variables
        """

        success = False

        self.beta = self.get_beta(old_dist)  # set entropy constraint
        # self.eta_0 = self._eta
        # self.omega_0 = self._omega

        self._old_term = old_dist.log_det + old_dist.mean.T @ old_dist.nat_mean
        self._old_dist = old_dist
        self._current_model = surrogate

        for i in range(10):
            self.opt.set_lower_bounds((1e-20, 1e-20))
            eta, omega, success = self.dual_opt()

            if success:
                break

            self.eta_0 *= 2
            self.omega_0 *= 2

        if success:
            self.eta_0 = self._eta
            self.omega_0 = self._omega
            return self._new_mean, self._new_cov, True
        else:
            # logger.debug("Optimization unsuccessful")
            self.eta_0 = self.options.eta_0
            self.omega_0 = self.options.omega_0
            return old_dist.mean, old_dist.cov, False

    def dual_opt(self):
        success_kl = False
        success_entropy = False
        try:
            eta, omega = self.opt.optimize(np.hstack([self.eta_0, self.omega_0]))
            opt_val = self.opt.last_optimum_value()
            result = self.opt.last_optimize_result()
        except (RuntimeError, nlopt.ForcedStop, nlopt.RoundoffLimited) as e:
            # logger.debug(e)
            if (np.sqrt(self._grad[0] ** 2 + self._grad[1] ** 2) < self._grad_bound) or \
                    (self._eta < 1e-10 and np.abs(self._grad[1]) < self._grad_bound) or \
                    (self._omega < 1e-10 and np.abs(self._grad[0]) < self._grad_bound):
                eta = self._eta
                omega = self._omega
                result = 1
                opt_val = self._dual
            else:
                eta = -1
                omega = -1
                result = 5
                opt_val = self._dual

        except (ValueError, np.linalg.LinAlgError) as e:
            # self.logger.debug("Error in mean optimization: {}".format(e))
            result = 5
            opt_val = self._dual
            eta = -1
            omega = -1

        except Exception as e:
            raise e

        finally:
            if result in (1, 2, 3, 4) and ~np.isinf(opt_val):
                if self._kl < 1.1 * self.epsilon:
                    success_kl = True

                if self._new_entropy > 0:
                    if self._new_entropy > 0.9 * self.beta:
                        success_entropy = True
                else:
                    if self._new_entropy > 1.1 * self.beta:
                        success_entropy = True

        # self.logger.debug("epsilon = {}, beta = {}".format(self.epsilon, self.beta))
        # self.logger.debug("eta = {}, omega = {}".format(eta, omega))
        success = success_kl and success_entropy

        return eta, omega, success

    def dual_and_grad(self, x, grad):
        eta = x[0]
        omega = x[1]
        self._eta = eta
        self._omega = omega

        mu_q = self._old_dist.mean
        new_lin, new_prec = self.new_natural_params(eta, omega, self._old_dist, self._current_model)

        try:
            chol_new_prec = np.linalg.cholesky(new_prec)
            inv_chol_new_prec = np.linalg.inv(chol_new_prec)
            new_cov = inv_chol_new_prec.T @ inv_chol_new_prec
            chol_new_cov = np.linalg.cholesky(new_cov)
            new_mean = new_cov @ new_lin

            # compute log(det(Sigma_pi))
            new_log_det = 2 * np.sum(np.log(np.diag(chol_new_cov)))

            g = eta * self.epsilon - omega * self.beta \
                + 0.5 * (omega * self._log_2_pi_k
                         - eta * self._old_term
                         + (eta + omega) * (new_log_det + new_mean.T @ new_lin)
                         )

            maha_dist = np.sum((self._old_dist.chol_prec.T @ (mu_q - new_mean)) ** 2)
            trace_term = np.sum(np.square(self._old_dist.chol_prec.T @ chol_new_cov))

            kl = 0.5 * (maha_dist
                        + self._old_dist.log_det
                        - new_log_det
                        + trace_term
                        - self.dim)

            entropy = 0.5 * (new_log_det + self._log_2_pi_k + self.dim)

            d_g_d_eta = self.epsilon - float(kl)
            d_g_d_omega = entropy - self.beta

            grad[0] = d_g_d_eta
            grad[1] = d_g_d_omega

        except np.linalg.LinAlgError:
            g = np.atleast_1d(np.inf)
            grad[0] = -0.1
            grad[1] = 0.
            kl = np.inf
            entropy = -np.inf
            new_mean = self._old_dist.mean
            new_cov = self._old_dist.cov
            maha_dist = 0
            trace_term = self.dim
            new_log_det = self._old_dist.log_det

        self._dual = g
        self._grad = grad
        self._kl = kl
        self._kl_mean = 0.5 * maha_dist
        self._kl_cov = 0.5 * (trace_term + self._old_dist.log_det - new_log_det - self.dim)
        self._new_entropy = entropy
        self._entropy_diff = self._old_dist.entropy - entropy
        self._new_mean = new_mean
        self._new_cov = new_cov
        return g


# %%%%%%%%%%%%%%%%%%%%%%%%%% fmin function interfaces %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def fmin(objective,
         x_start,
         init_sigma,
         n_iters,
         target_dist=1e-8,
         algo_config: dict = {},
         model_config: dict = {},
         sample_db_config: dict = {},
         budget=None,
         debug=False,
         minimize=False):

    from more.sample_db import SimpleSampleDatabase
    from more.quad_model import QuadModelLS
    from more.gauss_full_cov import GaussFullCov
    import attrdict as ad
    import logging

    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    logger = logging.getLogger('MORE')
    if debug:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    dim = len(x_start)
    default_algo_config = MORE.get_default_config(dim)
    default_algo_config.update(algo_config)
    default_model_config = QuadModelLS.get_default_config()
    default_model_config.update(model_config)
    default_sample_db_config = SimpleSampleDatabase.get_default_config(dim)
    default_sample_db_config.update(sample_db_config)

    algo_config = ad.AttrDict(default_algo_config)
    model_config = ad.AttrDict(default_model_config)
    sample_db_config = ad.AttrDict(default_sample_db_config)

    sample_db = SimpleSampleDatabase(sample_db_config.max_samples)

    search_dist = GaussFullCov(x_start, init_sigma * np.eye(dim))
    surrogate = QuadModelLS(dim, model_config)

    more = MORE(dim, algo_config, logger=logger)

    if budget is None:
        budget = np.inf
    it = 0
    obj_evals = 0
    dist_to_opt = 1e10

    while dist_to_opt > target_dist and it < n_iters and obj_evals < budget:
        logger.info("Iteration {}".format(it))
        new_samples = search_dist.sample(algo_config.samples_per_iter)
        obj_evals += algo_config.samples_per_iter

        new_rewards = objective(new_samples)
        if minimize:
            # negate, MORE maximizes, but we want to minimize
            new_rewards = -new_rewards

        sample_db.add_data(new_samples, new_rewards)

        if len(sample_db.data_x) < model_config.min_data_frac * surrogate.model_dim:
            continue

        samples, rewards = sample_db.get_data()

        success = surrogate.update_quad_model(samples, rewards, search_dist, )
        if not success:
            continue

        new_mean, new_cov, success = more.step(search_dist, surrogate)

        search_dist.update_params(new_mean, new_cov)

        lam = objective(search_dist.mean.T)
        logger.debug("Loss at mean {}".format(lam))
        logger.debug("Change KL cov {}, Change Entropy {}".format(more._kl_cov, more.beta))
        logger.debug("Dist to x_opt {}".format(np.linalg.norm(objective._xopt - search_dist.mean.flatten())))

        dist_to_opt = np.abs((objective._fopt - lam))
        logger.debug("Dist to f_opt {}".format(dist_to_opt))
        logger.debug("-------------------------------------------------------------------------------")

        if dist_to_opt < 1e-8:
            break

        it += 1

    return dist_to_opt, search_dist.mean
