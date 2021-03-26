import numpy as np
from types import SimpleNamespace
from cma.evolution_strategy import RecombinationWeights
from joblib import Parallel, delayed


def default_config_ls():
    config = {"output_weighting": "min_max",
              "whiten_input": True,
              "normalize_features": True,
              "normalize_output": "mean_std",
              "unnormalize_output": False,
              "ridge_factor": 1e-12,
              "limit_model_opt": True,
              "refit": False,
              "seed": None,
              "increase_complexity": False,
              "min_data_frac": 1.5,
              "use_prior": True,
              "model_limit": 20
              }

    return config


def default_config_ls_rank():
    config = {"output_weighting": None,
              "whiten_input": True,
              "normalize_features": True,
              "normalize_output": "rank",
              "unnormalize_output": False,
              "ridge_factor": 1e-12,
              "limit_model_opt": True,
              "refit": False,
              "seed": None}

    return config


class MoreModel:
    def __init__(self, n, config_dict):
        self.n = n
        self.options = SimpleNamespace(**config_dict)

        self.data_x_org = None
        self._data_x_mean = None
        self._data_x_inv_std = None

        self.data_y_org = None
        self._data_y_mean = None
        self._data_y_std = None

        self.data_y_min = None
        self.data_y_max = None

        self._a_quad = np.eye(self.n)
        self._a_lin = np.zeros(shape=(self.n, 1))
        self._a_0 = np.zeros(shape=(1, 1))

    @property
    def r(self):
        return self._a_lin

    @property
    def R(self):
        return self._a_quad

    def get_model_params(self):
        return self._a_quad, self._a_lin

    def fit(self, data_x, data_y, **fit_args):
        raise NotImplementedError("has to be implemented in subclass")

    def preprocess_data(self, data_x, data_y, dist, **kwargs):
        raise NotImplementedError

    def postprocess_params(self, params):
        raise NotImplementedError

    def poly_feat(self, data_x):
        lin_feat = data_x
        if self.options.increase_complexity:
            if self.current_complexity == "lin":
                phi = np.hstack([np.ones([data_x.shape[0], 1]), lin_feat])
            elif self.current_complexity == "diag":
                quad_feat = lin_feat ** 2
                phi = np.hstack([np.ones([data_x.shape[0], 1]), lin_feat, quad_feat])
            elif self.current_complexity == "full":
                quad_feat = np.transpose((data_x[:, :, None] @ data_x[:, None, :]),
                                         [1, 2, 0])[self.square_feat_lower_tri_ind].T
                phi = np.hstack([np.ones([data_x.shape[0], 1]), lin_feat, quad_feat])
            else:
                raise ValueError("Unrecognized model type")
        else:
            quad_feat = np.transpose((data_x[:, :, None] @ data_x[:, None, :]),
                                     [1, 2, 0])[self.square_feat_lower_tri_ind].T

            phi = np.hstack([np.ones([data_x.shape[0], 1]), lin_feat, quad_feat])

        return phi

    def normalize_features(self, phi):
        phi_mean = np.mean(phi[:, 1:], axis=0, keepdims=True)
        phi_std = np.std(phi[:, 1:], axis=0, keepdims=True, ddof=1)
        phi[:, 1:] = phi[:, 1:] - phi_mean  # or only linear part? use theta_mean?
        phi[:, 1:] = phi[:, 1:] / phi_std

        self._phi_mean = phi_mean
        self._phi_std = phi_std

        return phi

    def denormalize_features(self, par):
        par[1:] = par[1:] / self._phi_std.T
        par[0] = par[0] - self._phi_mean @ par[1:]
        return par

    def output_weighting(self, y):
        output_weighting = self.options.output_weighting
        if output_weighting is None or not output_weighting:
            weighting = np.ones(shape=(y.size, 1))
        elif output_weighting == "rank":
            cma_weights = RecombinationWeights(y.size * 2)
            cma_weights.finalize_negative_weights(cmu=1, dimension=y.size * 2, c1=1)
            ind = np.argsort(y.flatten())
            weighting = np.zeros(shape=y.shape)
            weighting[ind] = cma_weights.positive_weights[::-1, None]
        elif output_weighting == "min_max":
            weighting = (y - np.min(y)) / (np.max(y) - np.min(y))
        elif output_weighting == "inverse":
            weighting = 1 / (np.abs(y) + 1)
        elif output_weighting == "linear":
            ind = np.argsort(y.flatten())
            weighting = np.zeros(shape=y.shape)
            weighting[ind] = np.linspace(0, 20, num=y.size)[:, None]
        elif output_weighting == "linear":
            weighting = np.linspace(0, 20, num=y.size)[:, None] + 1e-6
        else:
            raise NotImplementedError

        return weighting

    def normalize_output(self, y):
        norm_type = self.options.normalize_output
        if norm_type == "mean_std":
            data_y_mean = np.mean(y)
            data_y_std = np.std(y, ddof=1)
            self._data_y_mean = data_y_mean
            new_y = (y - data_y_mean) / data_y_std

        elif norm_type == "mean_std_clipped":
            ind = np.argsort(y, axis=0)[int((1 - self.options.top_data_fraction) * len(y)):]
            top_y = y[ind[:, 0]]
            top_data_y_mean = np.mean(top_y)
            top_data_y_std = np.std(top_y)
            new_y = (y - top_data_y_mean) / top_data_y_std
            new_y[new_y < self.options.min_clip_value] = self.options.min_clip_value

        elif norm_type == "min_max":
            new_y = (y - np.min(y)) / (np.max(y) - np.min(y))

        elif norm_type == "rank_linear":
            ind = np.argsort(y.flatten())
            new_y = np.zeros(shape=y.shape)
            new_y[ind] = np.linspace(-3, 3, num=y.size)[:, None]

        elif norm_type == "rank_cma":
            cma_weights = RecombinationWeights(2 * y.size)[0:y.size]
            ind = np.argsort(y.flatten())
            new_y = np.zeros(shape=y.shape)
            new_y[ind] = ((cma_weights - np.min(cma_weights)) / (np.max(cma_weights) - np.min(cma_weights)))[::-1,
                           None]
        elif norm_type is None or not norm_type:
            new_y = y
        else:
            raise NotImplementedError

        return new_y

    def unnormalize_output(self, a_quad, a_lin, a_0):
        norm_type = self.options.normalize_output
        if norm_type == "mean_std":
            # std_mat = np.diag(self.data_y_std)
            new_a_quad = self._data_y_std * a_quad
            new_a_lin = self._data_y_std * a_lin
            new_a_0 = self._data_y_std * a_0 + self._data_y_mean
        elif norm_type == "min_max":
            new_a_quad = (self.data_y_max - self.data_y_min) * a_quad
            new_a_lin = (self.data_y_max - self.data_y_min) * a_lin
            new_a_0 = (self.data_y_max - self.data_y_min) * a_0 + self.data_y_min
        else:
            return a_quad, a_lin, a_0

        return new_a_quad, new_a_lin, new_a_0

    def whiten_input(self, x, dist):
        data_x_mean = np.mean(x, axis=0, keepdims=True)

        try:
            data_x_inv_std = np.linalg.inv(np.linalg.cholesky(np.cov(x, rowvar=False))).T
        except np.linalg.LinAlgError:
            data_x_inv_std = dist.sqrt_prec
        finally:
            if np.any(np.isnan(data_x_inv_std)):
                data_x_inv_std = dist.sqrt_prec

        self.data_x_org = x
        self._data_x_mean = data_x_mean
        self._data_x_inv_std = data_x_inv_std
        x = x - data_x_mean
        x = x @ data_x_inv_std
        return x

    def unwhiten_params(self, a_quad, a_lin, a_0):
        int_a_quad = self._data_x_inv_std @ a_quad @ self._data_x_inv_std.T
        int_a_lin = self._data_x_inv_std @ a_lin
        a_quad = - 2 * int_a_quad  # to achieve -1/2 xMx + xm form
        a_lin = - 2 * (self._data_x_mean @ int_a_quad).T + int_a_lin
        a_0 = a_0 + self._data_x_mean @ (int_a_quad @ self._data_x_mean.T - int_a_lin)

        return a_quad, a_lin, a_0

    def refit_pos_def(self, data_x, data_y, M, weights):
        w, v = np.linalg.eig(M)
        w[w > 0.0] = -1e-8
        M = v @ np.diag(np.real(w)) @ v.T

        # refit quadratic
        aux = data_y - np.einsum('nk,kh,nh->n', data_x, M, data_x)[:, None]
        lin_feat = data_x
        phi = np.hstack([np.ones([data_x.shape[0], 1]), lin_feat])

        phi_weighted = phi * weights

        phi_t_phi = phi_weighted.T @ phi

        par = np.linalg.solve(phi_t_phi + self.options.ridge_factor * np.eye(self.n + 1),
                              phi_weighted.T @ aux)

        return par


class QuadModelLS(MoreModel):
    def __init__(self, dim, config_dict):
        super().__init__(dim, config_dict)

        self.dim_tri = int(self.n * (self.n + 1) / 2)

        if self.options.increase_complexity:
            self.model_dim_lin = 1 + self.n
            self.model_dim_diag = self.model_dim_lin + self.n
            self.model_dim_full = 1 + self.n + self.dim_tri
            self.model_dim = self.model_dim_lin
            self.current_complexity = "lin"
        else:
            self.model_dim = 1 + self.n + self.dim_tri
            self.current_complexity = "full"

        self.model_params = np.zeros(self.model_dim)
        self.square_feat_lower_tri_ind = np.tril_indices(self.n)

        self._phi_mean = np.zeros(shape=(1, self.model_dim - 1))
        self._phi_std = np.ones(shape=(1, self.model_dim - 1))

        self.ridge_factor = self.options.ridge_factor

        self.phi = None
        self.targets = None
        self.weights = None

        self.prior = np.hstack([np.zeros(1 + self.n), - 1 / np.sqrt(2 * self.n) * np.eye(self.n)[self.square_feat_lower_tri_ind]])[:, None]

    def preprocess_data(self, data_x, data_y, dist, imp_weights=None):
        if self.options.increase_complexity:
            sufficient_data = self.update_complexity(data_x)
            if not sufficient_data:
                return False

        if len(data_y.shape) < 2:
            data_y = data_y[:, None]

        self.data_y_org = np.copy(data_y)

        if imp_weights is None:
            imp_weights = np.ones(data_y.shape)

        self._data_y_std = np.std(data_y, ddof=1)

        if self._data_y_std == 0:
            return False

        if self.options.whiten_input:
            data_x = self.whiten_input(data_x, dist)
        else:
            self.data_x_mean = np.zeros((1, self.n))
            self.data_x_inv_std = np.eye(self.n)

        phi = self.poly_feat(data_x)

        if self.options.normalize_features:
            self.normalize_features(phi)

        data_y = self.normalize_output(data_y)

        weights = self.output_weighting(data_y, )

        self.targets = data_y
        self.phi = phi
        self.weights = weights

        return True

    def postprocess_params(self, par):
        if self.options.normalize_features:
            par = self.denormalize_features(par)

        if self.current_complexity == "lin":
            a_quad = np.zeros(shape=(self.n, self.n))
        elif self.current_complexity == "diag":
            a_quad = np.diag(par[self.n + 1:].flatten())
        elif self.current_complexity == "full":
            a_quad = np.zeros((self.n, self.n))
            a_tri = par[self.n + 1:].flatten()
            a_quad[self.square_feat_lower_tri_ind] = a_tri
            a_quad = 1 / 2 * (a_quad + a_quad.T)
        else:
            raise ValueError("Unrecognized model type")

        a_0 = par[0]
        a_lin = par[1:self.n + 1]
        # a_quad = np.zeros((self.n, self.n))
        # a_tri = par[self.n + 1:].flatten()
        # a_quad[self.square_feat_lower_tri_ind] = a_tri
        # a_quad = 1 / 2 * (a_quad + a_quad.T)

        if self.options.whiten_input:
            a_quad, a_lin, a_0 = self.unwhiten_params(a_quad, a_lin, a_0)
            if self.current_complexity == 'lin':
                # a_quad = np.eye(self.n)
                a_quad = np.zeros(shape=(self.n, self.n))
        else:
            a_quad = - 2 * a_quad

        if self.options.unnormalize_output:
            a_quad, a_lin, a_0 = self.unnormalize_output(a_quad, a_lin, a_0)

        return a_quad, a_lin, a_0

    def update_complexity(self, data_x):
        if self.options.increase_complexity:
            if len(data_x) < self.options.min_data_frac * self.model_dim_lin:
                return False
            elif len(data_x) < self.options.min_data_frac * self.model_dim_diag:
                self.current_complexity = "lin"
                self.model_dim = self.model_dim_lin
                self.prior = np.zeros((1 + self.n, 1))
                return True
            elif self.options.min_data_frac * self.model_dim_diag <= len(
                    data_x) < self.options.min_data_frac * self.model_dim_full:
                self.current_complexity = "diag"
                self.model_dim = self.model_dim_diag
                self.prior = np.hstack([np.zeros(1 + self.n), - 1 / np.sqrt(2 * self.n) * np.ones(self.n)])[:, None]
                return True
            else:
                self.current_complexity = "full"
                self.model_dim = self.model_dim_full
                self.prior = np.hstack(
                    [np.zeros(1 + self.n), - 1 / np.sqrt(2 * self.n) * np.eye(self.n)[self.square_feat_lower_tri_ind]])[
                             :, None]
                return True

    def fit(self, data_x, data_y, dist=None):
        success = self.preprocess_data(data_x, data_y, dist)
        if not success:
            return False

        phi_weighted = self.phi * self.weights

        reg_mat = np.eye(self.model_dim)
        reg_mat[0, 0] = 0

        phi_t_phi = phi_weighted.T @ self.phi

        if self.options.use_prior:
            par = np.linalg.solve(phi_t_phi + self.options.ridge_factor * reg_mat,
                                  phi_weighted.T @ self.targets + self.options.ridge_factor * reg_mat @ self.prior)
        else:
            par = np.linalg.solve(phi_t_phi + self.options.ridge_factor * reg_mat, phi_weighted.T @ self.targets)

        self._par = par

        a_quad, a_lin, a_0 = self.postprocess_params(par)

        if self.options.limit_model_opt:
            try:
                model_opt = np.linalg.solve(a_quad, a_lin)
            except:
                model_opt = np.zeros_like(a_lin)

            if np.any(np.abs(model_opt) > self.options.model_limit):
                return False

        self._a_quad = a_quad
        self._a_lin = a_lin
        self._a_0 = a_0

        self.model_params = np.vstack([a_quad[self.square_feat_lower_tri_ind][:, None], a_lin, a_0])

        return True


class QuadModelSubBLR(MoreModel):
    def __init__(self, dim_x, config_dict):
        super().__init__(dim_x, config_dict)

        self.tau_squared = 100
        self.sigma_squared = 100
        self.sub_space_dim = 5
        self.k = 1000

        self.model_dim_lin = 1 + self.sub_space_dim
        self.model_dim_diag = self.model_dim_lin + self.sub_space_dim
        self.dim_tri = int(self.sub_space_dim * (self.sub_space_dim + 1) / 2)
        self.model_dim_full = 1 + self.sub_space_dim + self.dim_tri

        self.model_dim = 1 + self.sub_space_dim + self.dim_tri

        self.beta_prior_prec = 1 / self.tau_squared * np.eye(self.model_dim_full)
        # self.beta_prior_prec[0, 0] = 1e-10

        self.beta_prior_cov = self.tau_squared * np.eye(self.model_dim_full)
        # self.beta_prior_cov[0, 0] = 1e10

        self.square_feat_lower_tri_ind = np.tril_indices(self.sub_space_dim)

        self.data_x = None
        self.data_y = None
        self.weights = None

        self._a_quad = np.eye(self.n)
        self._a_lin = np.zeros(shape=(self.n, 1))
        self._a_0 = np.zeros(shape=(1, 1))

    def poly_feat(self, data_x):
        lin_feat = data_x
        quad_feat = np.transpose((data_x[:, :, None] @ data_x[:, None, :]),
                                 [1, 2, 0])[self.square_feat_lower_tri_ind].T

        phi = np.hstack([np.ones([data_x.shape[0], 1]), lin_feat, quad_feat])

        return phi

    def beta_post(self, phi):
        phi_weighted = phi * self.weights

        phi_t_phi = phi_weighted.T @ phi

        beta_mat = 1 / self.sigma_squared * phi_t_phi + self.beta_prior_prec

        mu_beta = np.linalg.solve(beta_mat, 1 / self.sigma_squared * phi_weighted.T @ self.data_y)

        return mu_beta

    def p_d_w(self, phi, log=True):
        cov_p_d_w = self.sigma_squared * np.diag(np.abs(self.data_y).flatten()) + phi @ self.beta_prior_cov @ phi.T
        # cov_p_d_w = self.sigma_squared * np.eye(len(self.data_y)) + phi @ self.beta_prior_cov @ phi.T

        chol_cov = np.linalg.cholesky(cov_p_d_w)
        log_det = 2 * np.sum(np.log(np.diag(chol_cov)))

        quad_term_half = np.linalg.solve(chol_cov, self.data_y)

        likelihood = -0.5 * (log_det + quad_term_half.T @ quad_term_half + len(self.data_y) * np.log(2 * np.pi))

        if log:
            return likelihood.flatten()
        else:
            return np.exp(likelihood).flatten()

    def create_sub_space_model(self, _):
        w = np.random.randn(self.n, self.sub_space_dim)

        phi = self.poly_feat(self.data_x @ w)
        if self.options.normalize_features:
            self.normalize_features(phi)

        mu_beta = self.beta_post(phi)

        if self.options.normalize_features:
            mu_beta[1:] = mu_beta[1:] / self._phi_std.T
            mu_beta[0] = mu_beta[0] - self._phi_mean @ mu_beta[1:]

        p_d_w_i_log = self.p_d_w(phi, log=True)

        a_0 = mu_beta[0]

        a_lin = mu_beta[1:self.sub_space_dim + 1]

        a_quad_sub = np.zeros((self.sub_space_dim, self.sub_space_dim))
        a_tri = mu_beta[self.sub_space_dim + 1:].flatten()
        a_quad_sub[self.square_feat_lower_tri_ind] = a_tri
        a_quad_sub = 1 / 2 * (a_quad_sub + a_quad_sub.T)

        return w, p_d_w_i_log, a_0, a_lin, a_quad_sub

    def fit(self, data_x, data_y, dist=None, imp_weights=None, objective=None):

        if len(data_x) < 0.1 * self.model_dim_full:
            return False

        if self._data_y_std == 0:
            return False

        self.data_x = data_x
        self.data_y = data_y
        self.weights = 1 / np.abs(data_y)

        w_all, p_d_w_all_log, a_0_all, a_lin_all, a_quad_sub_all = zip(
            *Parallel(n_jobs=8)(delayed(self.create_sub_space_model)(i) for i in range(self.k)))

        p_max = np.max(p_d_w_all_log)
        exp = [np.exp(p - p_max) for p in p_d_w_all_log]
        log_p_d = p_max + np.log(np.sum(exp)) - np.log(self.k)

        imp_weights = np.exp([pdwi - log_p_d for pdwi in p_d_w_all_log]) #/ self.k
        a_quad = np.mean([w_i @ a_q_i @ w_i.T * iw_i for a_q_i, w_i, iw_i in zip(a_quad_sub_all, w_all, imp_weights)],
                         axis=0)

        a_lin = np.mean([w_i @ a_l_i * iw_i for a_l_i, w_i, iw_i in zip(a_lin_all, w_all, imp_weights)],
                         axis=0)

        a_0 = np.mean([a_0_i * iw_i for a_0_i, iw_i in zip(a_0_all, imp_weights)],
                         axis=0)

        a_quad = -2 * a_quad

        self._a_quad = a_quad
        self._a_lin = a_lin
        self._a_0 = a_0

        self.model_params = np.vstack([a_quad[self.square_feat_lower_tri_ind][:, None], a_lin, a_0])

        return True
