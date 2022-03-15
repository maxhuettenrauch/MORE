import numpy as np
from types import SimpleNamespace
import scipy.stats as sst
from joblib import Parallel, delayed


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

        self._phi_mean = None
        self._phi_std = None

        self.data_y_min = None
        self.data_y_max = None

        self._a_quad = np.eye(self.n)
        self._a_lin = np.zeros(shape=(self.n, 1))
        self._a_0 = np.zeros(shape=(1, 1))

        self.model_dim = None
        self.prior = None

        self.phi = None
        self.targets = None
        self.weights = None

        self.square_feat_lower_tri_ind = np.tril_indices(self.n)
        self._par = None
        self._last_model_opt = None

    @property
    def r_0(self):
        return self._a_0

    @property
    def r(self):
        return self._a_lin

    @property
    def R(self):
        return self._a_quad

    def get_model_params(self):
        return self._a_quad, self._a_lin

    def update_quad_model(self, data_x, data_y, dist=None):
        raise NotImplementedError

    def limit_model_opt(self):
        if self.options.limit_model_opt:
            try:
                model_opt = np.linalg.solve(self._a_quad, self._a_lin)
                if self._last_model_opt is not None:
                    valid = (np.linalg.norm(model_opt - self._last_model_opt) < self.options.model_limit_diff) and \
                            (np.all(np.abs(model_opt) < self.options.model_limit))
                else:
                    valid = True
                self._last_model_opt = model_opt
            except:
                # model_opt = np.zeros_like(self._a_lin)
                valid = True

            # return np.all(np.abs(model_opt) < self.options.model_limit)
            return valid
        else:
            return True

    def fit(self):
        phi_weighted = self.phi * self.weights

        # res = np.linalg.lstsq(phi_weighted, self.targets)
        #
        # par = res[0]

        reg_mat = np.eye(self.model_dim)
        reg_mat[0, 0] = 0

        phi_t_phi = phi_weighted.T @ self.phi

        if self.options.use_prior:
            par = np.linalg.solve(phi_t_phi + self.options.ridge_factor * reg_mat,
                                  phi_weighted.T @ self.targets + self.options.ridge_factor * reg_mat @ self.prior)
        else:
            par = np.linalg.solve(phi_t_phi + self.options.ridge_factor * reg_mat,
                                  phi_weighted.T @ self.targets)

        self._par = par

        return True

    def preprocess_data(self, data_x, data_y, dist, imp_weights=None):
        if len(data_y.shape) < 2:
            data_y = data_y[:, None]

        self._data_y_std = np.std(data_y, ddof=1)
        self._data_y_mean = np.mean(data_y)

        if self._data_y_std == 0:
            return False

        self.data_y_org = np.copy(data_y)

        weights = self.output_weighting(data_y, )

        try:
            data_y = self.normalize_output(data_y)
        except ValueError:
            return False

        if imp_weights is None:
            imp_weights = np.ones(data_y.shape)

        if self.options.whiten_input:
            data_x = self.whiten_input(data_x, dist)
        else:
            self.data_x_mean = np.zeros((1, self.n))
            self.data_x_inv_std = np.eye(self.n)

        phi = self.poly_feat(data_x)

        if self.options.normalize_features:
            self.normalize_features(phi)

        # weights = self.output_weighting(data_y, )

        self.targets = data_y
        self.phi = phi
        self.weights = weights * imp_weights

        return True

    def postprocess_params(self):
        if self.options.normalize_features:
            par = self.denormalize_features(self._par)
        else:
            par = self._par

        a_quad = np.zeros((self.n, self.n))
        a_tri = par[self.n + 1:].flatten()
        a_quad[self.square_feat_lower_tri_ind] = a_tri
        # a_quad = 1 / 2 * (a_quad + a_quad.T)
        a_quad = - (a_quad + a_quad.T)

        a_0 = par[0]
        a_lin = par[1:self.n + 1]

        if self.options.whiten_input:
            self._whitened_a_quad = a_quad
            self._whitened_a_lin = a_lin
            a_quad, a_lin, a_0 = self.unwhiten_params(a_quad, a_lin, a_0)

        if self.options.unnormalize_output:
            a_quad, a_lin, a_0 = self.unnormalize_output(a_quad, a_lin, a_0)

        return a_quad, a_lin, a_0

    def poly_feat(self, data_x):
        lin_feat = data_x

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

        return weighting

    def normalize_robust(self, y):
        data_y_mean = np.mean(y)
        data_y_std = np.std(y, ddof=1)
        self._data_normalizer *= data_y_std
        # self._data_y_mean = data_y_mean
        new_y = (y - data_y_mean) / data_y_std
        new_y[new_y < -3] = -3
        new_y[new_y > 3] = 3
        idx = (-3 < new_y) & (new_y < 3)
        y_tmp = new_y[idx, None]

        if sst.kurtosis(y_tmp) > 0.55 and not np.isclose(data_y_std, 1):
            new_y[idx, None] = self.normalize_robust(y_tmp)
        # elif sst.kurtosis(y_tmp) < 0:
        #     new_y_tmp = np.linspace(np.min(new_y[idx]), np.max(new_y[idx]), num=y.size)[:, None]
        #     new_y[idx] = np.zeros(shape=y_tmp.shape)
        #     ind = np.argsort(y_tmp.flatten())
        #     new_y[ind] = new_y_tmp
            # return new_y
        new_y[new_y == -3] = np.min(new_y[idx])
        new_y[new_y == 3] = np.max(new_y[idx])
        return new_y

    def normalize_output(self, y):
        norm_type = self.options.normalize_output
        if norm_type == "mean_std":
            data_y_mean = np.mean(y)
            data_y_std = np.std(y, ddof=1)
            self._data_y_mean = data_y_mean
            new_y = (y - data_y_mean) / data_y_std

        elif norm_type == "mean_std_robust_recursive":
            new_y = self.normalize_robust(y)

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
        else:
            return a_quad, a_lin, a_0

        return new_a_quad, new_a_lin, new_a_0

    def whiten_input(self, x, dist):
        data_x_mean = np.mean(x, axis=0, keepdims=True)
        self.data_x_org = x
        self._data_x_mean = data_x_mean
        # data_x_inv_std = dist.sqrt_prec

        try:
            data_x_inv_std = np.linalg.inv(np.linalg.cholesky(np.cov(x, rowvar=False))).T
        except np.linalg.LinAlgError:
            data_x_inv_std = dist.sqrt_prec
        finally:
            if np.any(np.isnan(data_x_inv_std)):
                data_x_inv_std = dist.sqrt_prec
        self._data_x_inv_std = data_x_inv_std
        x = x - data_x_mean
        x = x @ data_x_inv_std

        return x

    def unwhiten_params(self, a_quad, a_lin, a_0):
        int_a_quad = self._data_x_inv_std @ a_quad @ self._data_x_inv_std.T
        int_a_lin = self._data_x_inv_std @ a_lin
        a_quad = int_a_quad  # to achieve -1/2 xMx + xm form
        a_lin = (self._data_x_mean @ int_a_quad).T + int_a_lin
        a_0 = a_0 + self._data_x_mean @ (int_a_quad @ self._data_x_mean.T - int_a_lin)

        return a_quad, a_lin, a_0


class QuadModelLS(MoreModel):
    @classmethod
    def get_default_config(cls):
        config = {"output_weighting": None,
                  "whiten_input": True,
                  "normalize_features": True,
                  "normalize_output": "mean_std",
                  "unnormalize_output": False,
                  "ridge_factor": 1e-8,
                  "seed": None,
                  "min_data_frac": 1.1,
                  "use_prior": True,
                  "limit_model_opt": True,
                  "model_limit": 20,
                  "model_limit_diff": 20,
                  }

        return config

    def __init__(self, dim, config_dict):
        super().__init__(dim, config_dict)

        self.dim_tri = int(self.n * (self.n + 1) / 2)

        self.model_dim = 1 + self.n + self.dim_tri

        self.model_params = np.zeros(self.model_dim)
        self._model_opt = np.zeros(shape=(self.n, 1))

        self._phi_mean = np.zeros(shape=(1, self.model_dim - 1))
        self._phi_std = np.ones(shape=(1, self.model_dim - 1))

        self.ridge_factor = self.options.ridge_factor

        self.prior = np.hstack([np.zeros(1 + self.n), - 1 / np.sqrt(2 * self.n) * np.eye(self.n)[self.square_feat_lower_tri_ind]])[:, None]

    def update_quad_model(self, data_x, data_y, dist=None):
        # TODO: Should this be in here?
        if len(data_x) < self.options.min_data_frac * self.model_dim:
            return False

        self._data_normalizer = 1

        success = self.preprocess_data(data_x, data_y, dist)
        if not success:
            return False

        success = self.fit()
        if not success:
            return False

        a_quad, a_lin, a_0 = self.postprocess_params()

        self._a_quad = a_quad
        self._a_lin = a_lin
        self._a_0 = a_0
        self.model_params = np.vstack([a_quad[self.square_feat_lower_tri_ind][:, None], a_lin, a_0])

        # self._model_opt = np.linalg.solve(self._a_quad, self._a_lin)

        success = self.limit_model_opt()

        return success


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

    def update_quad_model(self, data_x, data_y, dist=None, imp_weights=None, objective=None):

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
