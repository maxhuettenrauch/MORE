import numpy as np
from types import SimpleNamespace


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

    def normalize_output(self, y):
        self.data_y_org = y
        data_y_mean = np.mean(y)
        data_y_std = np.std(y, ddof=1)
        # if self.y_std == 0:
        #     return False
        y = (y - data_y_mean) / data_y_std

        self._data_y_mean = data_y_mean
        self._data_y_std = data_y_std

        return y

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

        self.model_dim = 1 + self.n + self.dim_tri

        self.model_params = np.zeros(self.model_dim)
        self.square_feat_lower_tri_ind = np.tril_indices(self.n)

        self._phi_mean = np.zeros(shape=(1, self.model_dim - 1))
        self._phi_std = np.ones(shape=(1, self.model_dim - 1))

        self.ridge_factor = self.options.ridge_factor

        self.phi = None
        self.targets = None
        self.weights = None

    def preprocess_data(self, data_x, data_y, dist, imp_weights=None):
        if len(data_y.shape) < 2:
            data_y = data_y[:, None]

        if imp_weights is None:
            imp_weights = np.ones(data_y.shape)

        self.data_y_std = np.std(data_y, ddof=1)

        if self.data_y_std == 0:
            return False

        if self.options.whiten_input:
            data_x = self.whiten_input(data_x, dist)
        else:
            self.data_x_mean = np.zeros((1, self.n))
            self.data_x_inv_std = np.eye(self.n)

        phi = self.poly_feat(data_x)

        if self.options.normalize_features:
            self.normalize_features(phi)

        if self.options.normalize_output:
            data_y = self.normalize_output(data_y)

        weights = np.ones_like(data_y)  # may weight with absolute values of reward

        self.targets = data_y
        self.phi = phi
        self.weights = weights

        return True

    def postprocess_params(self, par):
        if self.options.normalize_features:
            par = self.denormalize_features(par)

        a_0 = par[0]
        a_lin = par[1:self.n + 1]
        a_quad = np.zeros((self.n, self.n))
        a_tri = par[self.n + 1:].flatten()
        a_quad[self.square_feat_lower_tri_ind] = a_tri
        a_quad = 1 / 2 * (a_quad + a_quad.T)

        if self.options.whiten_input:
            a_quad, a_lin, a_0 = self.unwhiten_params(a_quad, a_lin, a_0)
        else:
            a_quad = - 2 * a_quad

        if self.options.unnormalize_output:
            a_quad, a_lin, a_0 = self.unnormalize_output(a_quad, a_lin, a_0)

        return a_quad, a_lin, a_0

    def fit(self, data_x, data_y, dist=None):
        if len(data_x) < 0.5 * self.model_dim:
            return False

        success = self.preprocess_data(data_x, data_y, dist)
        if not success:
            return False

        phi_weighted = self.phi * self.weights

        reg_mat = np.eye(self.model_dim)
        reg_mat[0, 0] = 0

        phi_t_phi = phi_weighted.T @ self.phi

        if self.options.ridge_factor is not None:
            par = np.linalg.solve(phi_t_phi + self.options.ridge_factor * reg_mat, phi_weighted.T @ self.targets)

        a_quad, a_lin, a_0 = self.postprocess_params(par)

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
        self.k = 500

        self.model_dim_lin = 1 + self.sub_space_dim
        self.model_dim_diag = self.model_dim_lin + self.sub_space_dim
        self.dim_tri = int(self.sub_space_dim * (self.sub_space_dim + 1) / 2)
        self.model_dim_full = 1 + self.sub_space_dim + self.dim_tri

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

        quad_term_half = np.linalg.solve(np.linalg.cholesky(cov_p_d_w), self.data_y)

        likelihood = -0.5 * (log_det + quad_term_half.T @ quad_term_half + len(self.data_y) * np.log(2 * np.pi))

        if log:
            return likelihood.flatten()
        else:
            return np.exp(likelihood).flatten()

    def fit(self, data_x, data_y, dist=None, imp_weights=None, objective=None):

        if len(data_x) < 0.1 * self.model_dim_full:
            return False

        if self._data_y_std == 0:
            return False

        self.data_x = data_x
        self.data_y = data_y
        self.weights = 1 / np.abs(data_y)

        mu_beta_all = []
        p_d_w_all = []
        p_d_w_all_log = []
        w_all = []
        a_0_all = []
        a_lin_all = []
        a_quad_sub_all = []

        for i in range(self.k):
            w = np.random.randn(self.n, self.sub_space_dim)
            w_all.append(w)

            phi = self.poly_feat(self.data_x @ w)
            if self.options.normalize_features:
                self.normalize_features(phi)

            mu_beta = self.beta_post(phi)

            if self.options.normalize_features:
                mu_beta[1:] = mu_beta[1:] / self._phi_std.T
                mu_beta[0] = mu_beta[0] - self._phi_mean @ mu_beta[1:]

            mu_beta_all.append(mu_beta)

            p_d_w_i = self.p_d_w(phi, log=False)
            p_d_w_i_log = self.p_d_w(phi, log=True)

            p_d_w_all.append(p_d_w_i)
            p_d_w_all_log.append(p_d_w_i_log)

            a_0 = mu_beta[0]

            a_lin = mu_beta[1:self.sub_space_dim + 1]

            a_quad_sub = np.zeros((self.sub_space_dim, self.sub_space_dim))
            a_tri = mu_beta[self.sub_space_dim + 1:].flatten()
            a_quad_sub[self.square_feat_lower_tri_ind] = a_tri
            a_quad_sub = 1 / 2 * (a_quad_sub + a_quad_sub.T)

            a_0_all.append(a_0)
            a_lin_all.append(a_lin)
            a_quad_sub_all.append(a_quad_sub)

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
