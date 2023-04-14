from _util import *

class Semi_env():
    @autoargs()
    def __init__(self, L, K, T, p, sigma_1, sigma_2
                 , mu_gamma, Sigma_gamma
                , seed = 42, with_intercept = True
                 , X_mu = None, X_sigma = None
                 , misspecification = None, cold_start = None, item_update_per100 = None, item_update_freq = None
                ):
        self.setting = locals()
        self.setting['self'] = None
        self.seed = seed
        np.random.seed(self.seed)
        self.K = K
        self.cold_start = cold_start
        self.item_update_per_100 = item_update_per100
        self.item_update_freq = item_update_freq

        self.gamma = np.random.multivariate_normal(mu_gamma, Sigma_gamma, 1)[0]

        self.get_Phi(L, p, T)
        self.get_theta(L, T, sigma_1)
        if self.cold_start:
            L_tot = L + T//self.item_update_freq*self.item_update_per_100
        else:
            L_tot = L
        self.errors = randn(T, L_tot) * sigma_2
        self._get_optimal()


    def get_Phi(self, L, p, T):
        """ consider the simple case now
        [L, p]
        """
        np.random.seed(self.seed)
        if self.with_intercept:
            if self.cold_start:
                L_tot = L + T//self.item_update_freq*self.item_update_per_100
                self.intercept_Phi = np.ones((L_tot,1))
                self.random_part_Phi = np.random.multivariate_normal(self.X_mu, self.X_sigma, L_tot)
                self.Phi = np.concatenate([self.intercept_Phi, self.random_part_Phi], axis = 1)
            else:
                self.intercept_Phi = np.ones((L,1))
                self.random_part_Phi = np.random.multivariate_normal(self.X_mu, self.X_sigma, L)
                self.Phi = np.concatenate([self.intercept_Phi, self.random_part_Phi], axis = 1)
        else:
            if self.cold_start:
                L_tot = L + T//self.item_update_freq*self.item_update_per_100
                self.Phi = np.random.multivariate_normal(self.X_mu, self.X_sigma, L_tot)
            else:
                self.Phi = np.random.multivariate_normal(self.X_mu, self.X_sigma, L)

    def get_theta(self, L, T, sigma_1):
        """
        misspecifications can be added here. nonlinear as the true model to show the robustness w.r.t. LMM
        """
        np.random.seed(self.seed)
        if self.cold_start:
            L_tot = L + T//self.item_update_freq*self.item_update_per_100
            self.deltas = np.random.normal(0, sigma_1, L_tot)
        else:
            self.deltas = np.random.normal(0, sigma_1, L)

        self.theta_mean = self.Phi.dot(self.gamma)

        if self.misspecification is not None:
            w_linear, w_non_linear = self.misspecification[1]
            if self.misspecification[0] == "cos":
                r_max = max([np.max(a) for a in self.theta_mean])
                self.theta_mean = [self.approximate_linear_with_cos(x, r_max) * w_non_linear + x * w_linear for x in self.theta_mean]

        self.theta = self.theta_mean + self.deltas

    def get_reward(self, S, t):
        K = len(S)
        errors = self.errors[t,][S]
        obs_R = self.theta[S] + errors
        R = np.sum(obs_R)
        exp_R = np.sum(self.theta[S])
        return [obs_R, exp_R, R]

    def _get_optimal(self):
        if self.cold_start:
            self.opt_S = []
            self.opt_exp_R = []
            self.opt_exp_R_T = []
            remain = self.T
            for i in range(self.T//self.item_update_freq+1):
                pos_theta = len([i for i in self.theta[i*self.item_update_per_100:self.L+i*self.item_update_per_100] if i>=0])
                if pos_theta<self.K:
                    opt_S = np.argsort(self.theta[i*self.item_update_per_100:self.L+i*self.item_update_per_100])[::-1][:pos_theta]+i*self.item_update_per_100
                else:
                    opt_S = np.argsort(self.theta[i*self.item_update_per_100:self.L+i*self.item_update_per_100])[::-1][:self.K]+i*self.item_update_per_100
                self.opt_S.append(opt_S)
                opt_exp_R = np.sum(self.theta[opt_S])
                self.opt_exp_R.append(opt_exp_R)
                if remain > self.item_update_freq:
                    self.opt_exp_R_T += list(np.repeat(opt_exp_R, self.item_update_freq))
                    remain -= self.item_update_freq
                else:
                    self.opt_exp_R_T += list(np.repeat(opt_exp_R, remain))
                    remain = 0
            self.opt_cum_exp_R = np.cumsum(self.opt_exp_R_T)
        else:
            pos_theta = len([i for i in self.theta if i>=0])
            if pos_theta<self.K:
                self.opt_S = np.argsort(self.theta)[::-1][:pos_theta]
            else:
                self.opt_S = np.argsort(self.theta)[::-1][:self.K]
            self.opt_exp_R = np.sum(self.theta[self.opt_S])
            self.opt_cum_exp_R = np.cumsum(np.repeat(self.opt_exp_R, self.T))


    def approximate_linear_with_cos(self, x, max_x):
        """
        sin is close to x between [-pi / 2, pi / 2]
        1. x back to this range
        2. get the value of sin(x)
        3. transform back
        """
        factor = 1 / max_x * np.pi / 2
        return np.cos(x * factor) / factor
