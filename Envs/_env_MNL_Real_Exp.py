from _util import *
class MNL_env():
    @autoargs()
    def __init__(self, L, K, T, mu_gamma, sigma_gamma, phi_beta, based_param = "gamma",
                 true_gamma = None, X_mu = None, X_sigma = None, true_v = None, X = None,
                 same_reward = True, seed = 0, p =6, with_intercept = True, clip=True):
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.same_reward = same_reward
        self.with_intercept = with_intercept

        if based_param == "gamma":
            self.gamma = true_gamma
            self.get_Phi(L,K,p)
            self.get_utility(phi_beta)
        elif based_param == "utility":
            self.v = true_v
            self.Phi = X
            self.theta = 1 / (self.v + 1) 

        self.get_r(L)
        self._get_optimal()

    def get_Phi(self, L,K,p):
        """
        [L,p]
        """
        np.random.seed(self.seed)
        if self.with_intercept:
            self.intercept_Phi = np.ones((L,1))
            self.random_part_Phi = np.random.multivariate_normal(self.X_mu, self.X_sigma, L)
            self.Phi = np.concatenate([self.intercept_Phi, self.random_part_Phi], axis = 1)
        else:
            self.Phi = np.random.multivariate_normal(self.X_mu, self.X_sigma, L)

    def get_utility(self, phi_beta):
        """
        v:[L,]
        """
        np.random.seed(self.seed)
        alpha_temp = self.Phi.dot(self.gamma)
        self.theta_mean = (logistic(alpha_temp)+1)/2
        self.alpha_Beta,self.beta_Beta = beta_reparameterize(self.theta_mean, phi_beta)
        self.theta = np.random.beta(self.alpha_Beta, self.beta_Beta)
        if self.clip:
            self.theta = np.clip(self.theta, 1/2, 0.999)
        self.v = 1/self.theta-1

    def get_r(self, L):
        np.random.seed(self.seed)
        if self.same_reward:
            self.r = np.repeat(1,self.L)
        else:
            self.r = runi(0, 1, self.L)

    def sample_reward(self, S):
        self.seed += 1
        np.random.seed(self.seed)
        purchase_prob = self.get_purchase_prob(S)
        c = np.random.choice(len(S) + 1, size = 1, replace = False, p = purchase_prob)[0]
        exp_R, _ = self.get_exp_R_of_S(S)
        if c > 0:
            c = S[c - 1]
            R = self.r[c]
        else:
            c = -1
            R = 0
        return [c, exp_R, R]

    def get_purchase_prob(self, S):
        v = self.v[S]
        v_full = np.insert(v, 0, 1)
        return v_full / sum(v_full)


    def get_exp_R_of_S(self, S):
        purchase_prob = self.v[S] / (sum(self.v[S]) + 1)
        return sum(purchase_prob * self.r[S]), purchase_prob

    def solve_S_MNL(self, v):
        """ based on XXX
        should be in the environment
        """
        np.random.seed(self.seed)
        v[v<1e-6] = 1e-6
        N = len(self.r)
        c = -np.append(self.r, 0)
        A_eq = ones((1, N + 1))
        b_eq = 1
        A_ub_1 = np.append(1 / v, -self.K)
        A_ub_2 = np.hstack([np.diag(1 / v), - ones((N, 1))])
        A_ub = np.vstack([A_ub_1, A_ub_2])
        b_ub = zeros(N + 1)
        l = zeros(N + 1)

        res = linprog(c
                      , A_eq = A_eq, b_eq = b_eq
                      , A_ub = A_ub, b_ub = b_ub
                      , bounds = [(l[k], None) for k in range(N + 1)]
                     , method="revised simplex")

        S = np.where(res.x[:-1] > 1e-6)[0]
        return S

    def _get_optimal(self):
        self.opt_S = self.solve_S_MNL(self.v)
        self.opt_exp_R, self.opt_purchase_prob = self.get_exp_R_of_S(self.opt_S)
        self.opt_cum_exp_R = np.cumsum(np.repeat(self.opt_exp_R, self.T))
