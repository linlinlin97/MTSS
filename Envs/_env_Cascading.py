from _util import *
class Cascading_env():
    @autoargs()
    def __init__(self, L, K, T, mu_gamma, sigma_gamma, X_mu, X_sigma,  phi_beta, same_reward = True, seed = 0, p =6, with_intercept = True
                , fixed_gamma0 = None):
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.gamma = np.random.multivariate_normal(mu_gamma, sigma_gamma, 1)[0]
        if self.fixed_gamma0 is not None:
            self.gamma[0] = self.fixed_gamma0
        self.same_reward = same_reward
        self.with_intercept = with_intercept
        self.get_Phi(L,K,p)
        self.K = K
        self.get_theta(phi_beta)

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

    def get_theta(self, phi_beta):
        """
        v:[L,]
        """
        np.random.seed(self.seed)
        self.seed += 1
        self.theta_mean = logistic(self.Phi.dot(self.gamma))
        self.alpha_Beta,self.beta_Beta = beta_reparameterize(self.theta_mean, phi_beta)
        self.theta = np.random.beta(self.alpha_Beta, self.beta_Beta)
        #self.theta = self.theta_mean
        
    def sample_reward(self, S):
        np.random.seed(self.seed)
        self.seed += 1
        W = np.random.binomial(1, self.theta[S])
        E = np.zeros(self.K)
        exam = True
        i = 0
        while exam and i<len(S):
            E[i] = 1
            if W[i] == 1:
                exam = False
            else:
                i += 1
                
        exp_R = self.get_exp_R_of_S(S)
        R = np.dot(W,E)
        return [W,E,exp_R,R]

    def get_exp_R_of_S(self, S):
        return 1-np.prod(1-self.theta[S])

    def _get_optimal(self):
        self.opt_S = np.argsort(self.theta)[::-1][:self.K]
        self.opt_exp_R = self.get_exp_R_of_S(self.opt_S)
        self.opt_cum_exp_R = np.cumsum(np.repeat(self.opt_exp_R, self.T))
