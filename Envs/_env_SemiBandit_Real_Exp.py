from _util import *

class Semi_env():
    @autoargs()
    def __init__(self, L, K, T, p, sigma_1, sigma_2, mu_gamma, sigma_gamma,true_theta, true_gamma, X, n_female
                , seed = 42, with_intercept = True
                ):
        self.setting = locals()
        self.setting['self'] = None
        self.seed = seed
        np.random.seed(self.seed)
        self.K = K
        
        
        self.gamma = true_gamma
        self.theta = true_theta
        self.Phi = X

        self.errors = randn(T, L) * sigma_2
        self._get_optimal(n_female)
                
    def get_reward(self, S, t):
        K = len(S)
        errors = self.errors[t,][S]
        obs_R = self.theta[S] + errors
        R = np.sum(obs_R)
        exp_R = np.sum(self.theta[S])
        return [obs_R, exp_R, R]
    
    def _get_optimal(self,n_female):
        opt_S = []
        opt_S += list(np.argsort(self.theta[:n_female])[::-1][:self.K//2]) #optimal female
        opt_S += list(np.argsort(self.theta[n_female:])[::-1][:self.K-self.K//2]+n_female) # optimal male
        self.opt_S = np.array(opt_S)
        self.opt_exp_R = np.sum(self.theta[self.opt_S])
        self.opt_cum_exp_R = np.cumsum(np.repeat(self.opt_exp_R, self.T))


   