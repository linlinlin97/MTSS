from _util import *
import copy

def beta_reparameterize(pi,phi_beta):
    return pi / phi_beta, (1 - pi) / phi_beta

class TS_agent():
    """ 
    TS_agent: u_prior_alpha = np.ones(L), u_prior_beta = np.ones(L)
    meta_oracle_agent : using true alpha and true beta
    """
    @autoargs()
    def __init__(self, K, u_prior_alpha = None, u_prior_beta = None):
        
        self.K = K
        self.L = L = len(u_prior_alpha)
        self.cnts = np.zeros(self.L)
        self._init_posterior(self.L)
        self.seed = 42
        self.timer = 0
        
    def _init_posterior(self, L):
        self.posterior_alpha = copy.deepcopy(self.u_prior_alpha)
        self.posterior_beta = copy.deepcopy(self.u_prior_beta)
        
    def take_action(self, X = None):
        np.random.seed(self.seed)
        self.seed += 1
        self.theta = np.random.beta(self.posterior_alpha, self.posterior_beta, self.L)
        S = np.argsort(self.theta)[::-1][:self.K]
        return S
        
    def receive_reward(self, S, W, E, exp_R, R, t, X = None):
        current_time = now()
        
        # update_data. update posteriors
        exam = True
        i = 0
        while exam and i<len(S):
            if E[i] == 1:
                exam = True
                self.posterior_alpha[S[i]] += W[i]
                self.posterior_beta[S[i]] += 1-W[i]
                self.cnts[S[i]] += 1
            else:
                exam = False
            i += 1
            
        self.timer += now() - current_time