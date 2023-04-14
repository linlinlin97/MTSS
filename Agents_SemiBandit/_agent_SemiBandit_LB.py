from _util import *

class LB_agent():
    """ "Lec 9: linear bandits and TS"
        1. prior_theta = prior_theta
        2. magnitute of the errors is the marginalized one
    * already incremental
    """
    @autoargs()
    def __init__(self, sigma = 1
                 , prior_gamma_mu = None, prior_gamma_cov = None
                 , L = 100, L_tot = None, K = 2, p = 3, real = None
                 , n_female = None, cold_start = None, item_update_per100 = None, item_update_freq = None):

        self.K = K
        self.L_tot = L_tot
        self.L = L
        self.cnts = np.zeros(self.L_tot)

        self._init_posterior()
        self.seed = 42
        self.timer = 0

        self.real = real
        self.n_female = n_female

        self.cold_start = cold_start
        self.item_update_per_100 = item_update_per100
        self.item_update_freq = item_update_freq

        if self.cold_start:
            self.item_from = 0
            self.item_to = L
            self.round_update = 0

    def _init_posterior(self):
        self.Cov = self.prior_gamma_cov.copy()
        self.Cov_inv = inv(self.Cov)
        self.mu = self.prior_gamma_mu.copy()

    def take_action(self, X):
        """
        X = [L, p]
        """
        np.random.seed(self.seed)
        self.seed += 1
        self.sampled_gamma = np.random.multivariate_normal(self.mu, self.Cov)

        if self.cold_start:
            self.Rs = self.sampled_Rs = X[self.item_from:self.item_to,:].dot(self.sampled_gamma)
            A = self._optimize()
            A += self.round_update*self.item_update_per_100
        else:
            self.Rs = self.sampled_Rs = X.dot(self.sampled_gamma)
            A = self._optimize()
        return A

    def receive_reward(self, t, S, obs_R, X):
        current_time = now()

        # update_data. update posteriors
        x = X[S]
        self.w_tilde = obs_R - x.dot(self.sampled_gamma)
        self.Cov_inv_last = self.Cov_inv.copy()
        self.Cov_inv += x.T.dot(x) / self.sigma ** 2
        self.Cov = inv(self.Cov_inv)

        self.mu = self.Cov.dot(self.Cov_inv_last.dot(self.mu) + x.T.dot(obs_R / self.sigma ** 2))

        self.cnts[S] += 1

        if self.cold_start:
            if (t+1)%self.item_update_freq == 0:
                self.round_update = (t+1)//self.item_update_freq
                self.item_from = self.round_update*self.item_update_per_100
                self.item_to = self.L+self.round_update*self.item_update_per_100

        self.timer += now() - current_time

    def _optimize(self):
        """ the optimal solution of a semi-bandit depends on the feasible set.
        In the simplest setting where \mathcal{A} = all sets of size K, it can be efficiently solved.
        Otherwise, we need some approximate algorithm
        """
        if self.real:
            A = []
            A += list(np.argsort(self.Rs[:self.n_female])[::-1][:self.K//2]) #optimal female
            A += list(np.argsort(self.Rs[self.n_female:])[::-1][:self.K-self.K//2]+self.n_female) # optimal male
            A = np.array(A)
        else:
            pos_R = len([i for i in self.Rs if i>=0])
            if pos_R<self.K:
                A = np.argsort(self.Rs)[::-1][:pos_R]
            else:
                A = np.argsort(self.Rs)[::-1][:self.K]
        return A
