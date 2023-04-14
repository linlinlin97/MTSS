from _util import *
class TS_agent():
    '''
    oracle: u_prior_mean and u_prior_cov are the true values
    TS: u_prior_mean and u_prior_cov are the random values
    '''
    @autoargs()
    def __init__(self, L, K, sigma = 1, u_prior_mean = None, u_prior_cov_diag = None, real = None
                 , n_female = None, cold_start = None, item_update_per100 = None, item_update_freq = None):
        ### R ~ N(mu, sigma)
        ### sigma as known
        ### prior over mu

        self.L = L
        self.L_tot = len(u_prior_mean)
        self.cnts = zeros(self.L_tot)
        self._init_posterior(self.L_tot)
        self.K = K
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

    def _init_posterior(self, L_tot):
        self.posterior_u_num = self.u_prior_mean / self.u_prior_cov_diag
        self.posterior_u_den = 1 / self.u_prior_cov_diag
        self.posterior_u = self.posterior_u_num / self.posterior_u_den
        self.posterior_cov_diag = 1 / self.posterior_u_den

    def take_action(self, X = None):
        np.random.seed(self.seed)
        self.seed += 1

        if self.cold_start:
            self.Rs = np.random.normal(self.posterior_u[self.item_from:self.item_to]
                                     , self.posterior_cov_diag[self.item_from:self.item_to])
            A = self._optimize()
            A += self.round_update*self.item_update_per_100
        else:
            self.Rs = np.random.normal(self.posterior_u
                                     , self.posterior_cov_diag)
            A = self._optimize()
        return A

    def receive_reward(self, t, S, obs_R, X = None):
        current_time = now()
        # update_data. update posteriors
        self.posterior_u_num[S] += (obs_R / self.sigma ** 2)
        self.posterior_u_den[S] += (1 / self.sigma ** 2)
        self.posterior_u[S] = self.posterior_u_num[S] / self.posterior_u_den[S]
        self.posterior_cov_diag[S] = 1 / self.posterior_u_den[S]
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
