from _util import *
import scipy.linalg
import pymc3 as pm
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

"""
can still be largely save, by using those block structure.
"""

class MTB_agent():
    @autoargs()
    def __init__(self, sigma_2 = 1, L=None, L_tot=None, T = None
                 , gamma_prior_mean = None, gamma_prior_cov = None
                 , sigma_1 = None
                 , K = None
                 , Xs = None # [L, p]
                 , update_freq = 1
                 , approximate_solution = False
                 , UCB_solution = False
                 , real = None
                 , n_female = None
                 , cold_start = None, item_update_per100 = None, item_update_freq = None
                 , sparse = False, sparse_p = None
                 , spike_slab = False
                ):
        self.seed = 42
        self.n_init = 500
        self.traces = []
        self.spike_slab = spike_slab
        #self.recorder = {"sampled_gammas" : []
        #                , "sampled_priors" : []}      
        self.K = K
        self.L = L
        self.L_tot = L_tot

        if self.sparse:
            self.p = p = sparse_p
            self.Phi = Xs[:,:sparse_p]
            self.Xs = Xs[:,:sparse_p]
            self.gamma_prior_cov = gamma_prior_cov[:sparse_p,:sparse_p]
            self.gamma_prior_mean = gamma_prior_mean[:sparse_p]
        else:
            self.p = p = len(gamma_prior_mean)
            self.Phi = Xs # [L, p]

        self.delta_cov = sigma_1**2*identity(L_tot)

        self.real = real
        self.n_female = n_female

        self.cold_start = cold_start
        self.item_update_per_100 = item_update_per100
        self.item_update_freq = item_update_freq

        if self.cold_start:
            self.item_from = 0
            self.item_to = L
            self.round_update = 0

        self._init_data_storate(self.L_tot, p)
        self._init_else(self.L_tot, p)

        self.gamma_prior_cov_inv = inv(gamma_prior_cov)
        self.tt = 0

    def _init_else(self, L, p):
        self.gamma = np.random.multivariate_normal(self.gamma_prior_mean, self.gamma_prior_cov)
        self.theta_mean_prior = self.Xs.dot(self.gamma)
        self.theta_cov_prior = self.delta_cov

        self.theta_mean_post = self.Xs.dot(self.gamma)
        self.theta_cov_post = self.delta_cov

        # to save storage so as to save time
        self.inv = {}

    def _init_empty(self, aa = None, p = None):
        L_tot = self.L_tot
        if aa == "num":
            return [0 for a in range(L_tot)]
        if aa == "col":
            return np.zeros((L_tot, 0))
        if aa == "row":
            return [np.zeros((0, p)) for a in range(L_tot)]
        if aa == "mat_T":
            return [np.zeros((self.T, self.T)) for a in range(L_tot)]
        if aa == "null_scalar":
            return [np.zeros(0) for a in range(L_tot)]
        return [np.zeros((0, 0)) for a in range(L_tot)]

    def _init_data_storate(self, L_tot, p):
        """ initialize data storage and components required for computing the posterior
        """
        self.obsY = np.zeros(0)

        self.A_lst = []
        self.cnts = np.zeros(self.L_tot)
        self.Ct = len(self.A_lst)
        self.Sigma12 = self._init_empty("col") #np.zeros((L, 0))

    ################################################################################################################################################
    ################################################################################################################################################
    def receive_reward(self,  t, S, obs_R, X = None):
        """update_data
        """
        current_time = now()
        x_S = X[S]
        self.cnts[S] += 1
        
        for i in range(len(S)):
            A = S[i]
            self.A_lst.append(A)
            this_R = obs_R[i]
            self.obsY = np.append(self.obsY , this_R)
            Zit = [0]*self.L_tot
            Zit[A] = 1
            Zit = np.array(Zit)[:,np.newaxis]
            self.Sigma12  = np.append(self.Sigma12, Zit, axis=1)
        self.Ct = len(self.A_lst)
        
        if self.tt >= 1 and self.tt % self.update_freq == 0:
            self.sample_gamma(t)
            self.theta_mean_prior = self.Phi.dot(self.gamma)
        
        if self.tt >= 1 and (self.tt < 3 or self.tt % self.item_update_freq == 0):
            self.update_posterior()


    def take_action(self, X = None):
        np.random.seed(self.seed)
        self.seed += 1

        try:
            self.Rs = self.sampled_Rs = np.random.normal(self.theta_mean_post, np.diag(self.theta_cov_post))
        except:
            self.Rs = self.sampled_Rs = self.theta_mean_post + 1.96 * np.diag(self.theta_cov_post) / np.sqrt(np.sum(self.cnts,axis=0))
        A = self._optimize()

        self.tt += 1
        return A

    def _optimize(self):
        """ the optimal solution of a semi-bandit depends on the feasible set.
        In the simplest setting where \mathcal{A} = all sets of size K, it can be efficiently solved.
        Otherwise, we need some approximate algorithm
        """
        pos_R = len([i for i in self.Rs if i>=0])
        if pos_R<self.K:
            A = np.argsort(self.Rs)[::-1][:pos_R]
        else:
            A = np.argsort(self.Rs)[::-1][:self.K]
        return A


    ################################################################################################################################################
    def sample_gamma(self, t):
        """ randomly select a gamma from posterior based on all history information, using Pymc3
        """
        K, p, L = self.K, self.p, self.L
        n_tune = 500
        chains = 1
        n_sample = max(self.n_init - int(t*.002),100)
        
        if self.spike_slab:
            with pm.Model() as beta_bernoulli:
                Z = pm.Bernoulli('Z',p = max(.5,4/self.sparse_p),shape = p)
                mixture_sd = pm.Deterministic('mixture_sd', pm.math.switch(Z > 0.5, 1, 0.001))
                gamma_temp = pm.Normal('gamma', mu=0, sigma=mixture_sd,shape=p)
                #self.sigma_2**2*identity(self.Ct) + self.sigma_1**2*Z.T.dot(Z)
                theta_mean = pm.math.dot(self.Phi, gamma_temp)
                theta = pm.Normal('theta', mu = theta_mean, sigma = self.sigma_1, shape = self.L_tot)
                theta_full = theta[self.A_lst]

                obs = pm.Normal('obs', mu = theta_full, sigma = self.sigma_2, observed = self.obsY)
                if len(self.traces)==0:
                    last_trace = None
                else:
                    last_trace = self.trace
                trace = pm.sample(n_sample, tune = n_tune, chains = chains
                                  , cores = 1, progressbar = 0
                                 , nuts={'target_accept':0.99} # default = 0.8
                                  , trace = last_trace)
        else:
            with pm.Model() as beta_bernoulli:
                gamma_temp = pm.MvNormal('gamma', mu=self.gamma_prior_mean, cov=self.gamma_prior_cov,shape=p)
                #self.sigma_2**2*identity(self.Ct) + self.sigma_1**2*Z.T.dot(Z)
                theta_mean = pm.math.dot(self.Phi, gamma_temp)
                theta = pm.Normal('theta', mu = theta_mean, sigma = self.sigma_1, shape = self.L_tot)
                theta_full = theta[self.A_lst]

                obs = pm.Normal('obs', mu = theta_full, sigma = self.sigma_2, observed = self.obsY)
                if len(self.traces)==0:
                    last_trace = None
                else:
                    last_trace = self.trace
                trace = pm.sample(n_sample, tune = n_tune, chains = chains
                                  , cores = 1, progressbar = 0
                                 , target_accept = 0.99 # default = 0.8
                                  , trace = last_trace)
        self.trace = trace #update trace
        self.traces.append(trace)
        self.gamma = trace["gamma"][-1] 
        self.gamma_mean = np.mean(trace["gamma"], 0)
        #print(self.gamma_mean)
        #if self.sparse and self.spike_slab:
        #    print(np.mean(trace["Z"], 0))

        #self.recorder["sampled_gammas"].append(self.gamma)    ################################################################################################################################################

    def update_posterior(self):
        Z = self.Sigma12 # = self.Phi_i_Simga_theta_Phi_T[i] + self.Ms[i]
        if self.Ct < self.L_tot:
            sigma = inv(self.sigma_2**2*identity(self.Ct) + self.sigma_1**2*Z.T.dot(Z))
        else:
            sigma = inv((1/self.sigma_1**2)*identity(self.L_tot)+(1/self.sigma_2**2)*Z.dot(Z.T))
            sigma = (1/self.sigma_2**2)*identity(self.Ct) - (1/self.sigma_2**4)*Z.T.dot(sigma.dot(Z))
        
        obs_Phi = self.Phi[self.A_lst,:]
        centered_R = self.obsY - obs_Phi.dot(self.gamma)
            
        self.theta_mean_post = self.theta_mean_prior + self.sigma_1**2*Z.dot(sigma.dot(centered_R))
        self.theta_cov_post = np.diag([1/(1/self.sigma_1**2 + 1/self.sigma_2**2*self.cnts[i]) for i in range(self.L_tot)])
        
