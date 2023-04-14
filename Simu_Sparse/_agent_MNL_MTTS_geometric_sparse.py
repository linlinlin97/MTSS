from _util import *
import pymc3 as pm
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

########################################################################
import logging
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)
# logger1 = logging.getLogger('theano')
# logger1.setLevel(logging.ERROR)

# _logger = logging.getLogger("theano.gof.compilelock")
# _logger.setLevel(logging.ERROR)
_logger = logging.getLogger("INFO (theano.gof.compilelock)")
_logger.setLevel(logging.ERROR)
########################################################################

class MNL_MTTS():
    @autoargs()
    def __init__(self, L, r, K, Xs = None, phi_beta = None, n_init = 1000, 
                 gamma_prior_mean = None, true_gamma = None, true_v_mean = None, 
                 gamma_prior_cov = None, update_freq=10, seed = None, pm_core = 1, same_reward = True, clip = True, 
                 sparse = False, sparse_p = None, spike_slab = False):
        seed = 0 
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.spike_slab = spike_slab
        if self.sparse:
            self.p = p = sparse_p
            self.Phi = Xs[:,:sparse_p]
            self.gamma_prior_cov = gamma_prior_cov[:sparse_p,:sparse_p]
            self.gamma_prior_mean = gamma_prior_mean[:sparse_p]
        else:
            self.Phi = Xs
            self.p = p = len(gamma_prior_mean)
            self.gamma_prior_cov = gamma_prior_cov
            self.gamma_prior_mean = gamma_prior_mean
        self.phi_beta = phi_beta
        self.L = L
        self.r = r
        self.K = K
        self.traces = []
        self.update_freq = update_freq

        self.gamma = np.random.multivariate_normal(self.gamma_prior_mean, self.gamma_prior_cov)
        self.gamma_2_theta() #get theta
        self.theta_posterior_alpha, self.theta_posterior_beta = self.theta_prior_alpha, self.theta_prior_beta
        self.theta_2_v()

        self.explores_status = False
        self.S = self.solve_S_MNL(self.v)

        self._init_recorder(L)
        self._init_prior_posterior_info()
        
        self.timer = 0

    def _init_prior_posterior_info(self):
        self.idx = []
        self.n_select = []
        self.V = np.zeros(self.L)
        self.n = np.zeros(self.L)

        
    def _init_recorder(self, L):

        self.t = 0
        self.e = 1
        self.Rs = []
        self.exp_Rs = []
        self.Cs = []
        self.Ss = []
        self.epoch_rec = {} # S, and times for each item
        self.epoch_rec[1] = {"S" : self.S, "counts" : np.zeros(self.L), "T_l" : 0}
        self.epoch_mean = [[] for i in range(self.L)]
        self.cnt_of_epoch_each_i = np.repeat(0, L)
        self.S_cnt = {}
        self.change_point = []
        self.sampled_gamma = []
        self.gamma_mean = []


    def gamma_2_theta(self):
        self.theta_mean = np.dot(self.Phi,self.gamma)
        self.theta_mean = (logistic(self.theta_mean)+1)/2
        self.theta_prior_alpha, self.theta_prior_beta = beta_reparameterize(self.theta_mean, self.phi_beta)

    def theta_2_v(self):
        """ the sampling of theta is here
        """
        np.random.seed(self.seed)
        self.seed += 1
        self.theta = np.random.beta(self.theta_posterior_alpha, self.theta_posterior_beta)
        if self.clip:
            self.theta = np.clip(self.theta, 1/2, 0.999)

        self.v = 1/self.theta - 1

        
    def take_action(self):
        return self.S

    def receive_reward(self, S, c, R, exp_R):
        """ epoch_offering
        """
        #np.random.seed(self.seed)
        #random.seed(self.seed)
        #self.seed += 1
        self.t += 1
        if c == -1: # update the posterior of gamma and the prior of theta with collected data, renew the assortment, and update the storage
            timer_this = 0
            current_time = now()
            self.epoch_rec[self.e]["T_l"] += 1

            self.idx += list(self.S)
            for i in S:
                self.n_select.append(self.epoch_rec[self.e]["counts"][i])
                self.V[i] += self.epoch_rec[self.e]["counts"][i]
                self.n[i] += 1

            timer_this += now() - current_time
            ## Update Gamma
            if self.true_gamma is not None:
                self.gamma == self.true_gamma
                self.gamma_2_theta()
            else:
                if (self.e+1)%self.update_freq == 0:
                    self.get_gamma_pos_dist()
                    self.gamma_2_theta()
            current_time = now()
#             if self.true_v_mean is not None:
#                 self.true_theta_mean = 1 / (1+self.true_v_mean)
#                 theta = (self.true_theta_mean * 1 + self.theta_mean * 0)
#                 theta = (self.true_theta_mean * 0.75 + self.theta_mean * 0.25)
#                 theta = (self.true_theta_mean * 0.5 + self.theta_mean * 0.5)
#                 theta = (self.true_theta_mean * 0 + self.theta_mean * 1)
#                 self.theta_prior_alpha, self.theta_prior_beta = beta_reparameterize(theta, self.phi_beta)

            ## Update theta and v; even gamma (prior) is fixed
            # theta ~ Beta (both prior and posterior)
            self.theta_posterior_alpha = self.theta_prior_alpha + self.n
            self.theta_posterior_beta = self.theta_prior_beta + self.V

            self.theta_2_v()

            if self.e < 0:
                self.explores_status = True
            else:
                self.explores_status = False
            self.S = self.solve_S_MNL(self.v)
            self.e += 1
            self.epoch_rec[self.e] = {"S" : self.S, "counts" : np.zeros(self.L), "T_l" : 0}

            for i in self.S:
                self.cnt_of_epoch_each_i[i] += 1
            if tuple(self.S) in self.S_cnt:
                self.S_cnt[tuple(self.S)].append(self.e)
            else:
                self.S_cnt[tuple(self.S)] = [self.e]
                
            timer_this += now() - current_time
            self.timer += timer_this
        else: # offer the same S and collect data
            self.epoch_rec[self.e]["counts"][c] += 1
            self.epoch_rec[self.e]["T_l"] += 1
        self.Rs.append(R)
        self.exp_Rs.append(exp_R)
        self.Cs.append(c)
        self.Ss.append(self.S)

    def get_gamma_pos_dist(self):
        t = self.t
        L, p = self.L, self.p
        n_init = self.n_init
        n_tune = 100
        chains = 1
        n_sample = max(n_init - int(t*.002),100)
        mu_hat = 0
        sigma_hat = 10
        tau = 5
        if self.spike_slab:
            with pm.Model() as Normal_Geom:
                Z = pm.Bernoulli('Z',p = .95,shape = p)
                mixture_sd = pm.Deterministic('mixture_sd', pm.math.switch(Z > 0.5, 1, 0.0001))
                gamma_temp = pm.Normal('gamma', mu=0, sigma=mixture_sd,shape=p)
                alpha_temp = pm.math.dot(self.Phi, gamma_temp)
                mean_theta = (logistic(alpha_temp)+1)/2
                alpha_Beta, beta_Beta = beta_reparameterize(mean_theta, self.phi_beta)
                theta = pm.Beta('theta', alpha= alpha_Beta, beta=beta_Beta , shape=L)
                theta_full = theta[self.idx]

                y = pm.Geometric('y', p=theta_full, observed=np.array(self.n_select)+1) # add 1 for the reparam

                if len(self.traces)==0:
                    last_trace = None
                else:
                    last_trace = self.trace

                trace = pm.sample(n_sample, tune = n_tune, chains = chains
                                  , cores = self.pm_core, progressbar = 0, init='adapt_diag',
                                  nuts={'target_accept':0.99}, trace = last_trace);
        else:
            with pm.Model() as Normal_Geom:
                gamma_temp = pm.MvNormal('gamma', mu=self.gamma_prior_mean, cov=self.gamma_prior_cov,shape=p)
                alpha_temp = pm.math.dot(self.Phi, gamma_temp)
                mean_theta = (logistic(alpha_temp)+1)/2
                alpha_Beta, beta_Beta = beta_reparameterize(mean_theta, self.phi_beta)
                theta = pm.Beta('theta', alpha= alpha_Beta, beta=beta_Beta , shape=L)
                theta_full = theta[self.idx]

                y = pm.Geometric('y', p=theta_full, observed=np.array(self.n_select)+1) # add 1 for the reparam

                if len(self.traces)==0:
                    last_trace = None
                else:
                    last_trace = self.trace

                trace = pm.sample(n_sample, tune = n_tune, chains = chains
                                  , cores = self.pm_core, progressbar = 0, init='adapt_diag',
                                  target_accept=0.95, trace = last_trace);
            
        self.trace = trace
        self.traces.append(self.e)
        self.gamma = trace["gamma"][-1]
        self.sampled_gamma.append(self.gamma)
        self.gamma_mean.append(np.mean(trace["gamma"], 0))


    def solve_S_MNL(self, v):
        """ based on XXX
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.seed += 1
        v[v<1e-6] = 1e-6
        if self.explores_status == True:
            S = np.array(random.sample(range(0, self.L), self.K))
        else:
            if self.same_reward:
                return np.argpartition(v, -self.K)[-self.K:]

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
