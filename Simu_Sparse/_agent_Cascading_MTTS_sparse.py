from _util import *
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pymc3 as pm
import os


########################################################################
import logging
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)
logger1 = logging.getLogger('theano')
logger1.setLevel(logging.ERROR)

_logger = logging.getLogger("theano.gof.compilelock")
_logger.setLevel(logging.ERROR)
_logger = logging.getLogger("INFO (theano.gof.compilelock)")
_logger.setLevel(logging.ERROR)
########################################################################

class MTTS_agent():
    @autoargs()
    def __init__(self, start = "cold", phi_beta = None, K = None
                 , gamma_prior_mean = None, gamma_prior_cov = None
                 , Xs = None # [L,p]
                 , update_freq = 100, n_init = None
                , u_prior_alpha = None, u_prior_beta = None
                 , true_gamma_4_debug = None, gamma_r_4_debug = None
                 , exp_seed = None, sparse = False, sparse_p = None, spike_slab = False
                ):
        self.seed = 42 #random seed
        self.K = K
        self.L = L = len(Xs)
        self.spike_slab = spike_slab
        if self.sparse:
            self.p = p = sparse_p
            self.Phi = Xs[:,:sparse_p]
        self.gamma_prior_cov = gamma_prior_cov[:sparse_p,:sparse_p]
        self.gamma_prior_mean = gamma_prior_mean[:sparse_p]
        self._init_prior_posterior(L)
        self.traces = []
        self.cnts = np.zeros(self.L) #record number of pulls for each individual each action
        if self.sparse:
            self.recorder = {"sampled_gammas" : []
                            , "sampled_priors" : []
                            , "Z":[]}   
        else:
            self.recorder = {"sampled_gammas" : []
                            , "sampled_priors" : []}      
        self.timer = 0
        
    def _init_prior_posterior(self, L):
        self.gamma = np.random.multivariate_normal(self.gamma_prior_mean, self.gamma_prior_cov)
        self.gamma_2_alpha_beta() #get r prior alpha and beta
        
        if self.start == "oracle_4_debug":
            ### get from oracle-TS
            self.r_prior_alpha = self.u_prior_alpha
            self.r_prior_beta = self.u_prior_beta    
        
        #initialize the posterior of r
        self.posterior_alpha = copy.deepcopy(self.r_prior_alpha)
        self.posterior_beta = copy.deepcopy(self.r_prior_beta)
        
        self.posterior_alpha_wo_prior_mean = zeros(L)
        self.posterior_beta_wo_prior_mean = zeros(L)
        
        self.idx = []
        self.n_trails = zeros(L)
        self.n_success = zeros(L)
    

    ########################################################################################################
    ########################################################################################################
    #take action after encountering task i with X at time t
    def take_action(self, X = None):
        np.random.seed(self.seed)
        self.seed += 1       
        self.theta = np.random.beta(self.posterior_alpha, self.posterior_beta+.0001, self.L)
        S = np.argsort(self.theta)[::-1][:self.K]
        return S
    
    def receive_reward(self, S, W, E, exp_R, R, t, X):
        """update_data, and posteriors
        record ia with data, the number of trails and the number of success. 
        """
        current_time = now()
        exam = True
        i = 0
        while exam and i<len(S):
            if E[i] == 1:
                exam = True
                temp_index = S[i]
                if temp_index not in self.idx:
                    self.idx.append(int(temp_index))
                self.n_trails[S[i]] += 1
                self.n_success[S[i]] += W[i]
                # Posterior updated each time based on the history of individual i
                self.posterior_alpha_wo_prior_mean[S[i]] += W[i]
                self.posterior_beta_wo_prior_mean[S[i]] += (1 - W[i])
                self.cnts[S[i]] += 1
            else:
                exam = False
            i += 1
        self.posterior_alpha = self.r_prior_alpha + self.posterior_alpha_wo_prior_mean
        self.posterior_beta = self.r_prior_beta + self.posterior_beta_wo_prior_mean
            
        self.timer += now() - current_time
        
        if self.start != "oracle_4_debug" and (t % self.update_freq == 0):
            self.sample_gamma(t)
            self.gamma_2_alpha_beta()

        
        
    ######################################## get the posterior of gamma by all history information ####################################        
    def sample_gamma(self, t):
        """ randomly select a gamma from posterior based on all history information, using Pymc3
        """
        K, p, L = self.K, self.p, self.L
        n_tune = 500
        chains = 1
        n_sample = max(self.n_init - int(t*.002),100)
        self.temp_idx = np.array([int(x) for x in self.idx])
        x = self.Phi[:,:self.sparse_p][self.temp_idx]
        R = self.n_success[self.temp_idx].astype('int32')
        n_trails = self.n_trails[self.temp_idx].astype('int32')
        if self.spike_slab:
            with pm.Model() as beta_bernoulli:
                Z = pm.Bernoulli('Z',p = .05,shape = p)
                mixture_sd = pm.Deterministic('mixture_sd', pm.math.switch(Z > 0.5, 1, 0.001))
                gamma_temp = pm.Normal('gamma', mu=0, sigma=mixture_sd,shape=p)
                #gamma_temp = pm.MvNormal('gamma', mu=self.gamma_prior_mean, cov=self.gamma_prior_cov,shape=p)
                alpha_temp = pm.math.dot(x, gamma_temp)
                mean_beta = logistic(alpha_temp)
                alpha_Beta, beta_Beta = beta_reparameterize(mean_beta, self.phi_beta)

                obs = pm.BetaBinomial('obs', alpha_Beta, beta_Beta, n_trails, observed = R)
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
                gamma_temp = pm.MvNormal('gamma', mu=self.gamma_prior_mean, cov=self.gamma_prior_cov,shape=self.sparse_p)
                alpha_temp = pm.math.dot(x, gamma_temp)
                mean_beta = logistic(alpha_temp)
                alpha_Beta, beta_Beta = beta_reparameterize(mean_beta, self.phi_beta)

                obs = pm.BetaBinomial('obs', alpha_Beta, beta_Beta, n_trails, observed = R)
                if len(self.traces)==0:
                    last_trace = None
                else:
                    last_trace = self.trace
                trace = pm.sample(n_sample, tune = n_tune, chains = chains
                                  , cores = 1, progressbar = 0
                                 , target_accept = 0.85 # default = 0.8
                                  , trace = last_trace)
        self.trace = trace #update trace
        self.traces.append(trace)
        self.gamma = trace["gamma"][-1] 
        self.gamma_mean = np.mean(trace["gamma"], 0)
        

        self.recorder["sampled_gammas"].append(self.gamma)
        if self.sparse:
            self.recorder["Z"].append(trace["Z"][-1])
            print(np.mean(trace["Z"], 0))
        if (self.exp_seed % 5 == 0) and (t % 100 == 0):
            self.mse_mean_gamma = arr([self.RMSE_gamma(np.mean(trace1["gamma"], 0)) for trace1 in self.traces])
            self.mse_sampled_gamma = arr([self.RMSE_gamma(trace1["gamma"][-1]) for trace1 in self.traces])
            self.std_gamma = arr([np.mean(np.std(trace1["gamma"], 0)) for trace1 in self.traces])
            pd.set_option("display.precision", 3)

            result = np.array([self.mse_mean_gamma, self.mse_sampled_gamma, self.std_gamma])
            s = DF(result, index=["mean", "sampled", "std"])
            #display(s)        
        
    def gamma_2_alpha_beta(self):
        """ get the prior for each task"""
        alpha_temp = self.Phi[:,:self.sparse_p].dot(self.gamma[:self.sparse_p])
        self.sample_beta_prior_mean = logistic(alpha_temp)
        self.r_prior_alpha, self.r_prior_beta = beta_reparameterize(self.sample_beta_prior_mean, self.phi_beta)
        
    def RMSE_gamma(self, v):
        return np.sqrt(np.mean((v - self.true_gamma_4_debug[:self.sparse_p]) **2))
