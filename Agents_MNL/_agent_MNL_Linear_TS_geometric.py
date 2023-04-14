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

class MNL_TS_Contextual():
    @autoargs()
    def __init__(self, L, r, K, Xs = None, n_init = 1000,
                 gamma_prior_mean = None, true_gamma = None,
                 gamma_prior_cov = None,update_freq=10, seed = None, pm_core = 1, same_reward = True):
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.Phi = Xs
        self.L = L
        self.r = r
        self.K = K
        self.p = p = len(gamma_prior_mean)
        self.traces = []
        self.update_freq = update_freq

        self.gamma = np.random.multivariate_normal(self.gamma_prior_mean, self.gamma_prior_cov)
        self.gamma_2_theta() #get theta

        self.explores_status = False
        self.S = self.solve_S_MNL(self.v)

        self._init_recorder(L)
        self._init_prior_posterior_info()
        self.timer = 0

    def _init_prior_posterior_info(self):
        self.idx = []
        self.n_select = []

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
        self.theta = np.dot(self.Phi,self.gamma)
        self.theta = (logistic(self.theta)+1)/2
        self.v = 1/self.theta - 1


    def take_action(self):
        return self.S

    def receive_reward(self, S, c, R, exp_R):
        """ epoch_offering
        """
        current_time = now()

        np.random.seed(self.seed)
        random.seed(self.seed)
        self.seed += 1
        self.t += 1
        if c == -1: # update the posterior of theta with collected data, renew the assortment, and update the storage
            self.epoch_rec[self.e]["T_l"] += 1

            self.idx += list(self.S)
            for i in S:
                self.n_select.append(self.epoch_rec[self.e]["counts"][i])

            ## Update Gamma
            if self.true_gamma is not None:
                self.gamma == self.true_gamma
            else:
                if (self.e+1)%self.update_freq == 0:
                    self.get_gamma_pos_dist()
                    #self.gamma_2_theta()
                if len(self.traces) > 0:
                    #sapmle gamma from the posterior distribution and get the corresponding theta
                    self.gamma = self.trace['gamma'][np.random.randint(0,len(self.trace['gamma']),1)[0]]
                else:
                    self.gamma = np.random.multivariate_normal(self.gamma_prior_mean, self.gamma_prior_cov)
            self.sampled_gamma.append(self.gamma)
            self.gamma_2_theta()

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
        else: # offer the same S and collect data
            self.epoch_rec[self.e]["counts"][c] += 1
            self.epoch_rec[self.e]["T_l"] += 1

        self.Rs.append(R)
        self.exp_Rs.append(exp_R)
        self.Cs.append(c)
        self.Ss.append(self.S)
        
        self.timer += now() - current_time
        

    def get_gamma_pos_dist(self):
        t = self.t
        L, p = self.L, self.p
        n_init = self.n_init
        n_tune = 100
        chains = 1
        n_sample = max(n_init - int(t*.001),100)
        with pm.Model() as Normal_Geom:
            gamma_temp = pm.MvNormal('gamma', mu=np.zeros(p), cov=np.identity(p),shape=p)
            alpha_temp = pm.math.dot(self.Phi, gamma_temp)
            mean_theta = (logistic(alpha_temp)+1)/2
            theta_full = mean_theta[self.idx]

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
        self.gamma_mean.append(np.mean(trace["gamma"], 0))


    def solve_S_MNL(self, v):
        """ based on XXX
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.seed += 1
        if self.same_reward:
            return np.argpartition(v, -self.K)[-self.K:]

        v[v<1e-6] = 1e-6
        #np.random.seed(self.seed)
        if self.explores_status == True:
            S = np.array(random.sample(range(0, self.L), self.K))
        else:
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
