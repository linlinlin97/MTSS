from _util import *

class MNL_TS():
    @autoargs()
    def __init__(self, L, r, K, u_prior_alpha = None, u_prior_beta = None, seed = 100, same_reward = True, clip = True):
        """
        TS_agent: u_prior_alpha = np.ones(L), u_prior_beta = np.ones(L)
        meta_oracle_agent : using true alpha and true beta
        """
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.n = u_prior_alpha
        self.V = u_prior_beta

        self.theta = np.random.beta(self.n, self.V, self.L)
        if self.clip:
            self.theta = np.clip(self.theta, 1/2, 0.999)

        self.v = 1/self.theta - 1

        self.explores_status = True
        self.S = self.solve_S_MNL(self.v)

        self._init_recorder(L)
        self.timer = 0


    def _init_recorder(self, L):

        self.t = 1
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


    def take_action(self):
        return self.S

    def receive_reward(self, S, c, R, exp_R):
        """ epoch_offering
        """
        np.random.seed(self.seed)
        self.seed += 1
        current_time = now()
        if c == -1: # update the posterior of theta with collected data, renew the assortment, and update the storage
            self.epoch_rec[self.e]["T_l"] += 1
            self.update_posterior_theta()
            self.theta = np.random.beta(self.n, self.V, self.L)
            if self.clip:
                self.theta = np.clip(self.theta, 1/2, 0.999)

            self.v = 1/self.theta - 1
            if self.e < 1:
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

    def update_posterior_theta(self):
        for i in self.S:
            self.V[i] += self.epoch_rec[self.e]["counts"][i]
            self.n[i] += 1

    def solve_S_MNL(self, v):
        """ based on XXX
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.seed += 1
        #np.random.seed(self.seed)
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
