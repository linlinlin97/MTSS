from _util import *
import scipy.linalg
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
                ):
        self.K = K
        self.L = L
        self.L_tot = L_tot
        self.p = p = len(gamma_prior_mean)

        self.delta_cov = sigma_1**2*identity(L_tot)

        self.cnts = np.zeros(self.L_tot)
        self.Phi = Xs # [L, p]
        self.Phi_Simga = self.Phi.dot(gamma_prior_cov)

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
        self._init_Sigma_12_part()
        self.seed = 42
        self.recorder = {}

        self.time_record = {name : 0 for name in ["compute_inverse", "update_posterior", "update_Sigma_12_part", "update_K_delta_Cov", "inv_inner", "inv_block2", "1", "2", "3", "update_Phi", "4", "collect_4_Sigma_12_part", "update_centered_R"] + list(np.arange(10,100))}
        self.gamma_prior_cov_inv = inv(gamma_prior_cov)
        self.tt = 0

        ## approx
        self.posterior_u_num, self.posterior_u_den, self.posterior_u, self.posterior_cov_diag = {}, {}, {}, {}
        self.timer = 0

    def _init_else(self, L, p):
        self.theta_mean_prior = self.Xs.dot(self.gamma_prior_mean)
        self.theta_cov_prior = self.Xs.dot(self.gamma_prior_cov).dot(self.Xs.T) + self.delta_cov

        self.theta_mean_post = self.Xs.dot(self.gamma_prior_mean)
        self.theta_cov_post = self.Xs.dot(self.gamma_prior_cov).dot(self.Xs.T) + self.delta_cov

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
        self.centered_R_each_item = [np.zeros(0) for a in range(self.L_tot)]
        self.centered_R = np.zeros(0)

        self.next_pos_4_R = np.repeat(0, L_tot)
        #self.R_4_each_task = []

        self.observed_Xs = np.zeros((0, p))
        self.A_rec = np.array(self._init_empty("num")) #[0 for a in range(L)]

        self.Phi_obs = self._init_empty("row", p) #[np.zeros((0, p)) for a in range(L)]
        self.Phi_all = np.vstack(self.Phi_obs)

        ### Sigma_12_part
        # each i has a matrix
        self.Sigma12 = self._init_empty("col") #np.zeros((L, 0))
        # each i has a list, which stores the M_ij part for each j
        self.Sigma12_data = [np.zeros((L_tot, self.T)) for a in range(L_tot)]
        self.Sigma_idx = np.array(self._init_empty("num")) #[0 for a in range(L)]
        ### inverse part
        self.J_inv_to_be_updated = set()
        self.solve2_part1_to_be_updated = set()
        self.solve2_part1 = self._init_empty("row", L_tot)
        #self.K_delta_Cov_each_task = self._init_empty("mat_T") #[np.zeros((0, 0)) for i in range(N)]
        self.J_sigmaI_inv = self._init_empty()

        self.J_sigmaI_inv_dot_Phi_each = self._init_empty("row", p)
        self.J_sigmaI_inv_dot_R_each = self._init_empty('null_scalar')
        self.J_sigmaI_inv_dot_Phi_R_each = self._init_empty()


    def _init_Sigma_12_part(self):
        X = self.Xs
        a = self.Phi_Simga.dot(X.T)
        self.Phi_Simga_dot_X = a.T
        self.Phi_Simga_dot_X += self.delta_cov

    ################################################################################################################################################
    ################################################################################################################################################
    def receive_reward(self,  t, S, obs_R, X = None):
        """update_data
        """
        current_time = now()
        x_S = X[S]
        self.cnts[S] += 1
        a = now()
        self.update_Phi(S, x_S)
        self.time_record["update_Phi"] += (now() - a); a = now()


        self.collect_4_Sigma_12_part(obs_R, S, x_S)
        self.time_record["collect_4_Sigma_12_part"] += (now() - a); a = now()

        self.update_K_delta_Cov(S)
        self.time_record["update_K_delta_Cov"] += (now() - a); a = now()
        self.update_centered_R(obs_R, S)
        self.time_record["update_centered_R"] += (now() - a); a = now()

        self.A_rec[S] += 1
        #self.R_4_each_task.append(obs_R)
        self.Sigma_idx[S] += 1

        if self.cold_start:
            if (t+1)%self.item_update_freq == 0:
                self.round_update = (t+1)//self.item_update_freq
                self.item_from = self.round_update*self.item_update_per_100
                self.item_to = self.L+self.round_update*self.item_update_per_100

        self.timer += now() - current_time


    def take_action(self, X = None):
        np.random.seed(self.seed)
        self.seed += 1
        a = now()

        if self.tt >= 1 and (self.tt < 10 or self.tt % self.update_freq == 0):
            self.update_Sigma_12_part()
            self.time_record["update_Sigma_12_part"] += (now() - a); a = now()
            self.compute_inverse()
            self.time_record["compute_inverse"] += (now() - a); a = now()
            self.update_posterior()
            self.time_record["update_posterior"] += (now() - a); a = now()


        current_time = now()

        if self.cold_start:
            try:
                self.Rs = self.sampled_Rs = np.random.normal(self.theta_mean_post[self.item_from:self.item_to]
                                                             , np.diag(self.theta_cov_post[self.item_from:self.item_to,self.item_from:self.item_to]))
            except:
                self.Rs = self.sampled_Rs = self.theta_mean_post[self.item_from:self.item_to] + 1.96 * np.diag(self.theta_cov_post[self.item_from:self.item_to,self.item_from:self.item_to]) / np.sqrt(np.sum(self.cnts,axis=0))
            A = self._optimize()
            A += self.round_update*self.item_update_per_100
        else:
            try:
                self.Rs = self.sampled_Rs = np.random.normal(self.theta_mean_post, np.diag(self.theta_cov_post))
            except:
                self.Rs = self.sampled_Rs = self.theta_mean_post + 1.96 * np.diag(self.theta_cov_post) / np.sqrt(np.sum(self.cnts,axis=0))
            A = self._optimize()

        self.tt += 1
        self.timer += now() - current_time
        return A

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

    ################################################################################################################################################
    ###################################################### receive_reward #################################################################
    ################################################################################################################################################
    def update_Phi(self, S, x_S):
        for i in range(len(S)):
            A = S[i]
            x = x_S[i]
            self.Phi_obs[A] = self.vstack([self.Phi_obs[A], x])
            if self.Phi_obs[A].ndim == 1:
                self.Phi_obs[A] = self.Phi_obs[A][np.newaxis, :]

        self.Phi_all = np.vstack(self.Phi_obs)

    def collect_4_Sigma_12_part(self, obs_R, S, x_S):
        for i in range(len(S)):
            A = S[i]
            idx = self.Sigma_idx[A]
            self.Sigma12_data[A][:, idx] = self.Phi_Simga_dot_X[A] #new_col

    def update_K_delta_Cov(self, S):
        """ the second term
        """
        for i in range(len(S)):
            A = S[i]
            idx = self.Sigma_idx[A]
            num = self.delta_cov[A, A]
            to_add = np.repeat(num, self.A_rec[A] + 1)

            #self.K_delta_Cov_each_task[i][A][idx, :(idx + 1)] = to_add
            #self.K_delta_Cov_each_task[i][A][:(idx + 1), idx] = to_add
            self.J_inv_to_be_updated.add(A)

            self.solve2_part1_to_be_updated.add(A)

    def update_centered_R(self, obs_R, S):
        for i in range(len(S)):
            A = S[i]
            this_R = obs_R[i] - self.Xs[A].dot(self.gamma_prior_mean)
            self.centered_R_each_item[A] = np.append(self.centered_R_each_item[A], this_R)

            ik = A
            pos = self.next_pos_4_R[ik]
            self.centered_R = np.insert(self.centered_R, pos, this_R)
            self.next_pos_4_R[ik:] += 1


    ################################################################################################################################################
    ################################################################################################################################################
    def update_Sigma_12_part(self):
        self.Sigma12 = []
        for A in range(self.L_tot):
            idx = self.Sigma_idx[A]
            a = self.Sigma12_data[A][:, :idx]
            if a.shape[1] == 1:
                a = np.squeeze(a)
            self.Sigma12.append(a)

        self.Sigma12 = np.column_stack(self.Sigma12)
        if self.Sigma12.shape[1] == 1:
            self.Sigma12 = np.squeeze(self.Sigma12)

    def fast_inv(self, l):
        sigma_2, sigma_1 = self.sigma_2, self.sigma_1#np.sqrt(self.delta_cov[0, 0])
        return sigma_2 ** -2 * identity(l) - sigma_2 ** -4 * (sigma_1 ** -2 + sigma_2 ** -2 * l) ** -1

    def vstack_list_of_list(self, list_of_ele):
        return np.vstack([a for a in list_of_ele])

    def conca_list_of_list(self, list_of_ele):
        return np.concatenate([a for a in list_of_ele])


    def compute_inverse(self):
        """
        (J_ia + sigma I)^{-1}
        = sigma ** -2 * identity(N_ia) - sigma ** -4 * (sigma1 ** -2 + sigma ** -2 * N_ia) ** -1 * 11'
        """
        a = now()
        for A in self.J_inv_to_be_updated:
            N_i = j = self.Sigma_idx[A]
            #aa = self.K_delta_Cov_each_task[A][:(j), :(j)]
            self.J_sigmaI_inv[A] = self.fast_inv(j) #inv(aa + self.sigma ** 2 * np.identity(N_i))
            self.J_sigmaI_inv_dot_Phi_each[A] = self.J_sigmaI_inv[A].dot(self.Phi_obs[A])
            self.J_sigmaI_inv_dot_R_each[A] = self.J_sigmaI_inv[A].dot(self.centered_R_each_item[A])

        self.J_inv_to_be_updated = set()
        self.time_record["inv_inner"] += (now() - a); a = now()

        self.inv['J_sigmaI_inv_dot_Phi_all'] = self.vstack_list_of_list(self.J_sigmaI_inv_dot_Phi_each)

        ### Few time cost for steps below
        self.inv['inner_inv'] = inv(self.gamma_prior_cov_inv + self.Phi_all.T.dot(self.inv['J_sigmaI_inv_dot_Phi_all']))
        self.inv['J_sigmaI_inv_dot_Phi_all_dot_inner_inv'] = self.inv['J_sigmaI_inv_dot_Phi_all'].dot(self.inv['inner_inv'])
        self.inv['J_sigmaI_inv_dot_R'] = self.conca_list_of_list(self.J_sigmaI_inv_dot_R_each)
        self.aa = self.inv['J_sigmaI_inv_dot_Phi_all'].T.dot(self.centered_R)
        self.inv_dot_R = self.inv['J_sigmaI_inv_dot_R'] - self.inv['J_sigmaI_inv_dot_Phi_all_dot_inner_inv'].dot(self.aa)



    def update_posterior(self):
        sigma12 = self.Sigma12 # = self.Phi_i_Simga_theta_Phi_T[i] + self.Ms[i]
        try:
            a = now()
            # some time
            solve2_part2 = self.inv['J_sigmaI_inv_dot_Phi_all_dot_inner_inv'].dot(self.inv['J_sigmaI_inv_dot_Phi_all'].T.dot(sigma12.T))
            self.time_record["3"] += (now() - a); a = now()
            ## [1] only dominating for "concurrent"
            for A in self.solve2_part1_to_be_updated:
                idx = self.Sigma_idx[A]
                self.solve2_part1[A] = self.J_sigmaI_inv[A].dot(arr(self.Sigma12_data[A][:, :(idx)].T))
            self.solve2_part1_to_be_updated = set()
            self.time_record["inv_block2"] += (now() - a); a = now()
            # [2] some time
            solve2_part1 = self.vstack_list_of_list(self.solve2_part1)
            solve2 = solve2_part1 - solve2_part2
            self.time_record["1"] += (now() - a); a = now()

            self.theta_mean_post = self.theta_mean_prior + sigma12.dot(self.inv_dot_R)
            self.theta_cov_post = self.theta_cov_prior - sigma12.dot(solve2)
            self.time_record["2"] += (now() - a); a = now()

        except:
            self.theta_mean_post = self.theta_mean_prior + sigma12.dot(self.inv_dot_R)
            self.theta_cov_post = self.theta_cov_prior - sigma12.dot(sigma12.T) / self.sigma_2 #self.Kernel_sigmaI

    ################################################################################################################################################
    ###############################################################################################################################################
    def vstack(self, C):
        A, B = C
        if A.shape[0] == 0:
            return B
        else:
            return np.vstack([A, B])
