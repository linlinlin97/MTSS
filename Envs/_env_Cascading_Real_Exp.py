from _util import *
class Cascading_env():
    @autoargs()
    def __init__(self, L, K, T, mu_gamma, sigma_gamma, true_theta, phi_beta, W_test, true_gamma, X, regret_type = None, same_reward = True, seed = 0, p =6, with_intercept = True):
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.gamma = true_gamma
        self.W_test = W_test
        self.same_reward = same_reward
        self.with_intercept = with_intercept
        self.theta = true_theta
        self.K = K
        self.T = T
        self.Phi = X

        self._get_optimal(regret_type = regret_type)

    def sample_reward(self, S):
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.seed += 1
        selected_user = random.sample(range(self.W_test.shape[0]),1)[0]
        W = self.W_test[selected_user, S]
        E = np.zeros(self.K)
        exam = True
        i = 0
        while exam and i<len(S):
            E[i] = 1
            if W[i] == 1:
                exam = False
            else:
                i += 1
                
        exp_R = self.get_exp_R_of_S(S)
        R = np.dot(W,E)
        return [W,E,exp_R,R]

    def get_exp_R_of_S(self, S):
        return 1-np.prod(1-self.theta[S])

    def _get_optimal(self,regret_type = 1):
        if regret_type == 1:            
            ID = np.array(list(range(self.W_test.shape[1])))
            ID = np.reshape(ID, (1,-1))
            W = np.concatenate((ID, self.W_test))
            self.opt_S = self.optimal_K_true(W,self.K)
        else:
            self.opt_S = np.argsort(self.theta)[::-1][:self.K]
        self.opt_exp_R = self.get_exp_R_of_S(self.opt_S)
        self.opt_cum_exp_R = np.cumsum(np.repeat(self.opt_exp_R, self.T))

    def optimal_K_true(self,W,K):
        opt_S = []
        while K>0 and W.shape[0]>1:
            Y = W[1:,:]
            opt_item = np.argmax(np.sum(Y,axis=0))
            opt_S.append(int(W[0,opt_item]))
            X = [W[0,:]]
            for i in range(1,W.shape[0]):
                if W[i,opt_item] == 0:
                    X.append(W[i,:])
            if len(X)==1:
                break
            X = np.array(X)
            X = np.delete(X, opt_item, axis=1)
            W = X
            K -= 1
        return opt_S