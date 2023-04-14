from _util import *
import pymc3 as pm
import os

from sklearn.linear_model import LogisticRegression
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sklearn.linear_model import SGDClassifier

class GLB_agent():
    """ Randomized exploration in generalized linear bandits
    """
    @autoargs()
    def __init__(self, L = 100, K = 2, p = 3
                , alpha = 1 # same with the paper
                 , true_gamma_4_debug = None
                 , retrain_freq = 1
                ):
        
        self.K = K
        self.L = L
        self.Xs = []
        self.XX = []
        self.Ys = []
        self.seed = 42
        self.clf = SGDClassifier(max_iter=1000, tol=1e-3, loss = "log", fit_intercept = False, random_state = 42, warm_start = True)

        self.gamma_mean = ones(p)
        self.H = identity(p)
        self.gamma_acc = []
        self.time_cost = {"inv" : 0, "sample" : 0, "other1" : 0, "other2" : 0}
        self.random_exploration = 0
        self.cnt = 0
        self.timer = 0
        
    def derivative_logistic(self, x):
        num = np.exp(-x)
        return num / (1 + num) ** 2
        
    def take_action(self, X):
        """
        X = [L, p]
        """
        np.random.seed(self.seed)
        self.seed += 1
        try:
            self.inv_H = inv(self.H)
            self.sampled_gamma = np.random.multivariate_normal(self.gamma_mean, self.alpha ** 2 * self.inv_H)
            self.theta = X.dot(self.sampled_gamma) # monotone. logistic
            S = np.argsort(self.theta)[::-1][:self.K]
        except:
            S = random.sample(range(self.L),self.K)
            self.random_exploration += 1
        
        return S
        
    def receive_reward(self, S, W, E, exp_R, R, t, X):
        current_time = now()
        exam = True
        i = 0
        while exam and i<len(S):
            if E[i] == 1:
                exam = True
                x = X[S[i]]
                y = W[i]
                self.Xs.append(x)
                self.Ys.append(y)
                self.XX.append(np.outer(x, x))
            else:
                exam = False
            i += 1
        if len(set(self.Ys)) > 1 and self.cnt % self.retrain_freq == 0:
            self.clf.fit(self.Xs, self.Ys)
            self.gamma_mean = self.clf.coef_[0]
            self.weights = self.derivative_logistic(arr(self.Xs).dot(self.gamma_mean))
            self.H = arr(self.XX).T.dot(self.weights)

        if t % 100 == 0 and self.true_gamma_4_debug is not None:
            self.gamma_acc.append(self.RMSE_gamma(self.gamma_mean))
            #display(self.gamma_acc)

        self.cnt += 1
        
        self.timer += now() - current_time
        
    def RMSE_gamma(self, v):
        return np.sqrt(np.mean((v - self.true_gamma_4_debug) **2))