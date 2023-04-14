from _util import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import Envs._env_MNL as _env_MNL
reload(_env_MNL)
import Envs._env_MNL_Real_Exp as _env_MNL_real
reload(_env_MNL_real)

import Agents_MNL._agent_MNL_TS_geometric as _agent_MNL_TS_Geom
import Agents_MNL._agent_MNL_Linear_TS_geometric as _agent_MNL_Linear_TS_Geom
import Agents_MNL._agent_MNL_MTTS_geometric_sparse as _agent_MNL_MTTS_Geom
import Agents_MNL._agent_MNL_MTTS_geometric as _agent_MNL_MTTS_True
reload(_agent_MNL_TS_Geom)
reload(_agent_MNL_Linear_TS_Geom)
reload(_agent_MNL_MTTS_Geom)

import Envs._env_Cascading as _env_Cascading
reload(_env_Cascading)
import Envs._env_Cascading_Real_Exp as _env_Cascading_real
reload(_env_Cascading_real)
import Agents_Cascading._agent_Cascading_TS as _agent_Cascading_TS 
import Agents_Cascading._agent_Cascading_GLM as _agent_Cascading_GLM 
import Agents_Cascading._agent_Cascading_MTTS_sparse as _agent_Cascading_MTTS
import Agents_Cascading._agent_Cascading_MTTS as _agent_Cascading_MTTS_True
reload(_agent_Cascading_TS)
reload(_agent_Cascading_GLM)
reload(_agent_Cascading_MTTS)

import Envs._env_SemiBandit as _env_SemiBandit
reload(_env_SemiBandit)
import Envs._env_SemiBandit_Real_Exp as _env_SemiBandit_real
reload(_env_SemiBandit_real)
import Agents_SemiBandit._agent_SemiBandit_TS as _agent_SemiBandit_TS 
import Agents_SemiBandit._agent_SemiBandit_LB as _agent_SemiBandit_LB 
import Agents_SemiBandit._agent_SemiBandit_MTB_sparse as _agent_SemiBandit_MTB
reload(_agent_SemiBandit_TS)
reload(_agent_SemiBandit_LB)
reload(_agent_SemiBandit_MTB)

##############################################################################################################################

# Gaussian Semi Bandit: gamma~normal(0,1/p*I)
class Experiment():
    """ 
    main module for running the experimnet
    
    """
    @autoargs()
    def __init__(self, L, K, T, mu_gamma, sigma_gamma
                 , X_mu = None, X_sigma = None, phi_beta = None
                 , same_reward = True, seed = 0 
                 , sigma_1 = None, sigma_2 = None
                 , p = 6, with_intercept = None, clip=True
                 , Bandit_type= None
                 , real = False
                 , based_param = "gamma", true_gamma = None, true_v = None
                 , X = None, true_theta = None, W_test = None, regret_type = None
                 , n_female = None
                 , fixed_gamma0 = None, misspecification = None, skip_run = None
                 , cold_start = None, item_update_per100 = None, item_update_freq = None
                , sparse = False):
        self.setting = locals()
        self.setting['self'] = None
        
        if Bandit_type == "MNL" and sparse == True:
            self.env = _env_MNL.MNL_env(L, K, T, mu_gamma, sigma_gamma,                                   
                                X_mu, X_sigma,                                       
                                phi_beta, same_reward = same_reward, 
                                seed = seed, p = p, with_intercept = with_intercept, sparse =True)
            self.opt_purchase_prob = self.env.opt_purchase_prob
            self.r = self.env.r
            self.v = self.env.v
        
        elif Bandit_type == "MNL":
            if real:
                self.env = _env_MNL_real.MNL_env(L, K, T, mu_gamma, sigma_gamma, phi_beta, based_param = based_param,
                                    true_gamma = true_gamma, X_mu = X_mu, X_sigma = X_sigma, true_v = true_v, X = X,
                                    same_reward = same_reward,
                                    seed = seed, p = p, with_intercept = with_intercept, clip=clip)
            else:
                self.env = _env_MNL.MNL_env(L, K, T, mu_gamma, sigma_gamma,                                   
                                X_mu, X_sigma,                                       
                                phi_beta, same_reward = same_reward, 
                                seed = seed, p = p, with_intercept = with_intercept)
            self.opt_purchase_prob = self.env.opt_purchase_prob
            self.r = self.env.r
            self.v = self.env.v
        
        elif Bandit_type == "Cascading" and sparse == True:
            self.env = _env_Cascading.Cascading_env(L, K, T, mu_gamma, sigma_gamma,                                   
                                    X_mu, X_sigma,                                       
                                    phi_beta, same_reward = same_reward, 
                                    seed = seed, p = p, with_intercept = with_intercept
                                    , fixed_gamma0 = self.fixed_gamma0, sparse = True)
        elif Bandit_type == "Cascading":
            if real:
                self.env = _env_Cascading_real.Cascading_env(L, K, T, mu_gamma, sigma_gamma, true_theta,                                       
                                    phi_beta, W_test=W_test, true_gamma = true_gamma, X=X, regret_type = regret_type, same_reward = same_reward, 
                                    seed = seed, p = p, with_intercept = with_intercept)
            else:
                self.env = _env_Cascading.Cascading_env(L, K, T, mu_gamma, sigma_gamma,                                   
                                    X_mu, X_sigma,                                       
                                    phi_beta, same_reward = same_reward, 
                                    seed = seed, p = p, with_intercept = with_intercept
                                    , fixed_gamma0 = self.fixed_gamma0)
        elif Bandit_type == "SemiBandit" and sparse == True:
            self.env = _env_SemiBandit.Semi_env(L, K, T, p, sigma_1, sigma_2
                                                , mu_gamma, sigma_gamma, seed = seed
                                                , with_intercept = with_intercept
                                                , X_mu = X_mu, X_sigma = X_sigma, misspecification = misspecification
                                                , cold_start = cold_start, item_update_per100 = item_update_per100, item_update_freq = item_update_freq, sparse = True)
        elif Bandit_type == "SemiBandit":
            if real:
                self.env = _env_SemiBandit_real.Semi_env(L, K, T, p, sigma_1, sigma_2
                                                    , mu_gamma, sigma_gamma, true_theta=true_theta
                                                    , true_gamma=true_gamma, X=X, n_female = n_female
                                                    , seed = seed, with_intercept = with_intercept)
            else:
                self.env = _env_SemiBandit.Semi_env(L, K, T, p, sigma_1, sigma_2
                                                , mu_gamma, sigma_gamma, seed = seed
                                                , with_intercept = with_intercept
                                                , X_mu = X_mu, X_sigma = X_sigma, misspecification = misspecification
                                                , cold_start = cold_start, item_update_per100 = item_update_per100, item_update_freq = item_update_freq)
                
            
        
        
        self.theta = self.env.theta
        self.Phi = self.env.Phi
        self.opt_S = self.env.opt_S
        self.opt_exp_R = self.env.opt_exp_R
        self.opt_cum_exp_R = self.env.opt_cum_exp_R


    def _init_agents(self, agents = None):
        # sigma, Sigma_delta
        self.agents = agents
        self.agent_names = agent_names = list(agents.keys())
        self.record = {}
        self.record['R'] = {name : [] for name in agent_names}
        self.record['exp_R'] = {name : [] for name in agent_names}
        #self.record['W'] = {name : [] for name in agent_names}
        #self.record['E'] = {name : [] for name in agent_names}
        self.record['R']['oracle'] = []
        self.record['exp_R']['oracle'] = []
        self.record['A'] = {name : [] for name in agent_names}
        self.record['regret'] = {name : [] for name in agent_names}
        self.record['meta_regret'] = {name : [] for name in agent_names}
        self.record['total_computation_time'] = {name : [] for name in agent_names}
        self.record['optimize_computation_time'] = {name : [] for name in agent_names}
        self.record['online_computation_time'] = {name : [] for name in agent_names}
        self.record['model_update_computation_time'] = {name : [] for name in agent_names}
        
    def run(self):
        self.run_4_one_agent('oracle')
        for name in self.agent_names: 
            if name not in self.skip_run:
                current_time = now()
                self.run_4_one_agent(name)
                a = now() - current_time
                self.record['total_computation_time'][name] = a
                # self.record['optimize_computation_time'][name] = self.agents[name].timer['optimize_computation_time']
            #
            # self.agents[name]
            #
        if len(self.skip_run) == 2:
            pass
        else:
            self.post_process()

    def run_4_one_agent(self, name):
        if self.seed == 0 or name == "Meta TS (Ours)":
            for t in tqdm(range(self.T), desc = name
                                     , position=0
                                    , miniters = 20):
                self.run_one_time_point(t, name)
        else:
            for t in range(self.T):
                self.run_one_time_point(t, name)


    def run_one_time_point(self, t, name):
        if name == "oracle":
            S = self.opt_S
            if self.Bandit_type == "MNL":
                c, exp_R, R = self.env.sample_reward(S)
            elif self.Bandit_type == "Cascading":
                W, E, exp_R, R = self.env.sample_reward(S)
            elif self.Bandit_type == "SemiBandit":
                if self.cold_start:
                    tt = t//self.env.item_update_freq
                    S = self.opt_S[tt]
                obs_R, exp_R, R = self.env.get_reward(S, t) #observed R for each single item 
            self.record['R'][name].append(R)
            self.record['exp_R'][name].append(exp_R)
        else:
            # provide the action to the env and then get reward from the env
            if self.Bandit_type == "MNL":
                S = self.agents[name].take_action()
                c, exp_R, R = self.env.sample_reward(S)
                # provide the reward to the agent
                self.agents[name].receive_reward(S, c, R, exp_R)
            elif self.Bandit_type == "Cascading":
                S = self.agents[name].take_action(self.Phi)
                W, E, exp_R, R = self.env.sample_reward(S)
                # provide the reward to the agent
                self.agents[name].receive_reward(S, W, E, exp_R, R, t, self.Phi)
            elif self.Bandit_type == "SemiBandit":
                S = self.agents[name].take_action(self.Phi)
                obs_R, exp_R, R = self.env.get_reward(S, t)
                self.agents[name].receive_reward(t, S, obs_R, X = self.Phi)

            # collect the reward
            self.record['R'][name].append(R)
            self.record['exp_R'][name].append(exp_R)
            self.record['A'][name].append(S)
            #self.record['W'][name].append(W)
            #self.record['E'][name].append(E)
            
    def post_process(self):
        for name in self.agent_names:
            self.record['regret'][name] = arr(self.record['exp_R']["oracle"]) - arr(self.record['exp_R'][name])

        self.record['cum_regret'] = {name : np.cumsum(self.record['regret'][name]) for name in self.agent_names}
        # x: time, y: cum_regret: group, name
        self.record['cum_regret_df'] = self.organize_Df(self.record['cum_regret'])
                
        if "Oracle TS" in self.agent_names:
            for name in self.agent_names:
                self.record['meta_regret'][name] = arr(self.record['exp_R']['Oracle TS']) - arr(self.record['exp_R'][name])
            self.record['cum_meta_regret'] = {name : np.cumsum(self.record['meta_regret'][name]) for name in self.agent_names}
            self.record['cum_meta_regret_df'] = self.organize_Df(self.record['cum_meta_regret'])


    def organize_Df(self, r_dict):
        T = len(r_dict[self.agent_names[0]])
        a = pd.DataFrame.from_dict(r_dict)
        # a.reset_index(inplace=True)
        a = pd.melt(a)
        a['time'] = np.tile(np.arange(T), len(self.agent_names))
        a = a.rename(columns = {'variable':'method'
                           , "value" : "regret"
                           , "time" : "time"})
        return a

            
    def plot_regret(self, skip_methods = ["TS"], plot_meta = True):
        # https://seaborn.pydata.org/generated/seaborn.lineplot.html
        #ax.legend(['label 1', 'label 2'])
        if plot_meta:
            data_plot =  self.record['cum_meta_regret_df'] 
            data_plot = data_plot[data_plot.method != "Oracle TS"]
        else:
            data_plot =  self.record['cum_regret_df'] 
        if skip_methods is not None:
            for met in skip_methods:
                data_plot = data_plot[data_plot.method != met]
        ax = sns.lineplot(data = data_plot
                     , x="time", y="regret"
                          , n_boot = 100
                     , hue="method" # group variable
                    )
        

        
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class run_experiment():
    @autoargs()
    def __init__(self, L, T, K = None, p  = None
                 , phi_beta = None, Bandit_type = None
                 , print_SNR = True
                 , sigma_1 = None, sigma_2 = None
                 , with_intercept = True, same_reward = True, clip = True
                 , save_prefix = None
                 , debug_MTB = False
                 , Sigma_gamma_factor = 1, Sigma_x_factor = 1
                 , misspecification = None
                 , cold_start = False, item_update_per100 = None, item_update_freq = None
                 , only_ratio = False   
                 , MTS_freq = 100
                 , GLB_freq = 100
                 , n_init = 2000
                 , alpha_GLB = 1
                 , used_agends = 'all'
                 , real = False
                 , LB_freq = 1
                , based_param = None, true_gamma = None, true_v = None, X = None, 
                 true_theta = None, W_test = None, regret_type = None, n_female = None
                 , fixed_gamma0 = None
                 , skip_run = []
                 ,sparse = False, sparse_p_list = None
                 ,spike_slab = False
                 ,testtest = False
                ):
        self.setting = locals()
        self.setting['self'] = None
        self.title_settting = " ".join([str(key) + "=" + str(self.setting[key]) for key in self.setting if type(key) in [str, int] and type(self.setting[key]) in [str, int, float]])
        printR(self.title_settting)
        self.date_time = get_date() + get_time()
        ################
        if Sigma_gamma_factor == 'identity':
            # would like to use Sigma_gamma_factor = 1 below
            self.Sigma_gamma = np.identity(p)
        else:
            self.Sigma_gamma  = np.identity(p) / p #np.sqrt(p)  # such a normalization is not good. Should be ||theta|| = 1
            self.Sigma_gamma *= Sigma_gamma_factor
        self.mu_gamma = np.zeros(p)
        if with_intercept:
            self.X_mu = np.zeros(p-1)
            self.X_Sigma = identity(p-1) * Sigma_x_factor
        else:
            self.X_mu = np.zeros(p)
            self.X_Sigma = identity(p) * Sigma_x_factor
#        if Bandit_type in ["MNL", "Cascading", "Semi-Bandit"]:
        if self.testtest:
            self.names = ["Oracle TS", "sparse" , "true"]
        elif self.sparse:
            self.names = ["Oracle TS", "P = "+str(self.sparse_p_list[0]), "P = "+str(self.sparse_p_list[1]), "P = "+str(self.sparse_p_list[2])]
        elif not real:
            self.names = ["Meta TS (Ours)", "Feature-agnostic TS", "Oracle TS", "Feature-determined TS"]
        else:
            self.names = ["Meta TS (Ours)", "Feature-agnostic TS", "Feature-determined TS"]
    
        self.sparse = sparse
        self.sparse_p_list = sparse_p_list
    #############################################################################################################################################
                           #get the prior for TS algorithms
    #############################################################################################################################################
    def get_ri_prior(self, L, phi_beta, Bandit_type, n_rep = 100):
        """ get the prior mean and precision phi of ri for TS
        sometimes this part takes too much memory
        """
        sample_gammas = np.random.multivariate_normal(self.mu_gamma, self.Sigma_gamma, n_rep)
        if self.with_intercept:
            intercept_Phi = np.repeat(np.ones((L,1))[np.newaxis, :, :], n_rep, axis = 0)
            random_part_Phi = np.random.multivariate_normal(self.X_mu, self.X_Sigma, (n_rep, L))
            sample_Phi = np.concatenate([intercept_Phi, random_part_Phi], axis = 2)
            mu = np.exp(self.mu_gamma.dot(np.concatenate([np.array([1]), self.X_mu], axis = 0)))/(1+np.exp(self.mu_gamma.dot(np.concatenate([np.array([1]), self.X_mu], axis = 0))))
        else:
            sample_Phi = np.random.multivariate_normal(self.X_mu, self.X_Sigma, (n_rep, L))
            mu = np.exp(self.mu_gamma.dot(self.X_mu))/(1+np.exp(self.mu_gamma.dot(self.X_mu)))
        
        Conditional_Cov_ri=[]
        Conditional_Mu_ri=[]
        """ can be over-sized """
        for sample_gamma in sample_gammas:
            temp=[Phi_i.dot(sample_gamma) for Phi_i in sample_Phi]
                
            if Bandit_type == "Cascading":
                temp1 = logistic(temp)
            elif Bandit_type == "MNL":
                temp1 = (logistic(temp)+1)/2
                    
            E_ri_thetai = np.mean(temp1, 0)
            
            if Bandit_type == "Cascading":
                temp2 = logistic(temp) / (np.exp(temp)+1) # * (1-np.exp(temp))
                    
            elif Bandit_type == "MNL":
                temp2 = (1-logistic(temp)**2)/4
                    
            c_b_i1 = np.mean(temp2, 0) * phi_beta / (1+phi_beta)
            c_b_i2= np.var(temp1, 0)
            cov_ri_thetai=np.diag(c_b_i1)+np.diag(c_b_i2)
                    
            Conditional_Mu_ri.append(E_ri_thetai)
            Conditional_Cov_ri.append(cov_ri_thetai)

        E_cond_cov_ri=np.mean(Conditional_Cov_ri,axis=0)
        Cov_cond_mu_ri=np.cov(np.array(Conditional_Mu_ri).T,bias=False)
        Cov_ri=E_cond_cov_ri+Cov_cond_mu_ri

        ### TS
        if Bandit_type == "Cascading":
            ri_prior_mean_MC = np.array([logistic(mu)]*L)
        elif Bandit_type == "MNL":
            ri_prior_mean_MC = np.array([(logistic(mu)+1)/2]*L)
        ri_prior_phi_MC = meanvar_meanprex(ri_prior_mean_MC,np.diag(Cov_ri))
        ri_prior_alpha_MC, ri_prior_beta_MC = beta_reparameterize(ri_prior_mean_MC, ri_prior_phi_MC)
        
        return ri_prior_alpha_MC, ri_prior_beta_MC
        
    def run_one_testtest(self, seed):
        L, T, phi_beta, K, p, Bandit_type = self.L, self.T, self.phi_beta, self.K, self.p, self.Bandit_type
        self.exp = Experiment(L, K, T, self.mu_gamma[:4], self.Sigma_gamma[:4,:4]
                     , X_mu = self.X_mu[:4], X_sigma = self.X_Sigma[:4,:4], phi_beta = phi_beta
                     , same_reward = self.same_reward, seed = seed
                     , p = 4, with_intercept = self.with_intercept, clip=self.clip
                     , Bandit_type=self.Bandit_type, real = self.real, skip_run = self.skip_run
                    , fixed_gamma0 = self.fixed_gamma0, sparse = False)
        if self.only_ratio: 
            return [None, None] 
        ###################################### Priors ##############################################################
        #self.ri_prior_alpha_MC_cascading, self.ri_prior_beta_MC_cascading = self.get_ri_prior(L, phi_beta, "Cascading", n_rep = 50)
        self.ri_prior_alpha_MC_cascading, self.ri_prior_beta_MC_cascading = np.random.uniform(0, 1, L), np.random.uniform(0, 1, L)
        oracle_TS = _agent_Cascading_TS.TS_agent(K,u_prior_alpha = self.exp.env.alpha_Beta, u_prior_beta = self.exp.env.beta_Beta)
        MTTS_agent_sparse = _agent_Cascading_MTTS.MTTS_agent(start = "cold", phi_beta = phi_beta, K = K
                                                     , gamma_prior_mean = self.mu_gamma, gamma_prior_cov = self.Sigma_gamma
                                                     , Xs = self.exp.env.Phi # [L,p]
                                                     , update_freq = self.MTS_freq, n_init = self.n_init
                                                    , u_prior_alpha = None, u_prior_beta = None
                                                     , true_gamma_4_debug = self.exp.env.gamma, gamma_r_4_debug = None
                                                     , exp_seed = seed, sparse = self.sparse, sparse_p = 4
                                                    , spike_slab = self.spike_slab)
        MTTS_agent_normal = _agent_Cascading_MTTS_True.MTTS_agent(start = "cold", phi_beta = phi_beta, K = K
                                                     , gamma_prior_mean = self.mu_gamma[:4], gamma_prior_cov = self.Sigma_gamma[:4,:4]
                                                     , Xs = self.exp.env.Phi[:,:4] # [L,p]
                                                     , update_freq = self.MTS_freq, n_init = self.n_init
                                                    , u_prior_alpha = None, u_prior_beta = None
                                                     , true_gamma_4_debug = self.exp.env.gamma[:4], gamma_r_4_debug = None
                                                     , exp_seed = seed)

        ####################################################################################################
        agents = {
            "sparse" : MTTS_agent_sparse
            , "true" : MTTS_agent_normal
        }
        if not self.real:
            agents['Oracle TS'] = oracle_TS
        if self.used_agends != 'all':
            agents = {key : agents[key] for key in self.used_agends}

        self.exp._init_agents(agents)
        
        self.exp.run()  
        self.record = {0: self.exp.record}  
        self.agents = self.exp.agents   
        #if "MTB" in agents: 
        #    return [self.exp.record, self.exp.agents['MTB'].recorder]  
        #else:   
        #    return [self.exp.record, None] 
        return self.exp.record    #############################################################################################################################################
    def run_one_seed_sparse_MNL(self, seed):
        L, T, phi_beta, K, p = self.L, self.T, self.phi_beta, self.K, self.p
        self.exp = Experiment(L, K, T, self.mu_gamma, self.Sigma_gamma
                     , X_mu = self.X_mu, X_sigma = self.X_Sigma, phi_beta = phi_beta
                     , same_reward = self.same_reward, seed = seed
                     , p = p, with_intercept = self.with_intercept, clip=self.clip
                     , Bandit_type=self.Bandit_type, real = self.real, skip_run = self.skip_run
                    , fixed_gamma0 = self.fixed_gamma0, sparse = self.sparse)
        
        self.r = self.exp.env.r
        ###################################### Priors ##############################################################
        # step to get the prior for TS algorithm
        oracle_TS = _agent_MNL_TS_Geom.MNL_TS(L, self.r, K, u_prior_alpha = self.exp.env.alpha_Beta, u_prior_beta = self.exp.env.beta_Beta, seed = seed, same_reward = self.same_reward, clip = self.clip)
    
        MTTS_agent_sparse_p0 = _agent_MNL_MTTS_Geom.MNL_MTTS(L, self.r, K, self.exp.env.Phi, phi_beta = phi_beta,n_init = self.n_init,
                                    gamma_prior_mean = self.mu_gamma, true_gamma = None, true_v_mean = None, 
                                    gamma_prior_cov = self.Sigma_gamma,
                                    update_freq=self.MTS_freq, seed = seed, pm_core = 1, same_reward = self.same_reward, clip = self.clip,
                                    sparse = self.sparse, sparse_p = self.sparse_p_list[0], spike_slab = self.spike_slab)
        
        MTTS_agent_sparse_p1 = _agent_MNL_MTTS_Geom.MNL_MTTS(L, self.r, K, self.exp.env.Phi, phi_beta = phi_beta,n_init = self.n_init,
                                    gamma_prior_mean = self.mu_gamma, true_gamma = None, true_v_mean = None, 
                                    gamma_prior_cov = self.Sigma_gamma,
                                    update_freq=self.MTS_freq, seed = seed, pm_core = 1, same_reward = self.same_reward, clip = self.clip,
                                    sparse = self.sparse, sparse_p = self.sparse_p_list[1],spike_slab = self.spike_slab)
        
        MTTS_agent_sparse_p2 = _agent_MNL_MTTS_Geom.MNL_MTTS(L, self.r, K, self.exp.env.Phi, phi_beta = phi_beta,n_init = self.n_init,
                                    gamma_prior_mean = self.mu_gamma, true_gamma = None, true_v_mean = None, 
                                    gamma_prior_cov = self.Sigma_gamma,
                                    update_freq=self.MTS_freq, seed = seed, pm_core = 1, same_reward = self.same_reward, clip = self.clip,
                                    sparse = self.sparse, sparse_p = self.sparse_p_list[2], spike_slab = self.spike_slab)
        
        ####################################################################################################
        agents = {
            "P = "+str(self.sparse_p_list[0]) : MTTS_agent_sparse_p0
            , "P = "+str(self.sparse_p_list[1]) : MTTS_agent_sparse_p1
            , "P = "+str(self.sparse_p_list[2]) : MTTS_agent_sparse_p2
        }
        if not self.real:
            agents['Oracle TS'] = oracle_TS
        if self.used_agends != 'all':
            #print(self.used_agends, list(agents.keys()))
            agents = {key : agents[key] for key in self.used_agends}
        
        self.exp._init_agents(agents)

        self.exp.run()
        self.record = {0: self.exp.record}
        self.agents = self.exp.agents
        #if "MTB" in agents:
        #    return [self.exp.record, self.exp.agents['MTB'].recorder]
        #else:
        #    return [self.exp.record, None]
        return self.exp.record #############################################################################################################################################        
                            
    def run_one_seed_sparse_cascade(self, seed):
        L, T, phi_beta, K, p, Bandit_type = self.L, self.T, self.phi_beta, self.K, self.p, self.Bandit_type
        self.exp = Experiment(L, K, T, self.mu_gamma, self.Sigma_gamma
                     , X_mu = self.X_mu, X_sigma = self.X_Sigma, phi_beta = phi_beta
                     , same_reward = self.same_reward, seed = seed
                     , p = p, with_intercept = self.with_intercept, clip=self.clip
                     , Bandit_type=self.Bandit_type, real = self.real, skip_run = self.skip_run
                    , fixed_gamma0 = self.fixed_gamma0, sparse = self.sparse)
        if self.only_ratio: 
            return [None, None] 
        ###################################### Priors ##############################################################
        #self.ri_prior_alpha_MC_cascading, self.ri_prior_beta_MC_cascading = self.get_ri_prior(L, phi_beta, "Cascading", n_rep = 50)
        self.ri_prior_alpha_MC_cascading, self.ri_prior_beta_MC_cascading = np.random.uniform(0, 1, L), np.random.uniform(0, 1, L)
        oracle_TS = _agent_Cascading_TS.TS_agent(K,u_prior_alpha = self.exp.env.alpha_Beta, u_prior_beta = self.exp.env.beta_Beta)
        MTTS_agent_sparse_p0 = _agent_Cascading_MTTS.MTTS_agent(start = "cold", phi_beta = phi_beta, K = K
                                                     , gamma_prior_mean = self.mu_gamma, gamma_prior_cov = self.Sigma_gamma
                                                     , Xs = self.exp.env.Phi # [L,p]
                                                     , update_freq = self.MTS_freq, n_init = self.n_init
                                                    , u_prior_alpha = None, u_prior_beta = None
                                                     , true_gamma_4_debug = self.exp.env.gamma, gamma_r_4_debug = None
                                                     , exp_seed = seed, sparse = self.sparse, sparse_p = self.sparse_p_list[0]
                                                    , spike_slab = self.spike_slab)
        MTTS_agent_sparse_p1 = _agent_Cascading_MTTS.MTTS_agent(start = "cold", phi_beta = phi_beta, K = K
                                                     , gamma_prior_mean = self.mu_gamma, gamma_prior_cov = self.Sigma_gamma
                                                     , Xs = self.exp.env.Phi # [L,p]
                                                     , update_freq = self.MTS_freq, n_init = self.n_init
                                                    , u_prior_alpha = None, u_prior_beta = None
                                                     , true_gamma_4_debug = self.exp.env.gamma, gamma_r_4_debug = None
                                                     , exp_seed = seed,sparse = self.sparse, sparse_p = self.sparse_p_list[1]
                                                    , spike_slab = self.spike_slab)
        MTTS_agent_sparse_p2 = _agent_Cascading_MTTS.MTTS_agent(start = "cold", phi_beta = phi_beta, K = K
                                                     , gamma_prior_mean = self.mu_gamma, gamma_prior_cov = self.Sigma_gamma
                                                     , Xs = self.exp.env.Phi # [L,p]
                                                     , update_freq = self.MTS_freq, n_init = self.n_init
                                                    , u_prior_alpha = None, u_prior_beta = None
                                                     , true_gamma_4_debug = self.exp.env.gamma, gamma_r_4_debug = None
                                                     , exp_seed = seed,sparse = self.sparse, sparse_p = self.sparse_p_list[2]
                                                    , spike_slab = self.spike_slab)

        ####################################################################################################
        agents = {
            "P = "+str(self.sparse_p_list[0]) : MTTS_agent_sparse_p0
            , "P = "+str(self.sparse_p_list[1]) : MTTS_agent_sparse_p1
            , "P = "+str(self.sparse_p_list[2]) : MTTS_agent_sparse_p2
        }
        if not self.real:
            agents['Oracle TS'] = oracle_TS
        if self.used_agends != 'all':
            agents = {key : agents[key] for key in self.used_agends}

        self.exp._init_agents(agents)
        
        self.exp.run()  
        self.record = {0: self.exp.record}  
        self.agents = self.exp.agents   
        #if "MTB" in agents: 
        #    return [self.exp.record, self.exp.agents['MTB'].recorder]  
        #else:   
        #    return [self.exp.record, None] 
        return self.exp.record  
    
    def run_one_seed_sparse_SemiBandit(self, seed):
        L, T, sigma_1, sigma_2, K, p, Bandit_type = self.L, self.T, self.sigma_1, self.sigma_2, self.K, self.p, self.Bandit_type
        self.exp = Experiment(L, K, T, self.mu_gamma, self.Sigma_gamma
                 , X_mu = self.X_mu, X_sigma = self.X_Sigma, phi_beta = None
                 , sigma_1 = sigma_1, sigma_2 = sigma_2
                 , same_reward = self.same_reward, seed = seed
                 , p = p, with_intercept = self.with_intercept, clip=self.clip, skip_run = self.skip_run
                 , Bandit_type=self.Bandit_type, real = self.real, misspecification = self.misspecification
                , cold_start = self.cold_start, item_update_per100 = self.item_update_per100,item_update_freq = self.item_update_freq
                , sparse = self.sparse)

        if self.only_ratio: 
            return [None, None] 
        ###################################### Priors ##############################################################
        self.determine_some_priors_4_Gaussian(p, L, T, sigma_1)
        
        oracle_TS = _agent_SemiBandit_TS.TS_agent(L, K, u_prior_mean = self.exp.env.theta_mean, u_prior_cov_diag = sigma_1 ** 2 * np.ones(self.L_tot), cold_start = self.cold_start, item_update_per100 = self.item_update_per100, item_update_freq = self.item_update_freq)
        
        MTTS_agent_sparse_p0 = _agent_SemiBandit_MTB.MTB_agent(sigma_2 = sigma_2, L=L, L_tot=self.L_tot, T = T
                                                     , gamma_prior_mean = self.mu_gamma, gamma_prior_cov = self.Sigma_gamma
                                                     , sigma_1 = sigma_1
                                                     , K = K
                                                     , Xs = self.exp.env.Phi 
                                                     , update_freq = self.MTS_freq
                                                     , approximate_solution = False
                                                     , UCB_solution = False
                                                     , real = self.real, n_female = self.n_female
                                                     , cold_start = self.cold_start, item_update_per100 = self.item_update_per100
                                                     , item_update_freq = self.item_update_freq
                                                     , sparse = self.sparse, sparse_p = self.sparse_p_list[0]
                                                     , spike_slab = self.spike_slab)
        
        MTTS_agent_sparse_p1 = _agent_SemiBandit_MTB.MTB_agent(sigma_2 = sigma_2, L=L, L_tot=self.L_tot, T = T
                                                     , gamma_prior_mean = self.mu_gamma, gamma_prior_cov = self.Sigma_gamma
                                                     , sigma_1 = sigma_1
                                                     , K = K
                                                     , Xs = self.exp.env.Phi 
                                                     , update_freq = self.MTS_freq
                                                     , approximate_solution = False
                                                     , UCB_solution = False
                                                     , real = self.real, n_female = self.n_female
                                                     , cold_start = self.cold_start, item_update_per100 = self.item_update_per100
                                                     , item_update_freq = self.item_update_freq
                                                     , sparse = self.sparse, sparse_p = self.sparse_p_list[1]
                                                     , spike_slab = self.spike_slab)
        
        MTTS_agent_sparse_p2 = _agent_SemiBandit_MTB.MTB_agent(sigma_2 = sigma_2, L=L, L_tot=self.L_tot, T = T
                                                     , gamma_prior_mean = self.mu_gamma, gamma_prior_cov = self.Sigma_gamma
                                                     , sigma_1 = sigma_1
                                                     , K = K
                                                     , Xs = self.exp.env.Phi 
                                                     , update_freq = self.MTS_freq
                                                     , approximate_solution = False
                                                     , UCB_solution = False
                                                     , real = self.real, n_female = self.n_female
                                                     , cold_start = self.cold_start, item_update_per100 = self.item_update_per100
                                                     , item_update_freq = self.item_update_freq
                                                     , sparse = self.sparse, sparse_p = self.sparse_p_list[2]
                                                     , spike_slab = self.spike_slab)

        ####################################################################################################
        agents = {
            "P = "+str(self.sparse_p_list[0]) : MTTS_agent_sparse_p0
            , "P = "+str(self.sparse_p_list[1]) : MTTS_agent_sparse_p1
            , "P = "+str(self.sparse_p_list[2]) : MTTS_agent_sparse_p2
        }
        if not self.real:
            agents['Oracle TS'] = oracle_TS
        if self.used_agends != 'all':
            agents = {key : agents[key] for key in self.used_agends}

        self.exp._init_agents(agents)
        
        self.exp.run()  
        self.record = {0: self.exp.record}  
        self.agents = self.exp.agents   
        #if "MTB" in agents: 
        #    return [self.exp.record, self.exp.agents['MTB'].recorder]  
        #else:   
        #    return [self.exp.record, None] 
        return self.exp.record 
    #############################################################################################################################################
    #############################################################################################################################################        
    def run_one_seed_MNL(self, seed):
        L, T, phi_beta, K, p = self.L, self.T, self.phi_beta, self.K, self.p
        if self.real:
            self.exp = Experiment(L, K, T, self.mu_gamma, self.Sigma_gamma
                     , X_mu = self.X_mu, X_sigma = self.X_Sigma, phi_beta = phi_beta
                     , same_reward = self.same_reward, seed = seed
                     , p = p, with_intercept = self.with_intercept, clip=self.clip
                     , Bandit_type=self.Bandit_type, real = self.real
                    , based_param = self.based_param, true_gamma = self.true_gamma, true_v = self.true_v, X = self.X
                    , fixed_gamma0 = self.fixed_gamma0, skip_run = self.skip_run)
        else:
            self.exp = Experiment(L, K, T, self.mu_gamma, self.Sigma_gamma
                     , X_mu = self.X_mu, X_sigma = self.X_Sigma, phi_beta = phi_beta
                     , same_reward = self.same_reward, seed = seed
                     , p = p, with_intercept = self.with_intercept, clip=self.clip
                     , Bandit_type=self.Bandit_type, real = self.real, skip_run = self.skip_run
                    , fixed_gamma0 = self.fixed_gamma0)
        
        self.r = self.exp.env.r
        ###################################### Priors ##############################################################
        # step to get the prior for TS algorithm
        if self.only_ratio: 
            return [None, None] 
        #self.ri_prior_alpha_MC_MNL, self.ri_prior_beta_MC_MNL = self.get_ri_prior(L, phi_beta, "MNL", n_rep = 500)
        self.ri_prior_alpha_MC_MNL, self.ri_prior_beta_MC_MNL = np.random.uniform(0, 1, L), np.random.uniform(0, 1, L)
        
        TS = _agent_MNL_TS_Geom.MNL_TS(L, self.r, K, u_prior_alpha = self.ri_prior_alpha_MC_MNL, u_prior_beta = self.ri_prior_beta_MC_MNL, seed = seed, same_reward = self.same_reward, clip = self.clip)
        if not self.real:
            oracle_TS = _agent_MNL_TS_Geom.MNL_TS(L, self.r, K, u_prior_alpha = self.exp.env.alpha_Beta, u_prior_beta = self.exp.env.beta_Beta, seed = seed, same_reward = self.same_reward, clip = self.clip)
        GLB_agent = _agent_MNL_Linear_TS_Geom.MNL_TS_Contextual(L, self.r, K, self.exp.env.Phi, n_init = self.n_init,
                                                               gamma_prior_mean = self.mu_gamma, true_gamma = None, 
                                                               gamma_prior_cov = self.Sigma_gamma,
                                                               update_freq=self.GLB_freq, seed = seed, pm_core = 1, same_reward = self.same_reward)

        MTTS_agent = _agent_MNL_MTTS_Geom.MNL_MTTS(L, self.r, K, self.exp.env.Phi, phi_beta = phi_beta,n_init = self.n_init,
                                    gamma_prior_mean = self.mu_gamma, true_gamma = None, true_v_mean = None, 
                                    gamma_prior_cov = self.Sigma_gamma,
                                    update_freq=self.MTS_freq, seed = seed, pm_core = 1, same_reward = self.same_reward, clip = self.clip)
        
        ####################################################################################################
        agents = {
            "Meta TS (Ours)" : MTTS_agent
            , "Feature-agnostic TS" : TS
            , "Feature-determined TS" : GLB_agent
        }
        if not self.real:
            agents['Oracle TS'] = oracle_TS
        if self.used_agends != 'all':
            #print(self.used_agends, list(agents.keys()))
            agents = {key : agents[key] for key in self.used_agends}
        
        self.exp._init_agents(agents)

        self.exp.run()
        self.record = {0: self.exp.record}
        self.agents = self.exp.agents
        #if "MTB" in agents:
        #    return [self.exp.record, self.exp.agents['MTB'].recorder]
        #else:
        #    return [self.exp.record, None]
        return self.exp.record
    
    def determine_some_priors_4_Gaussian(self, p, L, T, sigma_1):
          
        MC_gammas = np.random.multivariate_normal(self.mu_gamma, self.Sigma_gamma, 1000)[:, 1:]
        MC_gammas = np.mean(np.sum(MC_gammas ** 2, 0))
        
        var_TS = MC_gammas + 1/p + sigma_1**2
        if self.cold_start:
            self.L_tot = L + T//self.item_update_freq*self.item_update_per100
        else:
            self.L_tot = L
        self.cov_TS_diag = var_TS * np.ones(self.L_tot)

    def run_one_seed_Cascading(self, seed):
        L, T, phi_beta, K, p, Bandit_type = self.L, self.T, self.phi_beta, self.K, self.p, self.Bandit_type
        if self.real:
            self.exp = Experiment(L, K, T, self.mu_gamma, self.Sigma_gamma
                     , X_mu = self.X_mu, X_sigma = self.X_Sigma, phi_beta = phi_beta
                     , same_reward = self.same_reward, seed = seed
                     , p = p, with_intercept = self.with_intercept, clip=self.clip
                     , Bandit_type=self.Bandit_type, real = self.real
                    , based_param = self.based_param, true_gamma = self.true_gamma, true_v = self.true_v, X = self.X
                    ,true_theta = self.true_theta, W_test = self.W_test, regret_type = self.regret_type, skip_run = self.skip_run
                    , fixed_gamma0 = self.fixed_gamma0)
        else:
            self.exp = Experiment(L, K, T, self.mu_gamma, self.Sigma_gamma
                     , X_mu = self.X_mu, X_sigma = self.X_Sigma, phi_beta = phi_beta
                     , same_reward = self.same_reward, seed = seed
                     , p = p, with_intercept = self.with_intercept, clip=self.clip
                     , Bandit_type=self.Bandit_type, real = self.real, skip_run = self.skip_run
                    , fixed_gamma0 = self.fixed_gamma0)
        if self.only_ratio: 
            return [None, None] 
        ###################################### Priors ##############################################################
        #self.ri_prior_alpha_MC_cascading, self.ri_prior_beta_MC_cascading = self.get_ri_prior(L, phi_beta, "Cascading", n_rep = 50)
        self.ri_prior_alpha_MC_cascading, self.ri_prior_beta_MC_cascading = np.random.uniform(0, 1, L), np.random.uniform(0, 1, L)
        TS = _agent_Cascading_TS.TS_agent(K,u_prior_alpha = self.ri_prior_alpha_MC_cascading, u_prior_beta = self.ri_prior_beta_MC_cascading )
        if not self.real:
            oracle_TS = _agent_Cascading_TS.TS_agent(K,u_prior_alpha = self.exp.env.alpha_Beta, u_prior_beta = self.exp.env.beta_Beta)
        GLB_agent = _agent_Cascading_GLM.GLB_agent(L, K, p, alpha = self.alpha_GLB, true_gamma_4_debug = self.exp.env.gamma, retrain_freq = self.GLB_freq)
        MTTS_agent = _agent_Cascading_MTTS.MTTS_agent(start = "cold", phi_beta = phi_beta, K = K
                                                     , gamma_prior_mean = self.mu_gamma, gamma_prior_cov = self.Sigma_gamma
                                                     , Xs = self.exp.env.Phi # [L,p]
                                                     , update_freq = self.MTS_freq, n_init = self.n_init
                                                    , u_prior_alpha = None, u_prior_beta = None
                                                     , true_gamma_4_debug = self.exp.env.gamma, gamma_r_4_debug = None
                                                     , exp_seed = seed)

        ####################################################################################################
        agents = {
            "Meta TS (Ours)" : MTTS_agent
            , "Feature-agnostic TS" : TS
            , "Feature-determined TS" : GLB_agent
        }
        if not self.real:
            agents['Oracle TS'] = oracle_TS
        if self.used_agends != 'all':
            agents = {key : agents[key] for key in self.used_agends}

        self.exp._init_agents(agents)
        
        self.exp.run()  
        self.record = {0: self.exp.record}  
        self.agents = self.exp.agents   
        #if "MTB" in agents: 
        #    return [self.exp.record, self.exp.agents['MTB'].recorder]  
        #else:   
        #    return [self.exp.record, None] 
        return self.exp.record 
    
    def run_one_seed_SemiBandit(self, seed):
        L, T, sigma_1, sigma_2, K, p, Bandit_type = self.L, self.T, self.sigma_1, self.sigma_2, self.K, self.p, self.Bandit_type
        if self.real:
            self.exp = Experiment(L, K, T, self.mu_gamma, self.Sigma_gamma
                                 , X_mu = self.X_mu, X_sigma = self.X_Sigma, phi_beta = None
                                 , sigma_1 = sigma_1, sigma_2 = sigma_2
                                 , same_reward = self.same_reward, seed = seed
                                 , p = p, with_intercept = self.with_intercept, clip=self.clip
                                 , Bandit_type=self.Bandit_type, real = self.real
                                 , true_theta=self.true_theta, true_gamma=self.true_gamma, X=self.X
                                  , n_female = self.n_female, skip_run = self.skip_run
                                , misspecification = self.misspecification)
        else:
            self.exp = Experiment(L, K, T, self.mu_gamma, self.Sigma_gamma
                 , X_mu = self.X_mu, X_sigma = self.X_Sigma, phi_beta = None
                 , sigma_1 = sigma_1, sigma_2 = sigma_2
                 , same_reward = self.same_reward, seed = seed
                 , p = p, with_intercept = self.with_intercept, clip=self.clip, skip_run = self.skip_run
                 , Bandit_type=self.Bandit_type, real = self.real, misspecification = self.misspecification
                , cold_start = self.cold_start, item_update_per100 = self.item_update_per100,item_update_freq = self.item_update_freq)

        if self.only_ratio: 
            return [None, None] 
        ###################################### Priors ##############################################################
        self.determine_some_priors_4_Gaussian(p, L, T, sigma_1)
        
        TS = _agent_SemiBandit_TS.TS_agent(L,K,u_prior_mean = np.zeros(self.L_tot), u_prior_cov_diag = self.cov_TS_diag, real = self.real, n_female = self.n_female, cold_start = self.cold_start, item_update_per100 = self.item_update_per100, item_update_freq = self.item_update_freq)
        if not self.real:
            oracle_TS = _agent_SemiBandit_TS.TS_agent(L, K, u_prior_mean = self.exp.env.theta_mean, u_prior_cov_diag = sigma_1 ** 2 * np.ones(self.L_tot), cold_start = self.cold_start, item_update_per100 = self.item_update_per100, item_update_freq = self.item_update_freq)
        LB_agent = _agent_SemiBandit_LB.LB_agent(sigma = np.sqrt(sigma_1**2+sigma_2**2), prior_gamma_mu = self.mu_gamma, prior_gamma_cov = self.Sigma_gamma
                                                  , L = L, L_tot = self.L_tot, K = K, p = p, real = self.real, n_female = self.n_female, cold_start = self.cold_start
                                                 , item_update_per100 = self.item_update_per100, item_update_freq = self.item_update_freq)
        MTTS_agent = _agent_SemiBandit_MTB.MTB_agent(sigma_2 = sigma_2, L=L, L_tot=self.L_tot, T = T
                                                     , gamma_prior_mean = self.mu_gamma, gamma_prior_cov = self.Sigma_gamma
                                                     , sigma_1 = sigma_1
                                                     , K = K
                                                     , Xs = self.exp.env.Phi 
                                                     , update_freq = self.MTS_freq
                                                     , approximate_solution = False
                                                     , UCB_solution = False
                                                     , real = self.real, n_female = self.n_female
                                                     , cold_start = self.cold_start, item_update_per100 = self.item_update_per100, item_update_freq = self.item_update_freq)

        ####################################################################################################
        agents = {
            "Meta TS (Ours)" : MTTS_agent
            , "Feature-agnostic TS" : TS
            , "Feature-determined TS" : LB_agent
        }
        if not self.real:
            agents['Oracle TS'] = oracle_TS
        if self.used_agends != 'all':
            agents = {key : agents[key] for key in self.used_agends}

        self.exp._init_agents(agents)
        
        self.exp.run()  
        self.record = {0: self.exp.record}  
        self.agents = self.exp.agents   
        #if "MTB" in agents: 
        #    return [self.exp.record, self.exp.agents['MTB'].recorder]  
        #else:   
        #    return [self.exp.record, None] 
        return self.exp.record 


    #############################################################################################################################################
    #############################################################################################################################################
    def run_multiple_parallel_in_batch(self, reps, batch = 1, parallel = 'parmap'):
        reps_each_batch = int(reps // batch)
        record = []
        for b in range(batch):
            print("batch = {}".format(b))
            r = self.run_multiple_parallel(reps_each_batch, init_seed = reps_each_batch * b, parallel = parallel)
            record += self.record
        self.record = record

    def run_multiple_parallel(self, reps, init_seed = 0, parallel = 'parmap'):
        rep = reps
        with open('log/{}.txt'.format(self.date_time), 'w') as f:
            print(self.title_settting, file=f)

        import ray
        if parallel == 'parmap':
            if self.testtest:
                record = parmap(self.run_one_testtest, range(init_seed, rep + init_seed))
            elif self.sparse and self.Bandit_type == "Cascading":
                record = parmap(self.run_one_seed_sparse_cascade, range(init_seed, rep + init_seed))
            elif self.sparse and self.Bandit_type == "SemiBandit":
                record = parmap(self.run_one_seed_sparse_SemiBandit, range(init_seed, rep + init_seed))
            elif self.sparse and self.Bandit_type == "MNL":
                record = parmap(self.run_one_seed_sparse_MNL, range(init_seed, rep + init_seed))
            elif self.Bandit_type == "MNL":
                record = parmap(self.run_one_seed_MNL, range(init_seed, rep + init_seed))
            elif self.Bandit_type == "Cascading":
                record = parmap(self.run_one_seed_Cascading, range(init_seed, rep + init_seed))
            elif self.Bandit_type == "SemiBandit":
                record = parmap(self.run_one_seed_SemiBandit, range(init_seed, rep + init_seed))   
        if parallel == 'ray':
            ray.shutdown()
            @ray.remote(num_cpus = 6) # num_cpus = 3
            def one_seed(seed):
                os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
                os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
                os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
                os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
                os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
                if self.testtest:
                    r = self.run_one_testtest(seed)
                elif self.sparse and self.Bandit_type == "Cascading":
                    r = self.run_one_seed_sparse_cascade(seed) 
                elif self.sparse and self.Bandit_type == "SemiBandit":
                    r = self.run_one_seed_sparse_SemiBandit(seed) 
                elif self.sparse and self.Bandit_type == "MNL":
                    r = self.run_one_seed_sparse_MNL(seed) 
                elif self.Bandit_type == "MNL":
                    r = self.run_one_seed_MNL(seed)
                elif self.Bandit_type == "Cascading":
                    r = self.run_one_seed_Cascading(seed)       
                elif self.Bandit_type == "SemiBandit":
                    r = self.run_one_seed_SemiBandit(seed)
                return r
            ray.init()
            ##########
            futures = [one_seed.remote(j) for j in range(init_seed, rep + init_seed)]
            record = ray.get(futures)
            ray.shutdown()
        self.record = record
        #if not self.only_ratio: 
            #self.record = [r[0] for r in record]
            #self.record_MTB = [r[1] for r in record]
        
    #############################################################################################################################################
    #############################################################################################################################################
    def plot_regret(self, skip_methods = []
                    , ci = None, freq = 20
                   , plot_mean = False, skip = 2
                   , y_min = None, y_max = None):
        from matplotlib.transforms import BlendedGenericTransform

        # https://seaborn.pydata.org/generated/seaborn.lineplot.html
        #ax.legend(['label 1', 'label 2'])
        self.fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 2))

        COLORS = sns.color_palette("Set2")
        palette = {name : color for name, color in zip(self.names, COLORS)}

        n_methods = 6

        reps = len(self.record)
        # stacked over seeds. for each seed, according to the orders 
        # how to select should depends on self.setting['order'] * purpose
        # and how to calculate the average. 
        # if episodic: T first, then N  (trend with N) -> MR
        # if concurrent: N first, then T (trend with T) -> BR
        # BR: trend with T
        # MR: trend with N
        # data_plot_BR.time = j: according to (i, t), not (t)

        data = pd.concat([self.record[seed]['cum_regret_df'] for seed in range(reps)])
        if self.real or self.sparse:
            data_meta = data
        else:
            data_meta = pd.concat([self.record[seed]['cum_meta_regret_df'] for seed in range(reps)])
        data_plot_meta = data_meta[data_meta.method != "Oracle TS"]


        data_plot_BR = data
        data_plot_BR.time = np.tile(np.arange(0, self.setting['T']), len(data_plot_BR) // self.setting['T'])
        data_plot_meta = data_plot_meta
        data_plot_meta.time = np.tile(np.arange(0, self.setting['T']), len(data_plot_meta) // self.setting['T'])

        self.data_plot_BR_original = data_plot_BR.copy()
        self.data_plot_meta_original = data_plot_meta.copy()

        if plot_mean:
            data_plot_BR.regret = data_plot_BR.regret / (data_plot_BR.time + 1)
            data_plot_meta.regret = data_plot_meta.regret / (data_plot_meta.time + 1)

        if skip_methods is not None:
            for met in skip_methods:
                data_plot_BR = data_plot_BR[data_plot_BR.method != met]
                data_plot_meta = data_plot_meta[data_plot_meta.method != met]

        data_plot_BR = data_plot_BR[data_plot_BR.time >= skip]
        data_plot_meta = data_plot_meta[data_plot_meta.time >= skip]

        data_plot_BR.reset_index(inplace = True)
        ax1 = sns.lineplot(data=data_plot_BR
                     , x="time", y="regret"
                     , hue="method" # group variable
                    , ci = ci # 95
                    , ax = ax1
                           , n_boot = 20
                    , palette = palette
                    )
        ax1.set(ylim=(y_min, y_max))


        ax1.set_title('Bayes regret')
        ax1.legend().texts[0].set_text("Method")
        
        data_plot_meta.reset_index(inplace = True)
        ax2 = sns.lineplot(data=data_plot_meta
                     , x="time", y="regret"
                     , hue="method" # group variable
                    , ci = ci # 95
                    , ax = ax2
                           , n_boot = 20
                    , palette = palette
                    )
        ax2.set(ylim=(y_min, None))


        ax2.set_title('Multi-task regret')
        ax2.legend().texts[0].set_text("Method")
        
        ax1.set_xlabel('T')
        ax2.set_xlabel('T')

        self.fig.suptitle(self.title_settting, fontsize=12, y = 1.1)

        handles, labels = ax1.get_legend_handles_labels()
        self.fig.legend(handles, labels, loc='lower center', ncol = len(labels)
                       , bbox_to_anchor=(0.5, -0.25))
        ax1.get_legend().remove()
        ax2.get_legend().remove()
        plt.show()

    #############################################################################################################################################
        #############################################################################################################################################
    def save(self, main_path = "res/", fig_path = "Fig/", sub_folder = [], no_care_keys = []
            , only_plot_matrix = 1):
        """
        Since all results together seems quite large
        a['record'][0].keys() = ([exp_R','R', 'A', 'regret', 'meta_regret', 'cum_regret', 'cum_meta_regret'])
        regret / R can almost derive anything, except for A
        The only thing is that, we may need to re-read them. Probably we need a function, to convert a "skim recorder" to a "full recorder", when we do analysis.
        """
        ########################################################
        date = get_date()

        result_path = main_path + date
        fig_path = fig_path + date
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        ########################################################
        aa = self.Bandit_type
        fig_path += "/" + aa 
        result_path += "/" + aa 
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        ########################################################
        if len(sub_folder) > 0:
            fig_path += "/" 
            result_path += "/"
            for key in sub_folder:
                fig_path += ("_" + str(key) + str(self.setting[key]))
                result_path += ("_" + str(key) + str(self.setting[key]))
                no_care_keys.append(key)
        no_care_keys.append('save_prefix')
                           
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)
        if not os.path.isdir(result_path):
            os.mkdir(result_path)

        ############################
        if only_plot_matrix:
            record_exp_R_only = {"data_plot_BR_original" : self.data_plot_BR_original
                            , "data_plot_meta_original" : self.data_plot_meta_original}
        else:
            record_exp_R_only = {seed : self.record[seed]['exp_R'] for seed in range(len(self.record))}

        r = {"setting" : self.setting
             , "record" : record_exp_R_only
            , "name" : self.names}

        ############################
        path_settting = "_".join([str(key) + str(self.setting[key]) for key in self.setting if type(key) in [str, int] and type(self.setting[key]) in [str, int, float] and key not in no_care_keys])
        print(path_settting)
        if self.save_prefix:
            path_settting = path_settting + "-" + self.save_prefix

        ############################
        r_path = result_path + "/"  + path_settting
        fig_path = fig_path + "/"  + path_settting + ".png"
        print("save to {}".format(r_path))
        self.fig.savefig(fig_path)
        dump(r,  r_path)
