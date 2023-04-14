from _util import *
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from matplotlib.transforms import BlendedGenericTransform
from matplotlib.offsetbox import AnchoredText
def get_tableau20():
    # These are the "Tableau 20" colors as RGB.   
    # , (174, 199, 232)
    tableau20 = [(31, 119, 180), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)    
    return tableau20

##################################################################################################################################################################################################################################################################################################################

def load_and_plot_ax(path, ax, target
                  , plot_mean = 0, title = None
                  , skip_methods = None, freq = 100
                  , skip = 0, ylabel = None, ylabel_size = 12, title_size = 12
                  , ci = None, names = ["Meta TS (Ours)", "Feature-agnostic TS", "Oracle TS", "Feature-determined TS"]
                  , y_min = None, y_max = None):
    a = load(path)
    data_plot_BR = a['record']['data_plot_BR_original'].copy()
    #data_plot_meta = a['record']['data_plot_meta_original'].copy()

    COLORS = sns.color_palette("Set2")

    palette = {name : color for name, color in zip(names, COLORS)}

    n_methods = 6
    

    if plot_mean:
        data_plot_BR.regret = data_plot_BR.regret / (data_plot_BR.time + 1)
        #data_plot_meta.regret = data_plot_meta.regret / (data_plot_meta.time + 1)

    if skip_methods is not None:
        for met in skip_methods:
            data_plot_BR = data_plot_BR[data_plot_BR.method != met]
            #data_plot_meta = data_plot_meta[data_plot_meta.method != met]

    data_plot_BR = data_plot_BR[data_plot_BR.time >= skip]
    #data_plot_meta = data_plot_meta[data_plot_meta.time >= skip]

    data_plot_BR = data_plot_BR[data_plot_BR.time % freq == 0]
    #data_plot_meta = data_plot_meta[data_plot_meta.time % freq == 0]
    data_plot_BR.reset_index(inplace = True)
    
    if target == 'BR':
        ax = sns.lineplot(data=data_plot_BR
                     , x="time", y="regret"
                     , hue="method" # group variable
                    , ci = ci # 95
                    , ax = ax
                           , n_boot = 20
                    , palette = palette
                    )
        ax.set(ylim=(y_min, y_max))

        ax.legend().texts[0].set_text("Method")

#     if target == 'MR':
#         ax = sns.lineplot(data=data_plot_meta
#                      , x="time", y="regret"
#                      , hue="method" # group variable
#                     , ci = ci # 95
#                     , ax = ax
#                            , n_boot = 20
#                     , palette = palette
#                     )
#         ax.set(ylim=(y_min, None))


#         ax.legend().texts[0].set_text("Method")
#         # fig.suptitle(self.title_settting, fontsize=12, y = 1.1)
    ax.set_title(title, fontsize= title_size)        
    ax.set_xlabel('T')
    ax.set_ylabel(ylabel, fontsize= ylabel_size)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', ncol = len(labels))
    ax.get_legend().remove()
    return handles, labels
##################################################################################################################################################################################################################################################################################################################

def load_and_plot(path
                  , plot_mean = 0
                  , skip_methods = None
                  , skip = 0
                  , ci = None, names = ["Meta TS (Ours)", "Feature-agnostic TS", "Oracle TS", "Feature-determined TS"]
                  , y_min = None, y_max = None, reps = 8):
    a = load(path)
    data_plot_BR = a['record']['data_plot_BR_original'].copy()
    data_plot_meta = a['record']['data_plot_meta_original'].copy()
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 2))

    COLORS = sns.color_palette("Set2")

    palette = {name : color for name, color in zip(names, COLORS)}

    n_methods = 6
    

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
    # fig.suptitle(self.title_settting, fontsize=12, y = 1.1)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol = len(labels))
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    plt.show()

##############################################################################################################################################################################################################################################################################################################################################################################

# class Analyzer():
#     def __init__(self):
#         pass

#     def recover_full_recorder(self, record_R_only = None, only_plot_matrix = True):
#         if only_plot_matrix:
#             self.names = self.agent_names = record_R_only['name']
#         else:
#             self.names = self.agent_names = agent_names = list(record_R_only['record'][0].keys())
#         self.record = {}
#         self.setting = record_R_only['setting']
#         self.record_R_only = record_R_only
#         self.only_plot_matrix = only_plot_matrix
#         if not only_plot_matrix:
#             seeds = list(record_R_only['record'].keys())
#             for seed in tqdm(seeds):
#                 self.record[seed] = self.recover_full_recorder_one_seed(seed)
                        
#     def recover_full_recorder_one_seed(self, seed):
#         r = {}
#         for metric in ['R', 'A', 'regret', 'meta_regret']:
#             r[metric] = {name : [] for name in self.agent_names}

#         r['R'] = self.record_R_only['record'][seed]
#         for name in self.agent_names:
#             r['regret'][name] = arr(r['R']["oracle"]) - arr(r['R'][name])
#         r['cum_regret'] = {name : np.cumsum(r['regret'][name]) for name in self.agent_names}
#         # x: time, y: cum_regret: group, name
#         if "oracle-TS" in self.agent_names:
#             for name in self.agent_names:
#                 r['meta_regret'][name] = arr(r['R']['oracle-TS']) - arr(r['R'][name])
#             r['cum_meta_regret'] = {name : np.cumsum(r['meta_regret'][name]) for name in self.agent_names}
#         return r

                
#     def organize_Df(self, r_dict):
#         T = len(r_dict[self.agent_names[0]])
#         a = pd.DataFrame.from_dict(r_dict)
#         # a.reset_index(inplace=True)
#         a = pd.melt(a)
#         a['time'] = np.tile(np.arange(T), len(self.agent_names))
#         a = a.rename(columns = {'variable':'method'
#                            , "value" : "regret"
#                            , "time" : "time"})
#         return a

#     def prepare_data_4_plot(self, skip_methods = [], plot_mean = None, skip = None, plot_which = None):
#         if plot_which == "auto":
#             if self.setting['order'] == "episodic":
#                 plot_which = "MR"
#             else:
#                 plot_which = "BR"
#         self.plot_which = plot_which
#         # https://seaborn.pydata.org/generated/seaborn.lineplot.html
#         #ax.legend(['label 1', 'label 2'])

#         n_methods = 7# - len(skip_methods)

#     #         self.labels_BR = ["individual-TS", "linear-TS", "oracle-TS", "meta-TS", "MTS"]
#     #         self.labels_MR = ["individual-TS", "linear-TS", "oracle-TS", "MTS"]

#         reps = len(self.record)

#         ########
#         if self.only_plot_matrix:
#             data_plot_BR = self.record_R_only['record']['data_plot_BR_original']
#             data_plot_meta = self.record_R_only['record']['data_plot_meta_original']
#         else:
#             data = pd.concat([self.organize_Df(self.record[seed]['cum_regret']) for seed in range(reps)])
#             data_meta = pd.concat([self.organize_Df(self.record[seed]['cum_meta_regret']) for seed in range(reps)])
#             data_plot_meta = data_meta[data_meta.method != "oracle-TS"]


#             if self.setting['order'] == "episodic":
#                 data_plot_BR = data.iloc[np.arange(0, len(data), step = self.setting['T']) + self.setting['T'] - 1]   
#                 data_plot_BR.time = np.tile(np.arange(0, self.setting['N'])
#                                             , len(data_plot_BR) // self.setting['N'])
#                 data_plot_meta = data_plot_meta.iloc[np.arange(0, len(data_plot_meta), step = self.setting['T']) + self.setting['T'] - 1] 
#                 data_plot_meta.time = np.tile(np.arange(0, self.setting['N'])
#                                             , len(data_plot_meta) // self.setting['N'])


#             elif self.setting['order'] == "concurrent":
#                 data_plot_BR = data.iloc[np.arange(0, len(data), step = self.setting['N']) + self.setting['N'] - 1]   
#                 data_plot_BR.time = np.tile(np.arange(0, self.setting['T'])
#                                             , len(data_plot_BR) // self.setting['T'])
#                 data_plot_meta = data_plot_meta.iloc[np.arange(0, len(data_plot_meta), step = self.setting['N']) + self.setting['N'] - 1] 
#                 data_plot_meta.time = np.tile(np.arange(0, self.setting['T'])
#                                             , len(data_plot_meta) // self.setting['T'])

#         self.data_plot_BR_original = data_plot_BR.copy()
#         self.data_plot_meta_original = data_plot_meta.copy()

#         if plot_mean:
#             data_plot_BR.regret = data_plot_BR.regret / (data_plot_BR.time + 1)
#             data_plot_meta.regret = data_plot_meta.regret / (data_plot_meta.time + 1)

#         data_plot_BR = data_plot_BR.rename(columns = {'method':'Method'})
#         data_plot_meta = data_plot_meta.rename(columns = {'method':'Method'})



#         # exp.plot_regret()
#         skip_methods.append("oracle")
#         if skip_methods is not None:
#             for met in skip_methods:
#                 data_plot_BR = data_plot_BR[data_plot_BR.Method != met]
#                 data_plot_meta = data_plot_meta[data_plot_meta.Method != met]

#         data_plot_BR = data_plot_BR[data_plot_BR.time >= skip]
#         data_plot_meta = data_plot_meta[data_plot_meta.time >= skip]

        
#         return data_plot_BR, data_plot_meta

#     def plot_regret(self, skip_methods = ["OSFA"]
#                     , ci = None, freq = 20
#                    , plot_which = "both", plot_mean = False, skip = 2
#                    , n_boot = 50, ylabel = "Average regret"
#                    , ax1 = None, w_title = False, y_min = None, y_max = None, i = 0, new_title = None
#                    , color_shift = 0, palette_idx = None
#                    , linewidth = 2, no_xtick = False
#                     , hue_order = None
#                     , complex_x_label = True
#                    ):
#         from matplotlib.transforms import BlendedGenericTransform
#         #COLORS = sns.color_palette("Set1")
# #         COLORS = sns.color_palette()
#         COLORS = get_tableau20() #sns.color_palette("tab10")
    
#         if palette_idx is None:
#             def rotate(l, n):
#                 n = -n
#                 return l[n:] + l[:n]
#             palette = {name : color for name, color in zip(rotate(self.names, color_shift), COLORS)}
#         else:
#             palette = {name : COLORS[idx] for name, idx in palette_idx.items()}

#         data_plot_BR, data_plot_meta = self.prepare_data_4_plot(skip_methods = skip_methods
#                                                                 , plot_mean = plot_mean, skip = skip, plot_which = plot_which)


#         if plot_which == "BR":
#             data_plot = data_plot_BR
#             title = 'Bayes regret'
#         else:
#             data_plot = data_plot_meta
#             title = 'Multi-task regret'
        
#         if complex_x_label:
#             if self.setting['order'] == "episodic":
#                 x_label = 'N (number of tasks)'
#             else:
#                 x_label = 'T (number of interactions)'
#         else:
#             if self.setting['order'] == "episodic":
#                 x_label = 'N'
#             else:
#                 x_label = 'T'


#         ##########################################
#         line = sns.lineplot(data=data_plot
#                      , x = "time", y="regret", hue="Method" # group variable
#                     , ci = ci # 95, n_boot= n_boot
#                     , ax = ax1
#                     , n_boot = 100
#                     , palette = palette
# #                     , err_style = "bars" #“band” or “bars”
#                     , linewidth = linewidth
#                     , hue_order = hue_order
#                     )
#         if no_xtick:
#             ax1.set_xticks([])


#         ax1.legend().texts[0].set_text("Method")
#         if w_title:
#             if new_title is None:
#                 ax1.set_title(title, fontsize= 14)
#             else:
#                 ax1.set_title(new_title, fontsize= 14)
#         ax1.set_xlabel(x_label, fontsize= 12)
#         if i == 0:
#             ax1.set_ylabel(ylabel, fontsize= 14)
#         else:
#             ax1.set_ylabel(None, fontsize= 12)
#         ax1.set(ylim=(y_min, y_max))
#         #########
#         handles, labels = ax1.get_legend_handles_labels()
# #         self.ax1 = ax1
# #         plt.show()
#         ax1.get_legend().remove()
    


#         return handles, labels

# #         title_settting = " ".join([str(key) + "=" + str(self.setting[key]) for key in self.setting if type(key) in [str, int] and type(self.setting[key]) in [str, int, float]])
# #         printR(title_settting)
#         #self.fig.suptitle(title_settting, fontsize=12, y = 1.1)

# #         anc = AnchoredText(str(self.setting['order']) + "\n" + str(self.setting['sigma1_square']), loc="upper right", frameon=True
# # #                           , set_alpha =1
# #                           )
# #         ax1.add_artist(anc)

#     #                 , ncol=7, numpoints=1
#     #                 , fontsize = 16
#     #                             , title = "test"
#     #                     , labels= self.labels_BR
#     #             self.fig.legend(handles, labels, loc='lower center', ncol = len(labels))

#     ####################################################################################################
#     def save(self, fig_path = "fig/", sub_folder = [], no_legend = True):
#         """
#         Since all results together seems quite large
#         a['record'][0].keys() = (['R', 'A', 'regret', 'meta_regret', 'cum_regret', 'cum_meta_regret'])

#         regret / R can almost derive anything, except for A

#         The only thing is that, we may need to re-read them. Probably we need a function, to convert a "skim recorder" to a "full recorder", when we do analysis.
#         """
#         date = get_date()
#         fig_path = fig_path + date
#         if len(sub_folder) > 0:
#             fig_path += "/"
#             for key in sub_folder:
#                 fig_path += ("_" + str(key) + str(self.setting[key]))
#         if not os.path.isdir(fig_path):
#             os.mkdir(fig_path)

#         path_settting = "_".join([str(key) + str(self.setting[key]) for key in self.setting if type(key) in [str, int] and key not in sub_folder and type(self.setting[key]) in [str, int, float]])

#         print(path_settting)
#         if no_legend:
#             self.ax1.get_legend().remove()

#         self.fig.savefig(fig_path + "/"  + path_settting + ".png"
#                                , bbox_inches= 'tight'
#     #                      , dpi=200
#     #                      , bbox_extra_artists = ["y_label"]
#     #                      , pad_inches = 0
#                         )

#     ####################################################################################################

#     def save_legend(self):
#         handles,labels = self.ax1.get_legend_handles_labels()
#         fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(3, 2))
#         axe.legend(handles, labels
#                 , ncol=7, numpoints=1
#                   )
#         axe.xaxis.set_visible(False)
#         axe.yaxis.set_visible(False)
#         for v in axe.spines.values():
#             v.set_visible(False)
#         fig.savefig("fig/legend.png"
#                     , bbox_inches= 'tight'
#                     , pad_inches = 0
#                         )
# #     def skim_recorder(self, path = None):
# #         full_recorded = load(path)
# #         record_R_only = {seed : r_one_seed['R'] for seed, r_one_seed in enumerate(full_recorded['record'])}
        
# #         r = {"setting" : full_recorded["setting"]
# #              , "record": record_R_only}
        
# #         dump(r, path)

