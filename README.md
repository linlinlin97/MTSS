# Towards Scalable and Robust Structured Bandits: A Meta-Learning Framework

This repository is the official implementation of the paper [Towards Scalable and Robust Structured Bandits: A Meta-Learning Framework](https://arxiv.org/pdf/2202.13227.pdf) in Python. 

>📋  **Abstract**: Online learning in large-scale structured bandits is known to be challenging due to the curse of dimensionality. In this paper, we propose a unified meta-learning framework for a general class of structured bandit problems where the  parameter space can be factorized to item-level. The novel bandit algorithm is general to be applied to many popular structured bandit problems, scalable to the huge parameter and action spaces, and robust to the generalization model specification. At the core of this framework is a Bayesian hierarchical model that allows information sharing among items via their features, upon which we design a meta Thompson sampling algorithm. Three representative examples are discussed thoroughly. Both theoretical and numerical results support the usefulness of the proposed method. 

## Agents for Different Structured Bandit Problems. (MTSS and Other Baseline Algorithms)
1. **Cascading Bandits**: within the folder `/Agents_Cascading`, there are codes for three different Thompson Sampling-based methods for Cascading Bandits problems
    1. `_agent_Cascading_GLM.py`: the feature-determined algorithm. 
    2. `_agent_Cascading_MTTS.py`: the proposed feature-guided algorithm MTSS. (Note: we use MTSS and MTTS interchangeably)
    3. `_agent_Cascading_TS.py`: the feature-agnostic algorithm.
2. **MNL Bandits**: within the folder `/Agents_MNL`, there are codes for three different Thompson Sampling-based methods for MNL Bandit problems
    1. `_agent_MNL_Linear_TS_geometric.py`: the feature-determined algorithm. 
    2. `_agent_MNL_MTTS_geometric.py`: the proposed feature-guided algorithm MTSS. 
    3. `_agent_MNL_TS_geometric.py`: the feature-agnostic algorithm.
3. **Semi-Bandits**: within the folder `/Agents_SemiBandit`, there are codes for three different Thompson Sampling-based methods for Semi-Bandits problems
    1. `_agent_SemiBandit_LB.py`: the feature-determined algorithm. 
    2. `_agent_SemiBandit_MTB.py`: the proposed feature-guided algorithm MTSS. 
    3. `_agent_SemiBandit_LB.py`: the feature-agnostic algorithm.

## Functions for Experiments
1. **Environments**: Within the folder `/Envs`, there are codes for generating the environments for synthetic/real experiments. 
    1. `_env_(Cascading/MNL/SemiBandit).py` are the synthetic experiments' environments under corresponding problem structures
    2. `_env_(Cascading/MNL/SemiBandit)_Real_Exp.py` are the real experiments' environments under corresponding problem structures.
2. **Other Functions Required**: The following three code files, in the main folder, are used to conduct the experiments and get the results under different problem structures.
    1. `_util.py`: helper functions.
    2. `_Experiement.py`: function to run different experiments.
    3. `_analyzer.py`: post-process simulation results.

## Scripts to Conduct Synthetic Experiments
In the main folder, 
1. `Simulation.ipynb`: Script to reproduce the simulation results showed in **Figure1** in Section 6 and **Figure3-5** in the Appendix.
2. `/Simu_Sparse`: There are the codes used for the experiments showing how the proposed method can be extended to address the issue of sparsity.

## Scripts to Conduct Real Experiments
### Data Pre-process
Within the folder `/Real_Analysis`, there are three subfolders containing data and scripts used for the real expriments.
1. `/Cascading`: For the cascading bandit problem, we focus on the [Yelp](https://www.yelp.com/dataset) dataset. 
    1. `Yelp Dataset.ipynb`: script used to preprocess the raw dataset
    2. `Cascading_W_test.zip`: include the observations in W_test for the Cascading problem
    3. `Cascading_realdata_d_10_X_transform_standardize_with_intercept_1`: include features and the true $\phi$ learned from the dataset
2. `/MNL`:  For the MNL bandit problem, we focus on the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset. 
    1. `MNL_Feature_Extraction.ipynb`: script used to preprocess the raw dataset
    2. `MNL_realdata_d_5_X_transform_l2_with_intercept_1`: include features and true utility and true $\phi$ learned from the dataset
3. `/SemiBandit`:  For the Semi-Bandit problem, we focus on the [Adult](https://archive.ics.uci.edu/ml/datasets/adult) dataset. 
    1. `Adult.ipynb`: script used to preprocess the raw dataset
    2. `Semi_realdata_d_4_X_transform_origin_with_intercept_0_L_3000`: include features, true $\theta$ and $\sigma_{1}$ learned from the dataset

### Other
In the main folder, the `Real_data.ipynb` file includes the template used to reproduce the results of real experiment showed in **Figure2**. 

## Script to Analyze the Results
To generate the plots(figures) included in the paper, the following script is used.
1. `plot.ipynb`: script to reproduce the **Figure1--5**.

## Steps to Reproduce the Experiments Results
1. Install the required packages included in the `_util.py`; 
2. Download all the required codes in the same folder (Main Folder);
3. Within the Main Folder, create two empty folders `/res` and `/log` to save simulation results and create another empty folder `/fig` to save figures;
4. Run the corresponding experiment scripts to get the simulation/real experiment results;
5. Analyze the results and get the figure by running the corresponding code in the `plot.ipynb`.
