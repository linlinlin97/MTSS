#############################################################################
# %matplotlib inline
# !tar -czf data.zip data
# !tar -czf code.zip code
from zipfile import ZipFile
from scipy.sparse.linalg import eigs
from typing import Dict, List, Set, Tuple
from datetime import datetime as dt
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from importlib import reload
# import boto3
import io
import sys
import warnings
warnings.simplefilter("ignore")
import logging
# logging.warning('Hah')
# from sagemaker import get_execution_role
# role = get_execution_role()
from IPython.display import display, clear_output, display_html
import collections
from types import MethodType
import functools 
from functools import reduce 
from scipy import stats
import random

from scipy.optimize import linprog

from scipy import stats
import random
import copy

# import pymc3 as pm
# from pymc3.distributions import Interpolated


# import shutil
# shutil.rmtree("@@MTB@@/res/0316") 
#############################################################################
# Packages
import scipy as sp
import pandas as pd
from pandas import DataFrame as DF
import statsmodels.api as sm
# import statsmodels as sm
from matplotlib.pyplot import hist
import pickle
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.ensemble import GradientBoostingRegressor as GBT
####################################
import seaborn as sns
from scipy.linalg import block_diag

from tqdm import tqdm
# Random
import random
from random import seed as rseed
from numpy.random import seed as npseed
from numpy import absolute as np_abs
from numpy.random import normal as rnorm
from numpy.random import uniform as runi
from numpy.random import binomial as rbin
from numpy.random import poisson as rpoisson
from numpy.random import shuffle,randn, permutation # randn(d1,d2) is d1*d2 i.i.d N(0,1)
from numpy import squeeze
from numpy.linalg import solve
####################################

# Numpy
import numpy as np
from numpy import mean, var, std, median
from numpy import array as arr
from numpy import sqrt, log, cos, sin, dot, diag, ones, identity, zeros, roll, multiply, stack, concatenate, transpose
# exp, 
from numpy import concatenate as v_add
from numpy.linalg import norm, inv
from numpy import apply_along_axis as apply
from numpy.random import multinomial, choice
####################################

# sklearn
import sklearn as sk
from sklearn import preprocessing as pre
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression as lm
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#############################################################################
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

np.set_printoptions(precision = 4)
#############################################################################
import time
now = time.time
import smtplib, ssl

import datetime, pytz

def EST():
    return datetime.datetime.now().astimezone(pytz.timezone('US/Eastern')).strftime("%H:%M, %m/%d")
def get_date():
    return EST()[7:9] + EST()[10:13]

def get_time():
    return EST()[:2] + EST()[3:5]


def get_MB(a):
    MB = sys.getsizeof(a) / 1024 / 1024
    return MB
#############################################################################
dash = "--------------------------------------"
DASH = "\n" + "--------------------------------------" + "\n"
Dash = "\n" + dash
dasH = dash + "\n"
#############################################################################
#%% utility funs

def meanvar_meanprex(mean, var):
    "mean-variance to precision"
    return 1/(mean*(1-mean)/var-1)

def beta_reparameterize(pi,phi_beta):
    """ mean-precision to standard """
    return pi / phi_beta, (1 - pi) / phi_beta
def alpha_beta_2_mean_var(alpha, beta):
    m = alpha / (alpha + beta)
    v = alpha * beta / (alpha + beta) ** 2 / (alpha + beta + 1)
    return m, v


def logistic(x):
    return np.exp(x) / (np.exp(x) + 1)
def logit(x):
    return np.log(x/(1-x))
#########################
def v_2_y(mat):
    temp = 2 / (1+mat) - 1
    return logit(temp)

def v_2_theta(v):
    return 1/(1+v)
def y_2_theta(y):
    return (logistic(y)+1)/2
def theta_2_v(theta):
    return 1/theta - 1

from multiprocessing import Pool
import multiprocessing as mp
n_cores = mp.cpu_count()

def mute():
    sys.stdout = open(os.devnull, 'w')    

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))
        
def parmap(f, X, nprocs = mp.cpu_count(), **args):#-2
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    
    def g(x):
        return f(x, **args)
    
    proc = [mp.Process(target=fun, args=(g, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

def setminus(A, B):
    return [item for item in A if item not in B]

def listinlist2list(theList):
    return [item for sublist in theList for item in sublist]

def if_exist(obj):
    return obj in locals() or obj in globals()

def getSize(one_object):
    print(one_object.memory_usage().sum() / 1024 ** 2, "MB")
#     print(sys.getsizeof(one_object) // 1024, "MB")

def dump(file, path):
    pickle.dump(file, open(path, "wb"))
    
def load(path):
    return pickle.load(open(path, "rb"))


def quantile(a, p):
    r = [a[0] for a in DF(a).quantile(p).values]
    return np.round(r, 3)

#############################################################################
#############################################################################
import inspect
import functools

def autoargs(*include, **kwargs):
    def _autoargs(func):
        attrs, varargs, varkw, defaults = inspect.getargspec(func)

        def sieve(attr):
            if kwargs and attr in kwargs['exclude']:
                return False
            if not include or attr in include:
                return True
            else:
                return False

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # handle default values
            if defaults:
                for attr, val in zip(reversed(attrs), reversed(defaults)):
                    if sieve(attr):
                        setattr(self, attr, val)
            # handle positional arguments
            positional_attrs = attrs[1:]
            for attr, val in zip(positional_attrs, args):
                if sieve(attr):
                    setattr(self, attr, val)
            # handle varargs
            if varargs:
                remaining_args = args[len(positional_attrs):]
                if sieve(varargs):
                    setattr(self, varargs, remaining_args)
            # handle varkw
            if kwargs:
                for attr, val in kwargs.items():
                    if sieve(attr):
                        setattr(self, attr, val)
            return func(self, *args, **kwargs)
        return wrapper
    return _autoargs
#############################################################################
#############################################################################
# pd.options.display.max_rows = 10

# with open('pred_columns.txt', 'w') as filehandle:
#     k = 0
#     for listitem in list(a):
#         filehandle.write('{}    {}\n'.format(k, listitem))
#         k += 1

def print_all(dat, column_only = True):
    if column_only:
        with pd.option_context('display.max_columns', None):  # more options can be specified also
            print(dat)
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(dat)

            
def quantile(a):
    return np.percentile(a, range(0,110,10))

#############################################################################

def unzip(path, zip_type = "tar_gz"):
    if zip_type == "tar_gz":
        import tarfile
        tar = tarfile.open(path, "r:gz")
        tar.extractall()
        tar.close()
    elif zip_type == "zip":        
        with ZipFile(path, 'r') as zipObj:
           # Extract all the contents of zip file in current directory
           zipObj.extractall()
def zip_folder(folder, zip_name):
    import zipfile

    zf = zipfile.ZipFile(zip_name, "w")
    for dirname, subdirs, files in os.walk(folder):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()

            
# import shutil

# total, used, free = shutil.disk_usage("/")

# print("Total: %d GiB" % (total // (2**30)))
# print("Used: %d GiB" % (used // (2**30)))
# print("Free: %d GiB" % (free // (2**30)))

#############################################################################

from termcolor import colored, cprint

# https://pypi.org/project/termcolor/#description
def printR(theStr):
    print(colored(theStr, 'red'))
          
def printG(theStr):
    print(colored(theStr, 'green'))
          
def printB(theStr):
    print(colored(theStr, 'blue'))

"""
text = colored('Hello, World!', 'green', attrs=['reverse', 'blink'])
print(text)
cprint('Hello, World!', 'green', 'on_red')

"""
    
def sets_intersection(d):
    return list(reduce(set.intersection, [set(item) for item in d ]))

def display_side_by_side(tables):
    html_str=''
    for df in tables:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

    
    
def freq_table(records, ordered = True, cumulative = False):
    cnt = Counter(records)
    if ordered:
        cnt = collections.OrderedDict(sorted(cnt.items()))
    cnts = arr(list(cnt.values()))
    cnts = cnts / np.sum(cnts)
    if cumulative:
        cnts = np.cumsum(cnts)
    keys = list(cnt.keys())
    table = DF(cnts.reshape(1, -1), columns = keys)
    display(table)
    