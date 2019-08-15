#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gplearn2
from gplearn2 import *
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import rankdata
pd.set_option("use_inf_as_na", True)


# In[2]:


cls = np.random.rand(100,10)
opn = np.random.rand(100,10)
volume = np.random.rand(100,10)
amount = np.random.rand(100,10)
mtm = np.random.rand(100,10)
y = np.random.rand(100,10)


# In[3]:


y[2] = [np.nan] * 10


# In[4]:


Train = pd.DataFrame()
Train["cls"] = np.concatenate(cls)
Train["opn"] = np.concatenate(opn)
Train["volume"] = np.concatenate(volume)
Train["amount"] = np.concatenate(amount)
Train.iloc[0] = np.nan
Train["returns"] = np.concatenate(y)


# In[5]:


Train = Train.dropna(axis = 0, subset = ['returns'])


# In[6]:


Train = Train.dropna(axis = 0, subset = ["cls","opn","volume","amount"], how = "all")


# In[7]:


Train.iloc[0,0], Train.iloc[1,1] = np.nan, np.nan


# In[8]:


fmt = np.array([True if _ in Train.index else False for _ in range(cls.shape[0] * cls.shape[1])])
mask = [fmt, len(cls)]


# In[9]:


def _minp(d):
    if not isinstance(d, int):
        d = int(d)
    if d <= 10:
        return d - 1
    else:
        return d * 2 // 3


# In[10]:


def _rank(x_input, mask):

    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
        
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    res = x.rank(axis=1).sub(0.5).div(x.count(axis=1), axis=0).values.reshape(sp)
    return res[np.where(mask[0])]
rank = gplearn2.functions.make_function(function = _rank, name = "rank", arity = 1, wrap = True)


# In[11]:


def _delay(x_input, d, mask):
    
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res = x.shift(d).values.reshape(sp)
    
    return res[np.where(mask[0])]

delay = gplearn2.functions.make_function(function = _delay, name = "delay",arity = 2, wrap = True)


# In[12]:


def _correlation(x_input, y_input, d, mask):
    
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    y = np.array([np.nan] * len(mask[0]))
    y[np.where(mask[0])] = y_input
    
    x = x.reshape(mask[1],-1)
    y = y.reshape(mask[1],-1)
    
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    
    res = (x.rolling(window=int(d), min_periods=_minp(d)).corr(y)).values.reshape(sp)
    return res[np.where(mask[0])]
    
correlation = gplearn2.functions.make_function(function = _correlation, name = "correlation", arity = 3, wrap = True)


# In[13]:


def _covariance(x_input, y_input, d, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    y = np.array([np.nan] * len(mask[0]))
    y[np.where(mask[0])] = y_input
    
    x = x.reshape(mask[1],-1)
    y = y.reshape(mask[1],-1)
    
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    res = (x.rolling(window=int(d), min_periods=_minp(d)).cov(y)).values.reshape(sp)
    return res[np.where(mask[0])]
    
covariance = gplearn2.functions.make_function(function = _covariance, name = "covariance",arity = 3, wrap = True)    


# In[14]:


def _scale(x_input, a, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res = x.mul(a).div(x.abs().sum(axis = 1), axis = 0).values.reshape(sp)
    return  res[np.where(mask[0])]

scale = gplearn2.functions.make_function(function= _scale, name = "scale", arity = 2, wrap = True)


# In[15]:


def _delta(x_input, d, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res = x.diff(int(d)).values.reshape(sp)
    return res[np.where(mask[0])]

delta = gplearn2.functions.make_function(function = _delta, name = "delta", arity = 2, wrap = True)


# In[16]:


def _signedpower(x_input, a, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res = (np.sign(x) * x.abs().pow(a)).values.reshape(sp)
    return res[np.where(mask[0])]

signedpower = gplearn2.functions.make_function(function= _signedpower, name = "signedpower", arity=2, wrap=True)


# In[17]:


def _decay_linear(x_input, d, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    weight = np.arange(0, int(d)) + 1
    res = (x.rolling(window = int(d), min_periods = _minp(d)).apply(lambda z: np.nansum(z * weight[-len(z):]) / weight[-len(z):][~np.isnan(z)].sum(),raw = True)).values.reshape(sp)
    return res[np.where(mask[0])]

decay_linear = gplearn2.functions.make_function(function= _decay_linear, name = "decay_linear", arity = 2, wrap = True)


# In[18]:


def _ts_min(x_input, d, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res = (x.rolling(window = int(d), min_periods = _minp(d), axis = 0).min()).values.reshape(sp)
    return res[np.where(mask[0])]

ts_min = gplearn2.functions.make_function(function = _ts_min, name = "ts_min", arity = 2, wrap = True)


# In[19]:


def _ts_max(x_input, d, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res = (x.rolling(window = int(d), min_periods = _minp(d), axis = 0).max()).values.reshape(sp)
    return res[np.where(mask[0])]

ts_max = gplearn2.functions.make_function(function = _ts_max, name = "ts_max", arity = 2, wrap = True)


# In[20]:


def _ts_argmin(x_input, d, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res =  (x.rolling(window = int(d), min_periods = _minp(d), axis = 0).apply(lambda z: np.nan if np.all(np.isnan(z)) else np.nanargmin(z), raw = True) + 1).values.reshape(sp)
    return res[np.where(mask[0])]
    
ts_argmin = gplearn2.functions.make_function(function = _ts_argmin, name = "ts_argmin", arity = 2, wrap = True)


# In[21]:


def _ts_argmax(x_input, d, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res = (x.rolling(window = int(d), min_periods = _minp(d), axis = 0).apply(lambda z: np.nan if np.all(np.isnan(z)) else np.nanargmax(z), raw = True) + 1).values.reshape(sp)
    return res[np.where(mask[0])]
    
ts_argmax = gplearn2.functions.make_function(function = _ts_argmax, name = "ts_argmax", arity = 2, wrap = True)


# In[26]:


def _ts_rank(x_input, d, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res = (x.rolling(int(d), min_periods = _minp(d)).apply(
    lambda z: np.nan if np.all(np.isnan(z)) else ((rankdata(z[~np.isnan(z)])[-1] - 1) * (len(z) - 1) / (len(z[~np.isnan(z)]) - 1) + 1), raw = True)).values.reshape(sp)
    return res[np.where(mask[0])]

ts_rank = gplearn2.functions.make_function(function = _ts_rank, name = "ts_rank", arity = 2, wrap = True)


# In[31]:


def _ts_sum(x_input, d, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res = (x.rolling(window = int(d), min_periods = _minp(d)).sum()).values.reshape(sp)
    return res[np.where(mask[0])]

ts_sum = gplearn2.functions.make_function(function = _ts_sum, name = "ts_sum", arity = 2, wrap = True)


# In[36]:


def _ts_product(x_input, d, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res =  (np.log(np.exp(x).rolling(window = int(d), min_periods = _minp(d), axis = 0).mean() * int(d))).values.reshape(sp)
    return res[np.where(mask[0])]

ts_product = gplearn2.functions.make_function(function = _ts_product, name = "ts_product", arity = 2, wrap = True)


# In[41]:


def _ts_stddev(x_input, d, mask):
    sp = mask[0].shape
    x = np.array([np.nan] * len(mask[0]))
    x[np.where(mask[0])] = x_input
    x = x.reshape(mask[1],-1)
    x = pd.DataFrame(x)
    
    res =  (x.rolling(window = int(d), min_periods = _minp(d), axis = 0).std()).values.reshape(sp)
    return res[np.where(mask[0])]

ts_stddev = gplearn2.functions.make_function(function = _ts_stddev, name = "ts_stddev", arity = 2, wrap = True)


# In[42]:


function_set_test = ["add", "sub", "mul", "div","abs", "sqrt","log", "inv", rank, delay, correlation, covariance, scale, delta, signedpower, 
                     decay_linear, ts_min, ts_max, ts_argmin, ts_argmax, ts_rank, ts_sum, ts_product, ts_stddev]
reshape_function_set_test = [rank, delay, correlation, covariance, scale, delta, signedpower, decay_linear, ts_min, ts_max, ts_argmin,
                            ts_argmax, ts_rank, ts_sum, ts_product, ts_stddev]
feature_function_set_test = [delay, correlation, covariance, scale, delta, signedpower, decay_linear, ts_min, ts_max, ts_argmin,
                            ts_argmax, ts_rank, ts_sum, ts_product, ts_stddev]


# In[45]:


gp = gplearn2.genetic.SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=20,
                         function_set=function_set_test,
                         reshape_function_set = reshape_function_set_test,
                         feature_function_set = feature_function_set_test,
                                          mask = mask,
                         parsimony_coefficient=0.0005, verbose = 1)


# In[46]:


gp.fit(Train.loc[:, ["cls","opn","volume","amount"]], Train.loc[: , ["returns"]])


# In[47]:


print(gp)

