'''
Created on August 20, 2021

@author: Xiaoshi Zhong (Primary), Muyin Wang, and Hongkun Zhang
'''
import math
import numpy as np
from scipy import stats
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from random import random

### Generate a continuous power-law sample.
### This method is from the footnode 3 in Newman (2005):
### Mark E.J. Newman. Power laws, Pareto distributions, and Zipf's law. Contemporary physics, 46(5): 323-351.
### This method is also used in the randht module described in https://aaronclauset.github.io/powerlaws
def generate_continuous_data(alpha = 2.5, size = 100000, xmin = 1.0):
    sample = []
    while len(sample) < size:
        x = xmin * pow(1.0 - random(), - 1.0/(alpha - 1.0))
        sample.append(x)
    return sample

def log2binning(x):
    x_max = max(x)
    x_min = min(x)
    
    log2width = 1
    
    bins = 2 ** (np.arange(int(np.log2(x_min)), math.ceil(np.log2(x_max)) + log2width, log2width))
    
    counts, edges = np.histogram(x, bins=bins)
    
    centers = list(map(lambda X, Y: np.sqrt(X * Y), edges[:-1], edges[1:]))
    
    probs = counts / np.sum(counts)
    
    widths = bins[1:] - bins[:-1]
    probs = probs / widths
    
    return zip(*((center, count, prob) for center, count, prob, in zip(centers, counts, probs) if count > 0))
    
    #return centers, counts, probs

### input two vectors, in which x is the key vector and y is the percentage vector of x.
def powerlaw_fit(x, y):
    length = len(x)
    
    B = np.ones([length, 1])
    
    X = np.column_stack([np.log(x), B])
    Y = np.column_stack([np.log(y)])
    
#     xlog = np.column_stack([np.log(x)])
#     ylog = np.column_stack([np.log(y)])
    
    w = lstsq(X, Y)
    yhat = np.dot(X, w[0])
    
    # K = np.round(np.exp(w[0][1][0]), 2)
    # alpha = np.round(w[0][0][0], 2)
    
    return w, yhat

def filter_zeros(centers, counts, probs):
    _centers, _counts, _probs = [], [], []
    for i in range(len(counts)):
        if counts[i] >= 1:
            _centers.append(centers[i])
            _counts.append(counts[i])
            _probs.append(probs[i])
    
    return _centers, _counts, _probs

#############################################
def calculate_K(key, prob, alpha):
    key_log = np.log(key)
    prob_log = np.log(prob)
    key_log_mean = np.mean(key_log)
    prob_log_mean = np.mean(prob_log)
    
    const = prob_log_mean - alpha * key_log_mean
    K = np.exp(const)
    
    return K

def calculate_alpha_mean(alpha_values):
    return np.mean(alpha_values)

def calculate_pdf_hat(key, K, alpha):
    return list(map(lambda X: K * X ** alpha, key))

def calculate_cdf(pdf):#cdf_freq_x=cdf of f(x)
    cdf = []
    for i in range(len(pdf)):
        cdf.append(sum(pdf[ : i + 1]))
    return cdf

def calculate_ks_D(cdf_data, cdf_model):# the KS statistic D(x)
    ks_d = abs(np.array(cdf_data) - np.array(cdf_model))
    return max(ks_d)

##################################################

alpha = 2.5
xmin = 1.0
## sample size (not data points)
n = 1000000

x = generate_continuous_data(alpha, n, xmin)

centers, counts, probs = log2binning(x)

w, probs_norm_hat = powerlaw_fit(centers, probs)

K_norm = np.round(np.exp(w[0][1][0]), 4)
alpha_norm = np.round(w[0][0][0], 4)
print("alpha_norm:", alpha_norm, "K_norm:", K_norm)
print()

#######################################################################
# Calculate alpha_values
alpha_values = []
for i in range(1, len(probs)): #X1 from 2 to key_X1
    centers_tem = centers[: i + 1]
    probs_tem = probs[: i + 1]
    
    #get alpha
    w_tem, prob_tem_hat = powerlaw_fit(centers_tem, probs_tem)
    
    alpha_tem = w_tem[0][0][0]
    alpha_values.append(alpha_tem)

#######################################################################

alpha_avg = np.round(calculate_alpha_mean(alpha_values), 4)
K_avg = calculate_K(centers, probs, alpha_avg)

probs_avg_hat = calculate_pdf_hat(centers, K_avg, alpha_avg)
print("alpha_avg:", alpha_avg, "K_avg:", np.round(K_avg, 4))


data = r'Data ($n=10^' + str(int(np.log10(n))) + '$)'
log2bin_data = r'Data (log2bins)'
ls_norm_label = r'LS$_{norm}$ ($\hat{\alpha}=' + str(-alpha_norm) + '$)'
ls_avg_label = r'LS$_{avg}$ ($\hat\alpha=' + str(-alpha_avg) + '$)'

plt.figure(figsize=(5.5, 4.5))
plt.plot(centers, probs, "o", label = data)
plt.plot(centers, np.exp(probs_norm_hat), "g-", label=ls_norm_label)
plt.plot(centers, probs_avg_hat, "r+-", label=ls_avg_label)


plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left', frameon=False, numpoints=1)

plt.show()

