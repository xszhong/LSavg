'''
Created on Apr 26, 2021

@authors: Xiaoshi Zhong (Primary), Muyin Wang, Hongkun Zhong
'''
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

x_min = min(x)
x_max = max(x)
width = 1

bins = np.arange(x_min, x_max + width, width)
counts, edges = np.histogram(x, bins=bins)

widths = bins[1:] - bins[:-1]
counts = counts / widths    # normalize

# centers  = edges[:-1] # use the minimal values
# centers = edges[1:] # use the maximal values
# centers = (edges[1:] + edges[:-1])/2.0    # set the center as the arithmetic average of the two end points
# centers = list(map(lambda X, Y: 10 ** ((np.log10(X) + np.log10(Y))/2.0), edges[1:], edges[:-1])) # set the center as the average of the logarithmic of the two end points
centers = list(map(lambda X, Y: np.sqrt(X * Y), edges[:-1], edges[1:]))

count_sum = sum(counts)

key_all = []
count_all = []
prob_all = []

flag_X1 = False
index_X1 = -1

index_Xf = -1
flag_Xf = False

for i in range(len(counts)):
    key = centers[i]
    count = counts[i]
    prob = count / count_sum
    
    if count <= 0.0:
        continue
    
    key_all.append(key)
    count_all.append(count)
    prob_all.append(prob)
    
    if flag_X1 == False:
        index_X1 = i
    
#     if count <= 1:
    if prob <= 1/n:
        flag_X1 = True
        
    if flag_Xf == False and i == len(counts) - 1:
        index_Xf = i
        flag_Xf = True
    
    if flag_Xf == False and i < len(counts) - 2 and count > max(counts[i + 1: ]):
        index_Xf = i
    else:
        flag_Xf = True
        
print("index_X1:", index_X1)
print("index_Xf:", index_Xf)

w_all, prob_all_hat = powerlaw_fit(key_all, prob_all)

K_all = np.round(np.exp(w_all[0][1][0]), 4)
alpha_all = np.round(w_all[0][0][0], 4)

print("key_all_max:", max(key_all))
print("K_all: ", K_all)
print("alpha_all: ", alpha_all)
print()

################################################
# Calculate alpha_values
alpha_values = []
for i in list(range(2, index_X1 + 2)): #X1 from 2 to key_X1
    key_tem = key_all[: i]
    prob_tem = prob_all[: i]
    cdf_data_tem = calculate_cdf(prob_tem)
    
    #get alpha
    w_tem, prob_tem_hat = powerlaw_fit(key_tem, prob_tem)
    
    alpha_tem = w_tem[0][0][0]
    alpha_values.append(alpha_tem)

#######################################################################
# for X1
key_X1 = key_all[: index_X1 + 1]
count_X1 = count_all[: index_X1 + 1]
prob_X1 = prob_all[: index_X1 + 1]

alpha_X1_values = alpha_values[:]
alpha_X1_mean = calculate_alpha_mean(alpha_X1_values)
K_X1 = calculate_K(key_X1, prob_X1, alpha_X1_mean)

prob_X1_hat = calculate_pdf_hat(key_X1, K_X1, alpha_X1_mean)
print("alpha_X1_mean:", alpha_X1_mean, "K_X1:", K_X1)
# print("prob_X1_hat:", prob_X1_hat)

x1min_X1 = (K_X1 * n) ** (1.0/np.abs(alpha_X1_mean))

cdf_X1 = calculate_cdf(prob_X1)
cdf_X1_hat = calculate_cdf(prob_X1_hat)

D_X1 = calculate_ks_D(cdf_X1, cdf_X1_hat)
print("D_X1:", D_X1)
print()

#######################################################################
# for X5th
index_X5th = 4
key_X5th = key_all[: index_X5th + 1]
count_X5th = count_all[: index_X5th + 1]
prob_X5th = prob_all[: index_X5th + 1]

alpha_X5th_values = alpha_values[:index_X5th]
alpha_X5th_mean = calculate_alpha_mean(alpha_X5th_values)
K_X5th = calculate_K(key_X5th, prob_X5th, alpha_X5th_mean)

prob_X5th_hat = calculate_pdf_hat(key_X1, K_X5th, alpha_X5th_mean)
print("alpha_X5th_mean:", alpha_X5th_mean, "K_X5th:", K_X5th)
# print("prob_X1_hat:", prob_X1_hat)

x1min_X5th = (K_X5th * n) ** (1.0/np.abs(alpha_X5th_mean))

cdf_X5th = calculate_cdf(prob_X1)
cdf_X5th_hat = calculate_cdf(prob_X5th_hat)

D_X5th = calculate_ks_D(cdf_X5th, cdf_X5th_hat)
print("D_X5th:", D_X5th)
print()

########################################################################
### For index_Xf
key_Xf = key_all[ : index_Xf + 1]
count_Xf = count_all[ : index_Xf + 1]
prob_Xf = prob_all[ : index_Xf + 1]

alpha_Xf_values = alpha_values[ : index_Xf]
alpha_Xf_mean = calculate_alpha_mean(alpha_Xf_values)
K_Xf = calculate_K(key_Xf, prob_Xf, alpha_Xf_mean)

prob_Xf_hat = calculate_pdf_hat(key_X1, K_Xf, alpha_Xf_mean)
print("alpha_Xf_mean:", alpha_Xf_mean, "K_Xf:", K_Xf)
# print("prob_Xf_hat:", prob_Xf_hat)

x1min_Xf = (K_Xf * n) ** (1.0/np.abs(alpha_Xf_mean))

cdf_Xf = calculate_cdf(prob_X1)
cdf_Xf_hat = calculate_cdf(prob_Xf_hat)

D_Xf = calculate_ks_D(cdf_Xf, cdf_Xf_hat)
print("D_Xf:", D_Xf)
print()

###################################################################################

parameters = r'Data ($n=10^' + str(int(np.log10(n))) + '$)'
ls_all_label = r'LS$_{all}$ ($\hat{\alpha}=' + str(-np.round(alpha_all, 4)) + '$)'
ls_X1_label = r'LS$_{X1}$ ($\hat{\alpha}=' + str(-np.round(alpha_X1_mean, 4)) + '$, $X_1^T=' + str(np.round(x1min_X1, 1)) + '$)'
ls_Xf_label = r'LS$_{Xf}$ ($\hat{\alpha}=' + str(-np.round(alpha_Xf_mean, 4)) + '$, $X_1^T=' + str(np.round(x1min_Xf, 1)) + '$)'
ls_X5th_label = r'LS$_{X5th}$ ($\hat{\alpha}=' + str(-np.round(alpha_X5th_mean, 4)) + '$, $X_1^T=' + str(np.round(x1min_X5th, 1)) + '$)'

plt.figure(figsize=(5.5, 4.5))
plt.plot(key_all, prob_all, "o", label=parameters)
plt.plot(key_all, np.exp(prob_all_hat), "k-", label=ls_all_label)
plt.plot(key_X1, prob_X1_hat, "g*-", label=ls_X1_label)
plt.plot(x1min_X1, 1.0/n, "g*")
plt.plot(key_X1, prob_Xf_hat, "r+-", label=ls_Xf_label)
plt.plot(x1min_Xf, 1.0/n, "r+")
plt.plot(key_X1, prob_X5th_hat, "y-", label=ls_X5th_label)
plt.plot(x1min_X5th, 1.0/n, "y+")

plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left', frameon=False, numpoints=1)
plt.savefig("continuous-powerlaw-1e" + str(int(np.log10(n))) + ".pdf")

plt.show()