'''
Created on Aug 15, 2021

@authors: Xiaoshi Zhong (Primary), Muyin Wang, Hongkun Zhong
'''
import numpy as np
from scipy import stats
from random import random

def find_max_D(alpha = 2.5, size = 100000, group = 500, step = 100, is_discrete = True):
    ## generate samples
    sample_list = []
    if is_discrete == True:
        for i in range(group):
            sample = np.random.zipf(alpha, size)
            sample_list.append(sample)
    else:
        for i in range(group):
            sample = []
            while len(sample) < size:
                ### Generate a continuous power-law sample.
                ### This method is from the footnode 3 in Newman (2005):
                ### Mark E.J. Newman. Power laws, Pareto distributions, and Zipf's law. Contemporary physics, 46(5): 323-351.
                ### This method is also used in the randht module described in https://aaronclauset.github.io/powerlaws
                x = 1.0 * pow(1.0 - random(), - 1.0/(alpha - 1.0))
                sample.append(x)
            sample_list.append(sample)
    
    ## use ks_2samp to compute the KS D and p-values among those samples
    D = np.empty((group, group))
    pvalue = np.empty((group, group))
    for i in range(group):
        for j in range(i + 1, group):
            D_tem, pvalue_tem = stats.ks_2samp(sample_list[i], sample_list[j])
            D[i][j] = D_tem
            pvalue[i][j] = pvalue_tem

    list = []
    for s in range(step, group + 1, step):
        D_max = 0.0
        pvalue_max = 1.0
        for i in range(s):
            for j in range(i + 1, s):
                if D[i][j] > D_max:
                    D_max = D[i][j]
                    pvalue_max = pvalue[i][j]
        list.append([s, D_max, pvalue_max])
    return list

if __name__ == "__main__":
    alpha = 2.5
    size = 10000
    group = 500
    step = 100
    is_discrete = False
    
    list = find_max_D(alpha, size, group, step, is_discrete)
    filename = "max_D_" + str(alpha) + "_" + str(size) + "_" + str(is_discrete).lower() + ".txt"
    f_out = open(filename, "w")
    for item in list:
        print(item)
        f_out.write(str(item[0]) + "\t" + str(item[1]) + "\t" + str(item[2]) + "\n")
    f_out.close()
    