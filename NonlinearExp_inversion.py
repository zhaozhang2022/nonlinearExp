
import sys
import torch
import torchvision
import torch.utils.data
import math
import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import cg
import random
import copy
import time
import os
from itertools import combinations
from scipy.stats import multivariate_normal as mvnorm
import matplotlib.pyplot as plt
from scipy import stats

qwt_real=np.loadtxt('realqwt_egg_noise_imp.txt')
q_prior=np.loadtxt('priorqwts.txt')

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def loglikelihoodprob(qwt_obs, qwt_sim):
    pp=0.0
    if len(qwt_obs)!=len(qwt_sim):
        print('observed and simulated sequences of different lengths')
        sys.exit(0)
    for i in range(len(qwt_obs)):
        p1=qwt_obs[i]
        p2=qwt_sim[i]
        sigma_=obsigma_
        z=(p2-p1)/sigma_
        f=1.0/math.sqrt(2*math.pi)*math.exp(-z**2/2)
        if f<1e-100:
            f=1e-100
        pp=pp+math.log(f)
    return pp


obsigma_=2e-4
nsamples=len(q_prior)
nt=len(q_prior[0])
likevec=np.zeros(nsamples)

starttime=time.time()

print("create groups and calculate expectations")
size=4
groups=np.array([c for c in combinations(range(nsamples), size)])
ngroups=len(groups)
q_exp=np.zeros((ngroups, nt))
like_exp=np.zeros(ngroups)
for ig in range(ngroups):
    qq=np.zeros(nt)
    print("evaluate group: ", ig, ' of ', ngroups)
    for i in range(size):
        isample=groups[ig, i]
        qq=qq+q_prior[isample]
    qq=qq/size
    q_exp[ig]=qq
    like_exp[ig] = loglikelihoodprob(qwt_real, qq)
endtime1=time.time()
npostgroups=nsamples
post_groups=groups[:npostgroups]
post_like=like_exp[:npostgroups]
likemin=min(post_like)
likelist=list(post_like)
likemin_id=likelist.index(likemin)
for ig in range(npostgroups,ngroups):
    like_=like_exp[ig]
    if like_>likemin:
        post_like[likemin_id]=like_
        post_groups[likemin_id]=groups[ig]
        likemin = min(post_like)
        likelist = list(post_like)
        likemin_id = likelist.index(likemin)
sorted_ind = np.argsort(post_like)[::-1]
post_like=post_like[sorted_ind]
post_groups=post_groups[sorted_ind]


# sorted_ind = np.argsort(like_exp)[::-1]
# sorted_like=like_exp[sorted_ind]
# sorted_q_exp=q_exp[sorted_ind]
# sorted_groups =groups[sorted_ind]
# npostgroups=nsamples
# post_groups=sorted_groups[:npostgroups]
# post_q_exp=sorted_q_exp[:npostgroups]
endtime=time.time()
print('time cost: ', endtime-starttime)
f = open('TimeCost-grouping-size4.txt','w')
f.write("%e \n" % (endtime-starttime))
f.write("timecost before sorting: %e \n" % (endtime1-starttime))
f.close()
f = open('loglikevec.txt','w')
for i in range(npostgroups):
    f.write("%e \n" % (-1*post_like[i]))
f.close()
# f = open('inversedqwt_nonlinear.txt','w')
# for i in range(npostgroups):
#     for j in range(nt):
#         f.write("%e " % post_q_exp[i,j])
#     f.write("\n")
# f.close()
markvec=np.zeros(nsamples)
for i in range(npostgroups):
    for j in range(size):
        markvec[post_groups[i,j]]=1
postsamples=[]
for i in range(nsamples):
    if markvec[i]==1:
        postsamples.append(i)
npostsamples=len(postsamples)
f = open('postsamples.txt','w')
for i in range(npostsamples):
    f.write("%d " % postsamples[i])
f.close()
f = open('postgroups.txt','w')
for i in range(npostgroups):
    for j in range(size):
        f.write("%d " % post_groups[i,j])
    f.write("\n")









