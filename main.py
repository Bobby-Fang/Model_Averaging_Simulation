import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load birth weight data set - txt data format
rawData = [] # initizalize empty array
with open('lowbwt.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        tmp = line[2:].strip('\n').split() # each entry of tmp is string type
        rawData.append([int(x) for x in tmp]) # 189x11 array

# Columns Variable Abbreviation 
# ------------------------------------------------------------------------- 
# 2-4 Identification Code ID
# 10 Low Birth Weight (0 = Birth Weight ge 2500g, l = Birth Weight < 2500g) LBW
# 17-18 Age of the Mother in Years AGE
# 23-25 Weight in Pounds at the Last Menstrual Period LWT
# 32 Race (1 = White, 2 = Black, 3 = Other) RACE
# 40 Smoking Status During Pregnancy (1 = Yes, 0 = No) SMOKE
# 48 History of Premature Labor (0 = None, 1 = One, etc.) PTL
# 55 History of Hypertension (1 = Yes, 0 = No) HYPER
# 61 Presence of Uterine Irritability (1 = Yes, 0 = No) URIRR
# 67 Number of Physician Visits During the First Trimester PVFT
#           (0 = None, 1 = One, 2 = Two, etc.) 
# 73-76 Birth Weight in Grams BWT 
# -------------------------------------------------------------------------

rawData = np.array(rawData)
data = rawData[:,2:] # drop first two columns, irrelevant to the model

# potential regressors
index_set = [[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0],[6,0,0],[7,0,0],[8,0,0],
[1,2,0],[1,3,0],[1,4,0],[1,5,0],[1,6,0],[1,7,0],[1,8,0],[2,3,0],[2,4,0],[2,5,0],
[2,6,0],[2,7,0],[4,7,0],[4,8,0],[5,6,0],[5,7,0],[5,8,0],[6,7,0],[6,8,0],[7,8,0],
[1,2,3],[1,2,4],[1,2,5],[1,2,6],[1,2,7],[1,2,8],[1,3,4],[1,3,5],[1,3,6],[1,3,7],
[1,3,8],[1,4,5],[1,4,6],[1,4,7],[1,4,8],[1,5,6],[1,5,7],[1,5,8],[1,6,7],[1,6,8],
[1,7,8],[2,3,4],[2,3,5],[2,3,6],[2,3,7],[2,3,8],[2,4,5],[2,4,6],[2,4,7],[2,4,8],
[2,5,6],[2,5,7],[2,5,8],[2,6,7],[2,6,8],[2,7,8],[3,4,5],[3,4,6],[3,4,7],[3,4,8],
[3,5,6],[3,5,7],[3,5,8],[3,6,7],[3,6,8],[3,7,8],[4,5,6],[4,5,7],[4,5,8],[4,6,7],
[4,6,8],[4,7,8],[5,6,7],[5,6,8],[5,7,8],[6,7,8],[1,1,0],[1,1,1],[2,2,0],[2,2,2]]

"""
	1) interactions of x_ix_jx_k based on choices of i, j, and k
	2) standardized the list such that ||x||^2/n = 1
"""
def read_index(dt,idx):
	init = dt[:,idx[0]-1]
	for i in idx[1:]:
		if i != 0:
			init *= dt[:,i-1]
	s = np.sum(np.dot(init,init))
	if s != 0:
		new_init = [np.sqrt(189)*x/np.sqrt(s) for x in init]
		return np.array(new_init)
	else:
		return np.array(init)

# all regressors in a 189x88 numpy array
regrs = []
for x in index_set:
	regrs.append(read_index(data, x))
regrs = np.array(regrs).T

# useful predictor variable
idx = []
for i in range(88):
	if np.sum(regrs[:,i]) != 0:
		idx.append(i)
x = regrs[:,idx]

# reponse variable
y = data[:,-1]

# estimate the correlation between predictor variable and the response variable
corr = [np.dot(x[:,i],y)/n for i in range(19)]
itr = np.argsort(corr) # sort correlations in ascending order

