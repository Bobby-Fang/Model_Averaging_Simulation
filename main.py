import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


"""
design a
"""

mse = []
for t in range(1):
	n = 50 # sample size
	p = 2000 # number of regressors

	x_mean = np.zeros(p) # mean of regressors

	# covariance matrix for regressors
	rho = 0.6
	x_cov = np.zeros((p,p))
	for i in range(p):
		for j in range(p):
			x_cov[i][j] = np.power(rho, np.abs(i-j))

	# regressors
	x = np.random.multivariate_normal(x_mean, x_cov, size=n)

	# true coefficients
	beta = np.random.normal(0,0.5,size=p)

	# true regressors index
	idx = [40*(j-1)+1 for j in range(1,51)]

	# normal noise
	eps = np.random.normal(0,0.2,size=n)

	# response variable
	y = np.zeros(n)
	for i in range(n):
		y[i] = np.dot(beta,x[i])+eps[i]

	# estimate correlation between each regressor and response
	corr = np.dot(x.T,y)/n
	# sort correlations in ascending order by index
	itr = np.argsort(corr)[::-1]

	M = 10 # number of models
	nh = 10 # number of regressors per model

	# models
	X = np.array([x[:,itr[10*k:10*(k+1)]] for k in range(M)])
	H = [np.matmul(np.matmul(X[k],np.linalg.inv(np.matmul(X[k].T,X[k]))),X[k].T) for k in range(M)]

	# CV
	D = [np.diag(1/(1-np.diag(H[i]))) for i in range(M)]
	H_tilda = [np.matmul(D[i],(H[i]-np.eye(n)))+np.eye(n) for i in range(M)]
	# mu_tilda = np.dot(np.sum(np.dot(wt[i],H_tilda[i])),rsp)
	# cv = np.dot((y-mu_tilda).T,(y-mu_tilda))

	fun = lambda w: np.matmul((y-np.matmul(np.sum(np.dot(w[10*k:10*(k+1)],H_tilda[i])),rsp)).T,(y-np.dot(np.sum(np.dot(w[10*k:10*(k+1)],H_tilda[i])),rsp)))
	bnds = [[0,1] for i in range(M*nh)]
	wt = optimize.minimize(fun,np.ones(M*nh),bounds=bnds)

	pred = np.matmul(wt,X.reshape((50,100)))
	mse.append(np.mean((pred-y)**2))


# plt.boxplot(mse)
# plt.show()


