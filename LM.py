from pylab import *
from numpy import *
from math import *

def Jacobian(a_est, b_est, data_1, Ndata, Nparams):
    J=zeros((Ndata,Nparams),float)
    for i in range(Ndata):
        J[i][0] = exp(-(b_est*data_1[i][0]))
        J[i][1] = -a_est*data_1[i][0]*exp(-(b_est*data_1[i][0]))
    return J
    
def NormG(g):
    absg = absolute(g)
    Normg = absg.argmax()
    num = absg[Normg]
    return num
    
def LM(obs_1, data_1, params, Ndata, Nparams, maxIter):
    lamda = 0.01
    v = 2
    nIter = 0
    updateJ = 0
    Threshold = 1e-5
    a_est = params[0][0]
    b_est = params[1][0]
    y_est = zeros((Ndata,1),float)
    y_est_lm = zeros((Ndata,1),float)
    while nIter < maxIter:
        if updateJ == 0:
            J = Jacobian( a_est, b_est, data_1, Ndata, Nparams )
            for i in range(Ndata):
                y_est[i][0] = a_est*exp(-b_est*data_1[i][0])
            d = obs_1-y_est
            H = dot(J.T, J)
            if nIter == 0:
                e = dot(d.T, d)
        H_lm = H + (lamda * np.eye(Nparams))
        dp = solve(H_lm, dot(J.T, d))
        a_lm = a_est+dp[0][0]
        b_lm = b_est+dp[1][0]
        for i in range(Ndata):
            y_est_lm[i][0] = a_lm*exp(-b_lm*data_1[i][0])
        d_lm = obs_1-y_est_lm
        e_lm = dot(d_lm.T, d_lm)
        if e_lm < e:
            lamda = lamda/v
            a_est = a_lm
            b_est = b_lm
            e = e_lm
            updateJ = 1;
        else:
            updateJ = 0
            lamda = lamda*v
        nIter += 1
            
    if nIter == maxIter:
        print "The Error:"
        print e
        print "The Data"
        print y_est_lm.T
        print "The Params:"
        print a_est, b_est


# f(x)=a*exp(-b*x);    
# Data
Ndata = 9
Nparams = 2
data_1 = np.array([[0.25], [0.5], [1], [1.5], [2], [3], [4], [6], [8]])
obs_1 = np.array([[19.21], [18.15], [15.36], [14.10], [12.89], [9.32], [7.45], [5.24], [3.01]])
# Parameters
params = np.array([[10],[0.5]])
# Number_of_Interation
maxIter = 200
    
# LM
LM(obs_1, data_1, params, Ndata, Nparams, maxIter)
    
    
