
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats, optimize

def argmin(f,x0,tol=1.0e-8,maxiter=100,args=(),fprime=None,fprime2=None):
    return sp.optimize.newton(f,x0,fprime,args,tol,maxiter,fprime2)

#find x where f(x) = y
def invert(f,y,x0,tol=1.0e-8):
    return argmin(f=lambda x: (f(x)-y)**2, x0=x0, tol=tol)

def chi2inv(p,dof,tol=1.0e-8):
    f = lambda x: sp.stats.chi2.cdf(x,df=dof)
    x0 = 3.0
    return invert(f,p,x0,tol)

# Find x_p in X such that P(X<x_p) ~= p
def inv_cdf(X,p):
    assert p>=0.0 and p<=1.0
    X = np.sort(np.ravel(X))
    return X[int(np.round(p*(X.size-1)))]


def gen_AR1(alpha,N,M=1,m=0.0,sigma_e=1.0):
    
    def format_input(var):
        if np.isscalar(var):
            var = np.asarray([var])
        return var

    m       = format_input(m)
    alpha   = format_input(alpha)
    sigma_e = format_input(sigma_e)
    M = np.max([M,alpha.size, m.size, sigma_e.size])

    assert m.ndim == 1 and alpha.ndim ==1 and sigma_e.ndim==1
    assert (alpha>=0.0).all() and (alpha<=1.0).all()
    assert (sigma_e>=0.0).all()
    
    shape = (M,N)
    En = np.random.randn(*shape)*sigma_e[:,None]
    AR1 = np.zeros(shape, dtype=float)

    AR1[:,0] = m[:] + En[:,0]
    for k in xrange(1,N):
        AR1[:,k] = m[:] + alpha[:]*AR1[:,k-1] + En[:,k]

    return AR1

def estimate_AR1_parameters(X):
    
    assert X.ndim in [1,2], 'Wrong shape in input!'

    if X.ndim==1:
        X = X[np.newaxis,:]
    M = X.shape[0]
    N = X.shape[1]
    
    # subtract mean
    X = X - np.mean(X,axis=1)[:,None]

    # AR0 and AR1 covariances
    C0 = np.sum(X*X                , axis=1) / N                  
    C1 = np.sum(X[:,0:N-1]*X[:,1:N], axis=1) / (N-1) 
     
    #A. Grinsteds
    A = C0* N**2
    B = (C0-C1)*(N-2) - (C0+C1)*N**2
    C = (C0-C1)*N + C1*N**2
    delta = B**2 - 4*A*C

    if (delta > 0.0).all():
        a = (-B-delta**0.5)/(2.0*A)
        alpha=a
    else:
        raise Exception('Serie is too short, bad delta!')
    
    mu2 = -1.0/N + (2.0/N**2)*((N-a**N)/(1.0-a) - a*(1.0-a**(N-1)) / (1.0-a)**2)
    C0e = C0 / (1.0 - mu2)
    sigma_e = np.sqrt((1.0-a**2)*C0e)
    
    # remove additional dim if just one AR1 serie is estimated
    alpha   = np.squeeze(alpha  )
    mu2     = np.squeeze(mu2    )
    sigma_e = np.squeeze(sigma_e)

    return alpha,mu2,sigma_e

# Wavelet power significant level
def WP_significant_level(wave,p):
    if wave.type()=='complex':
        return 0.5*chi2inv(p,dof=2)
    else:
        return 0.5*chi2inv(p,dof=1)


#def WPC_significant_level(t,scales,x1,x2,M=100):
    #assert x1.size == x2.size
    #N = x1.size

    #alpha1,mu1,sigma1 = estimate_AR1_parameters(x1)
    #alpha2,mu2,sigma2 = estimate_AR1_parameters(x2)
    
    #X1 = gen_AR1(alpha1,N,M)
    #X2 = gen_AR1(alpha2,N,M)

    #CWT1 = wave.WT(X1)
    #CWT2 = wave.WT(X1)

    #...




     



def _Pk(alpha,Wk,dt):
    return (1.0-alpha**2.0) / (1.0+alpha**2.0 -2.0*alpha*np.cos(Wk*dt))

def _Ps(scales,wave,alpha,Wk,dt):
    
    Pk = _Pk(alpha,Wk,dt)
    SW = np.outer(scales,Wk)

    Psi = wave.psi_hat(SW)
    Psi = np.abs(Psi)**2

    return np.mean(Pk[None,:]*Psi, axis=1)

if __name__ == '__main__':

    N=10000  #length of each serie
    M=10     #number of series
    m = 1.0                          #means
    a = 0.50 + np.random.rand(M)/2.0 #alphas
    se=1.5                           #sigma_es
    
    # generate AR1 series
    X = gen_AR1(m=m,alpha=a,N=N,sigma_e=se)

    # estimate parameters
    alpha,mu2,sigma_e = estimate_AR1_parameters(X)
    
    # compare
    print 'means=',np.round(np.mean(X,axis=1),2),'~=', np.round(m/(1.0-a),2)
    print 'alpha=',np.round(alpha,3),'~=',np.round(a,3)
    print 'sigma_e',np.round(sigma_e,3),'~=',np.round(se,3)
    
    #dt=0.1
    #W = 2.0*np.pi*np.fft.fftfreq(N)/dt
    #plt.figure()
    #for alpha in np.linspace(0.0,0.8,5):
        #P = _Pk(alpha,W,dt)
        #plt.plot(np.fft.fftshift(W),np.fft.fftshift(P))
    #plt.show()

    print 0.5*chi2inv(0.95,2)


    




