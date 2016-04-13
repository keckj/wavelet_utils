
import numpy as np
import scipy as sp
from scipy import stats, optimize
import matplotlib.pyplot as plt

def argmin(f,x0,tol=1.0e-8,maxiter=100,args=(),fprime=None,fprime2=None):
    return sp.optimize.newton(f,x0,fprime,args,tol,maxiter,fprime2)

#find x where f(x) = y
def invert(f,y,x0,tol=1.0e-8):
    return argmin(f=lambda x: (f(x)-y)**2, x0=x0, tol=tol)

def chi2inv(p,dof,tol=1.0e-8):
    f = lambda x: sp.stats.chi2.cdf(x,df=dof)
    x0 = 3.0
    return invert(f,p,x0,tol)


def gen_AR1(m,alpha,N,sigma_e=1.0):

    assert alpha>=0.0 and alpha<=1.0

    En = np.random.randn(N)*sigma_e
    AR1 = np.zeros((N), dtype=float)

    AR1[0] = m + En[0]
    for k in xrange(1,N):
        AR1[k] = m + alpha*AR1[k-1] + En[k]

    return AR1

def estimate_AR1_parameters(X):
    assert X.ndim==1, 'X should be a vector!'

    N = X.size
    
    X = X - np.mean(X)

    # AR0 and AR1 covariances
    C0 = X.transpose().dot(X) / N                  
    C1 = X[0:N-1].transpose().dot(X[1:N]) / (N-1) 
     
    #A. Grinsteds
    A = C0* N**2
    B = (C0-C1)*(N-2) - (C0+C1)*N**2
    C = (C0-C1)*N + C1*N**2
    delta = B**2 - 4*A*C

    if delta > 0:
        a = (-B-delta**0.5)/(2.0*A)
        alpha=a
    else:
        raise Exception('Serie is too short, bad delta!')
    
    mu2 = -1.0/N + (2.0/N**2)*((N-a**N)/(1.0-a) - a*(1.0-a**(N-1)) / (1.0-a)**2)
    C0e = C0 / (1.0 - mu2)
    sigma_e = np.sqrt((1.0-a**2)*C0e)

    return a,mu2,sigma_e

# Wavelet power significant level
def WP_significant_level(wave,p):
    if wave.type()=='complex':
        return 0.5*chi2inv(p,dof=2)
    else:
        return 0.5*chi2inv(p,dof=1)

def _Pk(alpha,Wk,dt):
    return (1.0-alpha**2.0) / (1.0+alpha**2.0 -2.0*alpha*np.cos(Wk*dt))

def _Ps(scales,wave,alpha,Wk,dt):
    
    Pk = _Pk(alpha,Wk,dt)
    SW = np.outer(scales,Wk)

    Psi = wave.psi_hat(SW)
    Psi = np.abs(Psi)**2

    return np.mean(Pk[None,:]*Psi, axis=1)

if __name__ == '__main__':
    m = 1.0
    a = 0.88
    N=10000
    se=1.5
    X = gen_AR1(m,a,N,se)

    print np.mean(X),'~=', m/(1.0-a)
    
    alpha,mu2,sigma_e = estimate_AR1_parameters(X)

    print 'alpha=',alpha,'~=',a
    print 'sigma_e',sigma_e,'~=',se
    
    dt=0.1
    W = 2.0*np.pi*np.fft.fftfreq(N)/dt
    
    #plt.figure()
    #for alpha in np.linspace(0.0,0.8,5):
        #P = _Pk(alpha,W,dt)
        #plt.plot(np.fft.fftshift(W),np.fft.fftshift(P))
    #plt.show()

    print 0.5*chi2inv(0.95,2)


    




