
import time
import numpy as np
import matplotlib.pyplot as plt


#input: 
#   CWT: Crosswavelet transform of two signals 
#     *Matrix of shape (P,N) 
#       -P is the number of scales
#       -N is the number of times
#   deg: (optional) default=True
#        Outputs MCA and MSD in degree if True, else in radian
#output: 
#   MCA:  Mean circular angle for each time
#      *Vector of size (N)
#   MCSD: Mean circular standard deviation for each time
#      *Vector of size (N)
def crosswavelet_angle(XWT, deg=True):
    
    #check inputs
    assert (XWT.ndim == 2), 'Provided XWT is not a matrix!'
    
    #precompute sizes
    P = XWT.shape[0] #number of scales
    N = XWT.shape[1] #number of times
    
    # compute phase shift matrix (in radians)
    A = np.angle(XWT, deg=False)

    # compute the circular mean angle over all scales
    X = np.sum(np.cos(A), axis=0)
    Y = np.sum(np.sin(A), axis=0)
    MCA = np.arctan2(Y,X)


    # compute circular standard deviation 
    R = np.sqrt(X*X + Y*Y)
    MCSD = np.sqrt(-2.0*np.minimum(np.log(R/P),0.0))
    
    # Special case R/P == 1.0 + epsilon => log can be computed positive ~= +1e-16
    #eps = np.finfo(float).eps
    #arr = np.asarray([1.0+k*eps for k in xrange(-3,4)])
    #print np.log(arr)
    #print np.minimum(0.0,np.log(arr))
    # np.minimum is used to bring those small positive values to 0.0

    # convert to degrees if necessary
    if deg:
        MCA *= 180.0 / np.pi
        MCSD *= 180.0 / np.pi

    # check outputs
    assert (MCA.size == N) and (MCSD.size == N), 'Error: Output values length mismatch!'

    return MCA,MCSD


#test function
if __name__ == '__main__':

    # generate complex matrix
    P = 10
    N = 19
    shape = (P,N)
    
    t = np.linspace(-np.pi,np.pi,N)
    Re = np.ones(shape) * np.cos(t)[None,:]
    Im = np.ones(shape) * np.sin(t)[None,:]
    CWT = 1.0*Re + 1.0j*Im

    # compute cross wavelet angle and standard deviation
    MCA, MCSD = crosswavelet_angle(CWT, deg=True)

    # plot MCA +/- standard deviation
    plt.figure()

    plt.plot(t,MCA)
    #plt.plot(t,MCA+MCSD)
    #plt.plot(t,MCA-MCSD)

    plt.xlim([min(t),max(t)])
    plt.ylim([-180.0,+180.0])
    
    plot_deg = [-180.0 + k*30.0 for k in xrange(13)]
    plt.yticks(plot_deg, [str(x) for x in plot_deg], color='red')

    plt.show()

