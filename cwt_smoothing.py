import numpy as np

#INPUT:
#   CWT: Matrix of shape (P,N) 
#       -P is the number of scales
#       -N is the number of times
#   window_size (H,W): size of the smoothing window on each axes (should be odd)
#       -H: window height = number of scales smoothed
#       -W: window width  = numver of times smoothed
#   F_filter: (optional) default = np.mean
#       - array function applied to the data inside the window
#OUTPUT:
#   SCWT: Smoothed matrix
#
def cwt_smooth(CWT, window_size, F_filter=np.mean):

    #check inputs
    assert (CWT.ndim == 2), 'Provided CWT is not a matrix!'
    P = CWT.shape[0] #number of scales
    N = CWT.shape[1] #number of times
    
    #window size should be odd
    H = window_size[0]
    W = window_size[1]
    assert H%2==1 and W%2==1, 'Smoothing winsow size is not odd!'
    assert H < P, 'Smoothing window height >= Scales count !' 
    assert W < N, 'Smoothing window width  >= Times  count !' 
    
    SCWT = np.zeros(CWT.shape, dtype=CWT.dtype)
    for i in xrange(P):
        imin = max(0, i-H//2)
        imax = min(P, i+H//2)
        for j in xrange(N):
            jmin = max(0, j-W//2)
            jmax = min(N, j+W//2)
            
            SCWT[i,j] = F_filter(CWT[imin:imax+1,jmin:jmax+1])

    return SCWT


#test function
if __name__ == '__main__':
    
    #create matrix
    shape = (5,5)
    CWT = np.round(np.random.rand(*shape),2)
    print 'CWT:\n',CWT,'\n'
    
    #mean smoothing over scales
    ws = (3,1)
    S = cwt_smooth(CWT,ws)
    print 'Mean over scales:\n',np.round(S,2),'\n'
    
    #mean smoothing over times
    ws = (1,3)
    S = cwt_smooth(CWT,ws)
    print 'Mean over times:\n',np.round(S,2),'\n'
    
    #mean smoothing over scales and times
    ws = (3,3)
    S = cwt_smooth(CWT,ws)
    print 'Mean over scales and times:\n',np.round(S,2),'\n'


    #max smoothing over time and scales
    ws = (3,3)
    S = cwt_smooth(CWT,ws,np.max)
    print 'MAX over scales and times:\n',np.round(S,2),'\n'



