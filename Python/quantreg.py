from scipy.optimize import minimize
import numpy as np
from numpy.linalg import lstsq as ldiv

def quantreg(x, y, tau):
    def rho(r):
        temp = y - np.dot(x,r)
        return np.sum(np.abs(temp - (temp < 0) * temp / tau))
    pmean = ldiv(x, y, rcond=None)[0]
    # return minimize(rho, pmean, method='SLSQP', options={'disp': False}).x
    # return minimize(rho, pmean, method='SLSQP', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='COBYLA', options={'disp': False}).x
    # return minimize(rho, pmean, method='COBYLA', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='TNC', options={'disp': False}).x
    # return minimize(rho, pmean, method='TNC', tol=None, options={'maxiter': 25, 'disp': False}).x
    return minimize(rho, pmean, method='TNC', tol=None, options={'maxiter': 10 * len(pmean), 'disp': False}).x
    # return minimize(rho, pmean, method='L-BFGS-B', options={'disp': False}).x
    # return minimize(rho, pmean, method='L-BFGS-B', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='BFGS', options={'disp': False}).x
    # return minimize(rho, pmean, method='BFGS', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='CG', options={'disp': False}).x
    # return minimize(rho, pmean, method='CG', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='Powell', options={'disp': False}).x
    # return minimize(rho, pmean, method='Powell', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='Nelder-Mead', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='Nelder-Mead', options={'disp': False}).x
