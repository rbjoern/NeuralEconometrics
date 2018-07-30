#Preliminaries
import numpy as np
import pandas as pd
from scipy.special import expit

#This file has the following sections: 
    # Simulate data from distributions
    # Compute properties of y from drawn data
    # Compute g(x,beta) for various theoretical dgp's
        # First section has then functions actually used
        # Second one provides a number of other functions explored. Do ignore. 
    

##############################################################################
# Simulate data from distributions
##############################################################################
##############################################################################
### Beta-coefficients
def draw_beta(k=2, low_bound=-5, high_bound=5): 
    beta = np.random.randint(low = low_bound, high = high_bound, size=k)
    return beta

def draw_beta_normal(k=2, mu=1, stdev=1):
    beta = np.random.normal(loc=mu, scale=stdev, size=k)
    return beta

##############################################################################
### x distribution
def draw_x_normal(mu, cov, n=10**4, **kwargs):
    x = np.random.multivariate_normal(mean = mu, cov=cov, size = n)
    x = pd.DataFrame(x)
    return x 

def draw_x_normal_iv(mu, cov, g_function,
                     n=10**4, 
                     k = 1, c=1, z=1,
                     **kwargs):
    
    #Draw exogenous parts 
    x_exog  = np.random.multivariate_normal(mean = mu[0:k], cov=cov[0:k,0:k], size = n)
    c_exog  = np.random.multivariate_normal(mean = mu[k:k+c], cov=cov[k:k+c,k:k+c], size = n) 
#    z_exog  = np.random.multivariate_normal(mean = mu[k+c:], cov=cov[k+c:,k+c:], size = n) 
    z_exog  = np.random.multivariate_normal(mean = mu[-z:], cov=cov[-z:,-z:], size = n) 
    
    
    
    #Generate variables
    x_obs = np.empty(np.shape(x_exog))
    z_obs = np.empty(np.shape(z_exog))
    for p in range(0,k):
#        x_obs[:,p] = g_function['g_dgp'](x=np.concatenate((x_exog[:,p].reshape((len(x_exog[:,p]),1)), c_exog), axis=1), 
#                                 beta = np.ones(g_function['g_parameters'](1+c)))
        x_obs[:,p] = x_exog[:,p]+g_function['g_dgp'](x=c_exog, 
                         beta = np.ones(g_function['g_parameters'](c)))
        z_obs[:,p] = g_function['g_dgp'](x=np.concatenate((x_exog[:,p].reshape((len(x_exog[:,p]),1)), 
                                 z_exog[:,p].reshape((len(z_exog[:,p]),1))), axis=1), 
                                 beta = np.ones(g_function['g_parameters'](2)))
    c_obs = c_exog #No change. Just exogenous.  
    
    
    
    #Return output
    x = np.concatenate((x_obs,c_obs,z_obs), axis=1)
    x = pd.DataFrame(x)
    
    return x 

def draw_x_normal_iv_v2(mu, cov, g_function,
                     n=10**4, 
                     k = 1, c=1, z=1,
                     **kwargs):
    #Draw exogenous parts 
    data = np.random.multivariate_normal(mean = mu, cov=cov, size = n)
    x_exog = data[:, 0:k]
    c_exog = data[:,k:k+c]
    z_exog = data[:, -z:]

#     #Check correlations    
#    print(np.corrcoef(c_exog[:,0], x_exog, rowvar=False))
    
    #Generate variables
    x_obs = np.empty(np.shape(x_exog))
    for p in range(0,k):
#        x_obs[:,p] = g_function['g_dgp'](x=np.concatenate((x_exog[:,p].reshape((len(x_exog[:,p]),1)), z_exog), axis=1), 
#                                 beta = np.ones(g_function['g_parameters'](1+z)))
#        x_obs[:,p] = g_function['g_dgp'](x=np.concatenate((x_exog[:,p].reshape((len(x_exog[:,p]),1)), 
#                                 z_exog[:,p].reshape((len(z_exog[:,p]),1))), axis=1), 
#                                 beta = np.ones(g_function['g_parameters'](1+1)))
        x_obs[:,p] = x_exog[:,p]+g_function['g_dgp'](x=z_exog[:,p].reshape((len(z_exog[:,p]),1)), 
                                                     beta = np.ones(g_function['g_parameters'](1)))
    
    c_obs = c_exog 
    z_obs = z_exog 
    
    #Check correlation
#    print(np.corrcoef(x_obs, z_obs, rowvar=False))    

    #Return output
    x = np.concatenate((x_obs,c_obs,z_obs), axis=1)
    x = pd.DataFrame(x)
    
    return x 

def gen_x_normal_unitvariance_samemean(k=2, mean=1): 
    from sklearn.datasets import make_spd_matrix
    from scipy.linalg import fractional_matrix_power 
    
    mu = np.zeros(k)+mean #Averages
    cov = make_spd_matrix(n_dim=k) #Random covariance matrix 
    diag = np.zeros((k,k)) #Initialize with zeroes 
    np.fill_diagonal(diag, np.diag(cov)) #Get diagonal
    #Compute correlation matrix based on formula: 
    cov = fractional_matrix_power (diag, -0.5)@cov@fractional_matrix_power(diag, -0.5)
    return mu, cov

def gen_x_normal_unitvariance_randommean(k=2, mean=1): 
    from sklearn.datasets import make_spd_matrix
    from scipy.linalg import fractional_matrix_power 
    
    mu = np.random.normal(loc=mean, scale = 1, size=k) #Random mean
    cov = make_spd_matrix(n_dim=k) #Random covariance matrix 
    diag = np.zeros((k,k)) #Initialize with zeroes 
    np.fill_diagonal(diag, np.diag(cov)) #Get diagonal
    #Compute correlation matrix based on formula: 
    cov = fractional_matrix_power(diag, -0.5)@cov@fractional_matrix_power(diag, -0.5)
    return mu, cov

def gen_x_normal_unitvariance_randommean_iv(k=2, c=1, z=1, mean=1): 
    from sklearn.datasets import make_spd_matrix
    from scipy.linalg import fractional_matrix_power 
    
    #Draw mean and covariance just as regularly. 
    mu = np.random.normal(loc=mean, scale = 1, size=k+c+z) #Random mean
    cov = make_spd_matrix(n_dim=k+c+z) #Random covariance matrix 
    diag = np.zeros((k+c+z,k+c+z)) #Initialize with zeroes 
    np.fill_diagonal(diag, np.diag(cov)) #Get diagonal
    #Compute correlation matrix based on formula: 
    cov = fractional_matrix_power(diag, -0.5)@cov@fractional_matrix_power(diag, -0.5)
    
    #Impose no correlation between c and z. 
    cov[k+c:, k:k+c]  = np.zeros(shape=(z,c))
    cov[k:k+c, k+c:] = np.zeros(shape=(c,z))
    
    return mu, cov

def gen_x_normal_randomcov(k=2, mean=1):
    from sklearn.datasets import make_spd_matrix
    mu = np.random.normal(loc=mean, scale = 1, size=k) #Random mean
    cov = make_spd_matrix(n_dim=k) #Random covariance matrix 
    return mu, cov

##############################################################################
# Draw u (error term)
def draw_y_logiterror(g, stdev=1): #Latent variable with logistic error 
    u = np.random.logistic(loc = 0, scale = stdev, size = len(g))
    #print(u[0])
    y_star = g + u #Latent variable
    y = (y_star>0) #Observed 
    y = pd.DataFrame(y)
    return y

def draw_u_logit(n, stdev=1): #Latent variable with logistic error 
    u = np.random.logistic(loc = 0, scale = stdev, size = n)
    return u

def draw_u_normal(n, stdev=1): #
    u = np.random.normal(loc=0, scale = stdev, size=n)
    return u 

##############################################################################
# Draw epsilon (measurement error for regressors)

def gen_error_normal(k=2, scale = 2): 
    from sklearn.datasets import make_spd_matrix
    from scipy.linalg import fractional_matrix_power 
    
    mu = np.zeros(k) #Zero error in expectation
    cov = make_spd_matrix(n_dim=k) #Random covariance matrix 
    diag = np.zeros((k,k)) #Initialize with zeroes 
    np.fill_diagonal(diag, np.diag(cov)) #Get diagonal
    diag = (1/scale)*diag #Adjust scale
    #Compute correlation matrix based on formula: 
    cov = fractional_matrix_power (diag, -0.5)@cov@fractional_matrix_power(diag, -0.5)
    return mu, cov

def gen_error_normal_indep(k=2, scale = 2): 
    
    mu = np.zeros(k) #Zero error in expectation
    cov = scale*np.eye(k) #Return eye
    
    return mu, cov

##############################################################################
# Compute properties of y from drawn data
##############################################################################
def gen_y_latent(g,u): 
    y_star = g + u #Latent variable
    y = (y_star>0) #Observed 
    y = pd.DataFrame(y)
    return y

def predict_latent(x,y,g): 
    F = logit_cdf(g)
    y_hat = pd.DataFrame(F>=0.5)
    accuracy = float(np.mean((y_hat==y))) 
    return F, y_hat, accuracy

def gen_y_reg(g,u): 
    y = g+u
    return pd.DataFrame(y)

##############################################################################
### Squashing functions and their (outer) derivatives
def logit_cdf(s):  #Vulnerable to overflow 
#    Lambda = 1/(1+np.exp(-s))
    Lambda = expit(s) #Imported from scipy, slightly more efficient (and numerically stable)
    return Lambda 

def logit_cdf_prime(s): 
    Lambda_prime = logit_cdf(s)*(1-logit_cdf(s))
    return Lambda_prime 

def linear_output(s): #No change
    return s

def linear_output_prime(s): 
    return np.ones(shape=np.shape(s))

##############################################################################
### DGP MARGINAL EFFECTS
def mrgeff_logit(g,g_prime): 
    F_prime = logit_cdf_prime(g).ravel()
    mrgeff = np.reshape(F_prime, (len(F_prime),1))*g_prime
    return mrgeff


def mrgeff_dgp(g, g_prime, y_squashing_prime): 
    F_prime = y_squashing_prime(g).ravel()
    mrgeff = np.reshape(F_prime, (len(F_prime),1))*g_prime
    return mrgeff

##############################################################################
# DEFINE THE UNDERLYING g-FUNCTIONS (see paper)
##############################################################################
##############################################################################
### CHECKER FUNCTION
def g_checker(g_function, parameters=3): #Used to check properties of new dgp functions
    n=1000
    x_mean=1
    k = 3 
    xmu, xcov = gen_x_normal_unitvariance_randommean(k, mean=x_mean)
    x = draw_x_normal(mu = xmu, cov=xcov,n=n)
    beta = draw_beta_normal(k=parameters, mu=0, stdev=1)
    g = g_function(x=x, beta=beta)
    print('Share of positive g\'s:', np.sum(g>0)/n)
    y = draw_y_logiterror(g)
    print('Share of one in y:', np.sum(y==1)/n)
    print('Average g:', np.mean(g))
    print('Median g:', np.median(g))
    print('Minimum  g:', np.min(g))
    print('Maximum g:', np.max(g))
    #return g

##############################################################################
### Basic linear scenario
def g_logit(x,beta, **kwargs): 
    g = np.dot(x,beta)
    return g
    
def g_logit_prime(x,beta, **kwargs):
    gprime = np.resize(beta,(len(x),len(beta)))
    return gprime

def g_logit_prime_beta(x,beta):
    gprime = x
    return gprime

def g_logit_hess_beta(x,beta):
    gprime = np.zeros((len(x),len(beta)))
    return gprime

def g_logit_pars(k): #Calculate number of parameters necessary for g
    return k

###############################################################################
### INTERACTION TERMS 
from sklearn.preprocessing import PolynomialFeatures 

def g_interactions(x, beta, **kwargs): 
    x_transformed  = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(x)
    g = g_logit(x_transformed, beta)
    return g

def g_interactions_prime(x, beta, **kwargs): 
    k = len(x.T)
    gprime= g_logit_prime(x, beta[0:k]) #Non-interaction terms
    var_names = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit(x).get_feature_names()
    for i in range(0, k): 
        var_name = 'x'+str(i)
        coeff_indices = [i for i,s in enumerate(var_names) if var_name in s][1:] #Get coefficients only for interaction terms
        gprime[:,i] += np.sum(beta[coeff_indices]*x.drop(i, axis=1), axis=1)
    return gprime

def g_interactions_pars(k): 
    pars = int(0.5*(k+k**2))
    return pars 
###############################################################################
### POLYNOMIAL & INTERACTIONS
    
def g_linear_poly(x,beta, interaction_only=False): 
    x_transformed  = PolynomialFeatures(2, interaction_only=interaction_only, include_bias=False).fit_transform(x)
    g = g_logit(x_transformed, beta)
    return g 
    
def g_linear_poly_prime(x,beta, interaction_only=False): 
    k = len(x.T)
    var_names = PolynomialFeatures(2, interaction_only=interaction_only, include_bias=False).fit(x).get_feature_names()
    
    #Add change from standard terms
    gprime= g_logit_prime(x, beta[0:k]) # Just their beta values 
    
    #Add change from interaction terms 
    for i in range(0, k): 
        var_name = 'x'+str(i)
        #Find the index of each interaction term
        beta_indices = [i for i,s in enumerate(var_names) \
                            if (var_name in s.split(' ')) & ('^2' not in s)][1:] #1: drops x itself. 
#        print(var_names)
#        print(var_name)
#        print(beta_indices)
        #Multiply the beta of each interaction term with the opposing x (through drop)
        gprime[:,i] += np.sum(beta[beta_indices]*x.drop(i, axis=1), axis=1)
        
    if interaction_only==False: 
        #Add change from quadratic terms. 
        beta_indices = [i for i,s in enumerate(var_names) if ('^2' in s)]
        gprime += 2*beta[beta_indices]*x
    return gprime
    
##############################################################################
### POLYNOMIAL SUMS

### General functions
def g_polynomial(x,beta,order=3, **kwargs):
    k = len(x.T)
    g = []
    for i in range(1,order+1):
        order_beta = beta[(i-1)*k:i*k]
        g.append(np.sum(order_beta*(x**i), axis=1))
    g = np.sum(g, axis=0)
    return g

def g_polynomial_prime(x,beta,order=3, **kwargs):
    g_prime = []
    x = np.array(x)
    k = len(x.T)
    
    for i in range(1,order+1):
        order_beta = beta[(i-1)*k:i*k]
        g_prime.append(i*order_beta*(x**(i-1)))
    g_prime = np.sum(g_prime, axis=0)
    return g_prime

def g_polynomial_prime_beta(x,beta,order=3, **kwargs): 
    x = np.array(x)
    prime = x
    for i in range(2,order+1):
        prime = np.concatenate((prime, x**i), axis=1)
    return prime 

### Second order 
def g_polynomial_2(x,beta,order=2, **kwargs):        
    g = g_polynomial(x,beta,order=order)
    return g

def g_polynomial_prime_2(x,beta,order=2, **kwargs):
    gprime = g_polynomial_prime(x,beta,order=order)
    return gprime

def g_polynomial_par_2(k, order=2):
    pars = order*k
    return pars 

def g_polynomial_prime_beta_2(x,beta,order=2, **kwargs):
    prime = g_polynomial_prime_beta(x,beta,order=order)
    return prime

### Third order 
def g_polynomial_3(x,beta,order=3, **kwargs):
    g = g_polynomial(x,beta,order=order)
    return g
    
def g_polynomial_prime_3(x,beta,order=3, **kwargs):
    gprime = g_polynomial_prime(x,beta,order=order)
    return gprime    

def g_polynomial_par_3(k, order=3):
    pars = order*k
    return pars 

def g_polynomial_prime_beta_3(x,beta,order=3, **kwargs):
    prime = g_polynomial_prime_beta(x,beta,order=order)
    return prime

### Previously defined in dict, but pickle problems :/
#g_polynomials, g_polynomial_primes = {}, {} #Store polynomials of various orders in a dict
#for i in range(0,21):
#    def g_polynomial_loop(x,beta,order=i, **kwargs):
#        k = len(x.T)
#        g = []
#        for i in range(1,order+1):
#            order_beta = beta[(i-1)*k:i*k]
#            g.append(np.sum(order_beta*(x**i), axis=1))
#            #g.append(np.sum((beta/(i))*(x**i), axis=1))
#        g = np.sum(g, axis=0)
#        return g
#    g_polynomials[i] = g_polynomial_loop
#    
#    def g_polynomial_prime_loop(x,beta,order=i, **kwargs):
#        g_prime = []
#        x = np.array(x)
#        k = len(x.T)
#        for i in range(1,order+1):
#            order_beta = beta[(i-1)*k:i*k]
#            g_prime.append(i*order_beta*(x**(i-1)))
#        g_prime = np.sum(g_prime, axis=0)
#        return g_prime
#    g_polynomial_primes[i] = g_polynomial_prime_loop    

##############################################################################
#TRIGONOMETRIC POLYNOMIAL
def g_trigpol(x,beta,order=3, **kwargs):
    k = len(x.T)
    g = []
    for i in range(1,order+1):
        order_beta = beta[2*(i-1)*k:2*i*k]
        g.append(np.sum(order_beta[0:k]*np.sin(i*x)+order_beta[k:2*k]*np.cos(i*x), axis=1))
    g = np.sum(g, axis=0)
    return g

def g_trigpol_prime(x,beta,order=3, **kwargs):
    k = len(x.T)
    g_prime = []
    x = np.array(x)
    for i in range(1,order+1):
        order_beta = beta[2*(i-1)*k:2*i*k]
        g_prime.append(i*order_beta[0:k]*np.cos(i*x)-i*order_beta[k:2*k]*np.sin(i*x))
    g_prime = np.sum(g_prime, axis=0)
    return g_prime

def g_trigpol_prime_beta(x,beta,order=3, **kwargs):
    x = np.array(x)
    prime = np.sin(x)
    prime = np.concatenate((prime, np.cos(x)), axis=1)
    if order>1: 
        for i in range(2,order+1):
            prime = np.concatenate((prime, np.sin(i*x)), axis=1)
            prime = np.concatenate((prime, np.cos(i*x)), axis=1)
    return prime

### First order
def g_trigpol_1(x,beta,order=1, **kwargs):
    g = g_trigpol(x,beta,order=order)
    return g

def g_trigpol_prime_1(x,beta,order=1, **kwargs):
    g_prime = g_trigpol_prime(x,beta,order=order)
    return g_prime

def g_trigpol_par_1(k, order=1):
    pars = 2*order*k
    return pars 

def g_trigpol_prime_beta_1(x,beta, order=1, **kwargs):  
    prime = g_trigpol_prime_beta(x,beta,order=order)
    return prime

### Third order
def g_trigpol_3(x,beta,order=3, **kwargs):
    g = g_trigpol(x,beta,order=order)
    return g

def g_trigpol_prime_3(x,beta,order=3, **kwargs):
    g_prime = g_trigpol_prime(x,beta,order=order)
    return g_prime

def g_trigpol_par_3(k, order=3):
    pars = 2*order*k
    return pars 

def g_trigpol_prime_beta_3(x,beta, order=3, **kwargs):  
    prime = g_trigpol_prime_beta(x,beta,order=order)
    return prime


### Previously defined in dict, but pickle problems :/
##Define polynomials of a sequence of orders 
#g_trigpols, g_trigpol_primes = {}, {}
#for i in range(0,21):
#    def g_trigpol_loop(x,beta,order=i, **kwargs):
#        k = len(x.T)
#        g = []
#        for i in range(1,order+1):
#            order_beta = beta[2*(i-1)*k:2*i*k]
#            g.append(np.sum(order_beta[0:k]*np.sin(i*x)+order_beta[k:2*k]*np.cos(i*x), axis=1))
#        g = np.sum(g, axis=0)
#        return g
#    g_trigpols[i] = g_trigpol_loop
#    
#    def g_trigpol_prime_loop(x,beta,order=i, **kwargs):
#        k = len(x.T)
#        g_prime = []
#        x = np.array(x)
#        for i in range(1,order+1):
#            order_beta = beta[2*(i-1)*k:2*i*k]
#            g_prime.append(i*order_beta[0:k]*np.cos(i*x)-i*order_beta[k:2*k]*np.sin(i*x))
#        g_prime = np.sum(g_prime, axis=0)
#        return g_prime
#    g_trigpol_primes[i] = g_trigpol_prime_loop

##############################################################################
### Wiggly function (basically a sum of ranom pol and trigpol)
    
def g_wiggly(x,beta,dgp=True, order=2,a=0.33, **kwargs):
    k = len(x.T)
    beta_1 = beta[0:k]
    if dgp==True:
        beta_2 = a*beta[k:2*k]
    else: 
        beta_2 = beta[k:2*k]
    beta_3 = beta[2*k:3*k]
    beta_4 = beta[3*k:4*k]
    
    g = beta_1*x \
        + beta_2*x**order \
        + beta_3*x*np.sin(order*x) \
        + beta_4*x*np.cos(order*x)
    return g.sum(1)


def g_wiggly_prime(x,beta, dgp=True, order=2, a=0.33, **kwargs):
    k = len(x.T)
    beta_1 = beta[0:k]
    if dgp==True:
        beta_2 = a*beta[k:2*k] #Reduce size for wellbehaved function
    else: 
        beta_2 = beta[k:2*k] #Parameter a just learned inside weights
    beta_3 = beta[2*k:3*k]
    beta_4 = beta[3*k:4*k]
    
    g_prime =  np.resize(beta_1,(len(x),len(beta_1))) \
                + order*beta_2*x**(order-1) \
                + beta_3*np.sin(order*x)  \
                + order*beta_3*x*np.cos(order*x) \
                + beta_4*np.cos(order*x) \
                - order*beta_4*x*np.sin(order*x)
#    
    return g_prime

def g_wiggly_pars(k): 
    return 4*k

def g_wiggly_prime_beta(x,beta, **kwargs):
    x = np.array(x)
    
    prime = x 
    prime = np.concatenate((prime, x**2), axis=1)
    prime = np.concatenate((prime, x*np.sin(2*x)), axis=1)
    prime = np.concatenate((prime, x*np.cos(2*x)), axis=1)
    
    return prime

##############################################################################
### Pointy function (Hartford et al 2017)

def g_pointy(x,beta, dgp=True, a=30, b=0.5):
    k = len(x.T)
    beta_1 = beta[0:k]
    beta_2 = beta[k:2*k]
    if dgp==True: 
        beta_3 = a*beta[2*k:3*k] 
#        beta_3=a*np.ones((len(x), k))
    else: 
        beta_3 = beta[2*k:3*k] 
#        beta_3=a*np.ones((len(x), k))
        b = beta[-1]
#        b=1
    beta_4 = beta[3*k:4*k] 
#    beta_4 = np.ones(k)
    
    g = beta_1*x\
        +beta_2*x**2\
        +beta_3*np.exp(-b*(x-beta_4)**2)
    
    return g.sum(1)

def g_pointy_prime(x,beta, dgp=True, a=30, b=0.5):
    k = len(x.T)
    beta_1 = beta[0:k]
    beta_2 = beta[k:2*k]
    if dgp==True: 
        beta_3 = a*beta[2*k:3*k] 
#        beta_3=a*np.ones((len(x), k))
    else: 
        beta_3 = beta[2*k:3*k] 
#        beta_3=a*np.ones((len(x), k))
        b = beta[-1]
#        b=1
    beta_4 = beta[3*k:4*k] 
#    beta_4 = np.ones(k)
    
    g_prime =  np.resize(beta_1,(len(x),len(beta_1))) \
                + 2*beta_2*x \
                - 2*b*beta_3*np.exp(-b*(x-beta_4)**2)*(x-beta_4)
    return g_prime 

def g_pointy_pars(k): 
    return 4*k+1

def g_pointy_prime_beta(x,beta, **kwargs):
    k = len(x.T)
#    beta_1 = beta[0:k]
#    beta_2 = beta[k:2*k]
    beta_3 = beta[2*k:3*k] 
    b = beta[-1]
    a = 1
    beta_4 = beta[3*k:4*k] 
#    a = 1 #Nonidentified parameters simply set to one. Coefficients will adjust. 
    
    prime_1 = x
    prime_2 = x**2
    prime_3 = a*np.exp(-b*(x-beta_4)**2)
#    prime_3 = np.zeros((len(x), k))
    prime_4 = 2*a*b*beta_3*np.exp(-b*(x-beta_4)**2)*(x-beta_4)
#    prime_4 = np.zeros((len(x), k))
    prime_b = (-a*beta_3*np.exp(-b*(x-beta_4)**2)*(x-beta_4)**2).sum(1).reshape((len(x),1))
#    prime_b = np.zeros((len(x), 1))
    
    prime = np.concatenate((prime_1, prime_2,prime_3,prime_4,prime_b), axis=1)
    return prime 
    
##############################################################################
### ACKLEY FUNCTION (https://www.sfu.ca/~ssurjano/ackley.html)
def g_ackley(x,beta, dgp=True, a=1/1, b=10,c=1/5, d=2,e=2*np.pi, f = 0.75): 
    k = len(x.T)
    if dgp == True: #The dgp knows fixed parameters
        beta_1 = np.ones(k)
    else: #Mle does not know parameters 
#        beta_1 = beta[0:k]
        beta_1 = np.abs(beta[0:k])
#        a = beta[2*k+1-1]
        b = beta[2*k+2-1]
        c = beta[2*k+3-1]
        d = beta[2*k+4-1]
        e = beta[2*k+5-1]
        f = beta[2*k+6-1]
    beta_2 = beta[k:2*k]
    sum_1 = np.sum(beta_1*x**2, axis=1)
    sum_2 = np.sum(beta_2*np.cos(e*x), axis=1)
    g = a*(-b*np.exp(-c*np.sqrt((1/k)*sum_1))-d*np.exp((1/k)*sum_2)+f*(b+np.exp(1)))
    return g 

def g_ackley_prime(x,beta, dgp=True, a=1/1, b=10,c=1/5, d=2,e=2*np.pi, f = 0.75): 
    k = len(x.T)
    
    if dgp == True: #The dgp knows fixed parameters
        beta_1 = np.ones(k)
    else: #Mle does not know parameters 
#        beta_1 = beta[0:k]
        beta_1 = np.abs(beta[0:k])
#        a = beta[2*k+1-1]
        b = beta[2*k+2-1]
        c = beta[2*k+3-1]
        d = beta[2*k+4-1]
        e = beta[2*k+5-1]
        f = beta[2*k+6-1]
    beta_2 = beta[k:2*k]
    sum_1 = np.sum(beta_1*x**2, axis=1)
    sum_2 = np.sum(beta_2*np.cos(e*x), axis=1)
    part_1 =  b*c*(np.exp(-c*np.sqrt((1/k)*sum_1)))/(k*np.sqrt((1/k)*sum_1))
    part_2 = (d*e/k)*np.exp((1/k)*sum_2)
    g_prime =a*(np.reshape(part_1.ravel(), (len(x),1))*beta_1*x\
                + np.reshape(part_2.ravel(), (len(x),1))*beta_2*np.sin(e*x))
#    print(type(g_prime))
    return np.array(g_prime)

def g_ackley_pars(k): 
    return 2*k+6


def g_ackley_prime_beta(x,beta): #Gradient vis a vis coefficients (see appendix)
    k = len(x.T)
#    beta_1 = beta[0:k]
    beta_1 = np.abs(beta[0:k])
#    a = beta[2*k+1-1]
    a=1/1 #Nonidentified parameters simply set to one. Coefficients will adjust. 
    b = beta[2*k+2-1]
    c = beta[2*k+3-1]
    d = beta[2*k+4-1]
    e = beta[2*k+5-1]
    f = beta[2*k+6-1]
    beta_2 = beta[k:2*k]
    
    sum_1 = np.sum(beta_1*x**2, axis=1).ravel().reshape((len(x),1))
    sqrt_1 = np.sqrt((1/k)*sum_1)
    exp_1 = np.exp(-c*sqrt_1)
    sum_2 = np.sum(beta_2*np.cos(e*x), axis=1).ravel().reshape((len(x),1))
    exp_2 = np.exp((1/k)*sum_2)
    sum_3 = np.sum(beta_2*np.sin(e*x)*x, axis=1).reshape((len(x),1))
    
    prime_beta1 = a*b*c*exp_1/(2*k*sqrt_1)*(beta[0:k]/beta_1)*x**2
#    prime_beta1 = relu(prime_beta1) #Drop negative values
    prime_beta2= (-a*d/k)*exp_2*np.cos(e*x)
#    prime_a = -b*exp_1-d*exp_2+f*(b+np.exp(1))
    prime_a = np.zeros(shape=(len(x),1))
    prime_b = -a*exp_1+a*f
    prime_c = a*b*exp_1*sqrt_1
    prime_d = -a*exp_2
    prime_e = a*d*exp_2*(1/k)*sum_3
    prime_f = np.resize(a*(b+np.exp(1)), (len(x),1))
    
    prime = np.concatenate((prime_beta1,prime_beta2, 
                            prime_a, prime_b, prime_c, prime_d, prime_e,prime_f,
                            ), axis=1)
    
    return prime
    
    

##############################################################################
### RASTRIGIN FUNCTION (https://www.sfu.ca/~ssurjano/rastr.html)
def g_rastrigin(x,beta, dgp=True, a=1/6, b=0.1, c=10):
    k = len(x.T)
    beta_1 = beta[0:k]
    beta_2 = beta[k:2*k]
    if dgp==False: 
        a = 1
        b = 1
        c = 1
    
    g = a*(b*beta_1*x**2-c*beta_2*np.cos(2*np.pi*x))
    #g = (1/(B))*(beta*x**2-A*np.cos(2*np.pi*x))
    #g = (beta*x**2-A*np.cos(2*np.pi*x))
    #g = A*k+np.sum(g,axis=1)
    g = np.sum(g,axis=1)
    return g 
#g_checker(g_rastrigin)

def g_rastrigin_prime(x,beta, dgp=True, a=1/6, b=0.1, c=10):
    k = len(x.T)
    beta_1 = beta[0:k]
    beta_2 = beta[k:2*k]
    if dgp==False: 
        a = 1
        b = 1
        c = 1
    
    g_prime = a*(2*b*beta_1*x+c*2*np.pi*beta_2*np.sin(2*np.pi*x))
    return np.array(g_prime) 

def g_rastrigin_pars(k): 
    return 2*k


def g_rastrigin_prime_beta(x,beta):
#    k = len(x.T)
#    beta_1 = beta[0:k]
#    beta_2 = beta[k:2*k]
    a = 1 #Nonidentified parameters simply set to one. Coefficients will adjust. 
    b = 1
    c = 1
    
    prime_1 = a*b*x**2
    prime_2 = -a*c*np.cos(2*np.pi*x)
    prime = np.concatenate((prime_1,prime_2), axis=1)
    return prime

##############################################################################
### Drop-wave (https://www.sfu.ca/~ssurjano/drop.html)
def g_dropwave(x,beta, dgp=True, a=30,b=2, c=3,d=10, e=0.05, f=0.15):
    #Lower C increases oscilliations of waves. 
    #Greater B increases the number of waves
    k = len(x.T)
    if dgp == True: #The dgp knows fixed parameters
        beta_1 = np.ones(k)
    else: #Mle does not know parameters 
#        beta_1 = beta[0:k]
        beta_1 = np.abs(beta[0:k])
#        a = beta[2*k+1-1]
        a=1 #Nonidentified parameters simply set to one. Coefficients will adjust. 
        b = beta[-5]
        c = beta[-4]
        d = beta[-3]
        e = beta[-2]
        f = beta[-1]
    beta_2 = beta[k:2*k]
    sum_1 = np.sqrt(np.sum(beta_1*x**2, axis=1))
    sum_2 = np.sum(beta_2*x**2, axis=1)  
    g = a*(-(b+np.cos(c*sum_1))/(d+e*sum_2)+f)
    return g

def g_dropwave_prime(x,beta, dgp=True, a=30,b=2, c=3,d=10, e=0.05, f=0.15):
    k = len(x.T)
    if dgp == True: #The dgp knows fixed parameters
        beta_1 = np.ones(k)
    else: #Mle does not know parameters 
#        beta_1 = beta[0:k]
        beta_1 = np.abs(beta[0:k])
#        a = beta[2*k+1-1]
        a=1 #Nonidentified parameters simply set to one. Coefficients will adjust. 
        b = beta[-5]
        c = beta[-4]
        d = beta[-3]
        e = beta[-2]
        f = beta[-1]
    beta_2 = beta[k:2*k]
    sum_1 = np.sqrt(np.sum(beta_1*x**2, axis=1))
    sum_2 = np.sum(beta_2*x**2, axis=1)  
    part_1 = (c*np.sin(c*sum_1))/(sum_1*(d+e*sum_2))
    part_2 = (2*e*(b+np.cos(c*sum_1)))/((d+e*sum_2)**2)
    g_prime =a*(np.reshape(part_1.ravel(), (len(x),1))*beta_1*x\
                +np.reshape(part_2.ravel(), (len(x),1))*beta_2*x)
    return np.array(g_prime)

def g_dropwave_pars(k): 
    return 2*k+5

def g_dropwave_prime_beta(x,beta): 
    k = len(x.T)
    beta_1 = np.abs(beta[0:k])
    a=1 #Nonidentified parameters simply set to one. Coefficients will adjust. 
    b = beta[-5]
    c = beta[-4]
    d = beta[-3]
    e = beta[-2]
    f = beta[-1]
    beta_2 = beta[k:2*k]

    sum_1 = np.sqrt(np.sum(beta_1*x**2, axis=1)).ravel().reshape((len(x),1))
    sum_2 = np.sum(beta_2*x**2, axis=1).ravel().reshape((len(x),1))
    part_1 = np.sin(c*sum_1)/(sum_1*(d+e*sum_2))
    part_2 = (b+np.cos(c*sum_1))/((d+e*sum_2)**2)
    
    prime_1 = a*c*0.5*part_1*(beta_1/np.abs(beta_1)*x**2)
    prime_2 = a*e*part_2*x**2
    prime_b = -a/(d+e*sum_2)
    prime_c = a*(np.sin(c*sum_1))/((d+e*sum_2))*sum_1
    prime_d = a*part_2
    prime_e = a*part_2*sum_2
    prime_f = a*np.ones(shape=((len(x),1))) 
    
    prime = np.concatenate((prime_1,prime_2, 
                           prime_b, prime_c, prime_d, prime_e, prime_f),
                           axis =1)
    return prime
    
    
##############################################################################
##############################################################################
##############################################################################
### Explored scenarios I did not end up using. 
##############################################################################
### EXPONENTIAL
def g_exp(x,beta):
    g = beta*np.exp(x)
    g = np.sum(g,axis=1)
    return g

def g_exp_prime(x,beta): 
    g_prime = beta*np.exp(x)
    return g_prime

def g_expall(x,beta, A=10):
    #g = (1/(1+np.exp(np.prod(beta))))*np.exp(-np.dot(x,beta))-(1+np.exp(np.prod(beta)))
    g = np.exp(-np.dot(x,beta))/(np.dot(x,beta))-np.dot(x,beta)
    return g
#np.random.seed(13)
#g_checker(g_expall)

#def g_log(x,beta): #Invalid since negative values :(
#    g = beta*np.log(x)
#    g = np.sum(g,axis=1)
#    return g

def g_logabs(x,beta): #Invalid since negative values :(
    g = beta*np.log(np.absolute(x))
    #g = beta*(np.absolute(x))
    g = np.sum(g,axis=1)
    return g

def comp_abs_prime(s): 
    prime = s.copy()
    prime[s>=0] = 1 #Not defined at zero - set to 1. 
    prime[s<0]  = -1 #np.divide(s,np.absolute(s))
    return prime 

def g_logabs_prime(x,beta):
    #g_prime = (beta/np.absolute(x))*comp_abs_prime(x)
    g_prime = beta/x
    return g_prime

##############################################################################
### RADIAL FUNCTION
def phi_pdf(x): #Standard normal PDF, used below. 
    phi = np.exp(-x**2/2)/np.sqrt(2*np.pi)
    return phi

def g_radial(x,beta, k=100):
    phi = beta*phi_pdf(x)
    g = (k**2)*np.prod(phi, axis=1)#-1
    return g
#g_checker(g_radial)

def g_radial_prime(x,beta):
    g = g_radial(x,beta)
    g_prime = g.values.reshape(len(g),1)*(-x)
    return g_prime

##############################################################################
### CIRCLE FUNCTION 
def g_circle(x,beta,A=5, R=2): 
    sum_1 = np.sum((x-beta)**2, axis=1)
    g = A*(np.sqrt(sum_1)-R)
    return g

##############################################################################
### ROSENBROCK function
def g_rosenbrock(x,beta, A=1):
    k = len(x.T)
    g = []
    for i in range(0,k-1):
        g.append(A*beta[i+1]*(x[i+1]-x[i]**2)**2+beta[i]*(1-x[i])**2)
    g = np.sum(g, axis=0)
    return g 

##############################################################################
### Bohachevsky (https://www.sfu.ca/~ssurjano/boha.html)
def g_bowl(x,beta,a=1,b=1,c=1, d=10): 
    sum_1 = np.sum(beta*x**2, axis=1)
    sum_2 = np.sum(beta*np.cos(c*np.pi*x))
    g = a*sum_1+b*sum_2-d
    return g

##############################################################################
### Trid (https://www.sfu.ca/~ssurjano/trid.html)
def g_trid(x,beta):
    k = len(beta)
    sum_1 = np.sum(beta*(x-1)**2, axis=1)
    summer = []
    for i in range(0,k-1):
        summer.append(x[i]*x[i+1])
    sum_2 = np.sum(summer, axis=0)
    g = sum_1 + sum_2 
    return g 

###############################################################################
### CROSS-IN_TRAY
def g_crossintray(x,beta): 
    prod_1 = np.prod(np.sin(beta*x), axis=1)
    sum_1 = np.sum(x**2, axis=1)
    g = -10**(-4)*(np.absolute(prod_1*np.exp(np.absolute(100-np.sqrt(sum_1)/np.pi)))+1)
    return g 
#g_checker(g_crossintray)

###############################################################################
#### MICHALEWICZ (https://www.sfu.ca/~ssurjano/michal.html)
#def g_michaelewicz(x,beta):
#    np.sum()

##############################################################################
### Griewank 
def g_griewank(x,beta): 
    sum_1 = np.sum(beta*x**2, axis=1)      
    prod_1 = np.prod(beta*np.cos(beta*x), axis=1)
    g = sum_1 + prod_1
    return g 

##############################################################################
### VARIOUS ODDITIES (UNUSED)
def g_weird_rosenbrock(x,beta): #Didn't work very well. 
    g = -(1-beta[0]*x[0])**2+1*(beta[0]*x[0]-(beta[1]*x[1])**2)**2
    g = (g-np.mean(g))/np.std(g)
    return g

def g_sphere(x,beta): #UNFINISHED
    g = np.sum(beta*x**2, axis=1)
    return g 

def g_sin(x,beta):
    g = np.sum(beta*np.sin(x), axis=1)
    return g

def g_sin_prime(x,beta): #Unfinished 
    g = np.sum(beta*np.cos(x), axis=1)
    return g

def g_chaos(x,beta):
    g = np.sum(beta*x*(1-x), axis=1)
    return g

def g_cobbdouglas(x,beta, A=1):
    g = A*np.prod(x**beta, axis=1)/(np.prod(2**beta))-0.5
    return g


def g_xpower(x,beta): 
    g = np.sum(beta**x, axis=1)
    return g 
#g_checker(g_xpower)

#k=6
#xmu, xcov = gen_x_normal_unitvariance_randommean(k, mean=x_mean)
#a1 = draw_x_normal(mu = xmu, cov=xcov,n=n)
#a = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(a1)
#a2 = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit(a1).get_feature_names()
#a3 = [i for i,s in enumerate(a2) if 'x3' in s ][1:] #Get coefficients only for interaction terms
#a4 = a1.drop(0,axis=1)


