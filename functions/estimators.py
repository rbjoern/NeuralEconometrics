#Preliminaries 
import numpy as np
import pandas as pd
import dgp_stuff as dgp

import statsmodels.api as sm
import warnings

#This file has the following sections (each one an estimator):     
    # Helper functions    
    # Logistic regression (basic)
    # Logistic regression (polynomial)
    # Ordinary least squares (OLS) regression (LPM)
    # Ordinary least squares (polynomial)
    # Two-stage least squares (2SLS)
    # Maximum likelihood estimation (MLE)
    # Non-parametric kernel regression (Nadaraya-Watson)
    # Semi-parametric kernel regression (Semi-Linear)
    # Semi-parametric kernel regression (Single Index)


##############################################################################
# Helper functions
##############################################################################
##############################################################################    
# Add constant to feature matrix 
def add_constant(x): 
    a = np.hstack((np.ones((len(x),1)),x))
    return a 

#Arbitrary level defaultdict
def dd_inf():
    from collections import defaultdict
    return defaultdict(dd_inf)

##############################################################################
# Logistic regression (basic)
##############################################################################
##############################################################################
### Estimate model

def estimate_logit(x,y, constant=True):
    if constant==True: 
        logit = sm.Logit(y,add_constant(x)) #Logistic regression
    else: 
        logit = sm.Logit(y,x) #Logistic regression
    result = logit.fit(disp = None)
    beta_hat = result.params
#    margeff_tjek = result.get_margeff(at='overall')
#    print(margeff_tjek.summary_frame())
    return beta_hat

##############################################################################
### Extract results
def logit_predict(x,y,beta): #Estimated predictions
    F = dgp.logit_cdf(np.dot(x,beta))
    y_hat = pd.DataFrame(F>=0.5)
    accuracy = float(np.mean((y_hat==y)))
    return F, y_hat, accuracy

def logit_prob(x,beta):
    if len(beta)>len(x.T): #Check if there is a constant term 
        s = np.dot(add_constant(x),beta)
    else: #If no constant term, it works fine
        s = np.dot(x,beta)
    prob = dgp.logit_cdf(s)
    return prob 

def logit_mrgeff(x, beta):
    if len(beta)>len(x.T): #Check if there is a constant term 
        s = np.dot(add_constant(x),beta)    
        beta = beta[1:] #remove constant term (after computing signal) because there is no marginal effect
    else: #If no constant term, it works fine
        s = np.dot(x,beta)
    mrgeff = dgp.logit_cdf_prime(s)*beta.reshape(len(beta),1)
    return mrgeff.T

##############################################################################
### Wrapper, which does it all

def estimator_logit(data, **kwargs): #data must be dict with form data[x/y][train/test]
    # Estimate model 
    betahat = np.array(estimate_logit(data['x']['Train'], data['y']['Train']))
    
    #Extract results
    prob, mrgeff = {}, {}
    for split in ('Train', 'Test'):
        prob[split]     = logit_prob(x=data['x'][split], beta=betahat)
        mrgeff[split]   = logit_mrgeff(x=data['x'][split],beta=betahat) 

    return betahat, prob, mrgeff


##############################################################################
# Logistic regression (polynomial)
##############################################################################
##############################################################################

##############################################################################
### Extract results
##Marginal effects
def logit_poly_mrgeff(x, x_transformed, beta, constant=True, interaction_only=False):
    if constant==True: 
        s = np.dot(add_constant(x_transformed),beta)    
        beta = beta[1:] #remove constant term (after computing signal) because there is no marginal effect
    else: #If no constant term, it works fine
        s = np.dot(x_transformed,beta)
    mrgeff = np.reshape(dgp.logit_cdf_prime(s), (len(s),1))*dgp.g_linear_poly_prime(x,beta)
    return mrgeff

##############################################################################
# Wrapper/estimator    
def estimator_logit_poly(data, **kwargs): 
    from sklearn.preprocessing import PolynomialFeatures     
    
    #Calculate quadratic features. 
    x_transformed = {}
    for split in ('Train', 'Test'): 
        x_transformed[split] = PolynomialFeatures(degree=2, interaction_only=False
                                 , include_bias=False).fit_transform(data['x'][split])
    
    # Estimate model (same as basic) 
    betahat = np.array(estimate_logit(x_transformed['Train'], data['y']['Train']))
    
    #Extract results
    prob, mrgeff = {}, {}
    for split in ('Train', 'Test'): 
        prob[split]     = logit_prob(x=x_transformed[split], beta=betahat) #(same)
        mrgeff[split]   = logit_poly_mrgeff(x=data['x'][split], x_transformed=x_transformed[split], beta=betahat) 

    return betahat, prob, mrgeff


##############################################################################
# Ordinary least squares (OLS) regression (LPM)
##############################################################################
##############################################################################
### Helper functions
def estimate_ols(x,y,constant=True):
    from statsmodels.regression.linear_model import OLS
    if constant==True: 
        reg = OLS(y,add_constant(x)) 
    else: 
        reg = OLS(y,x) #Logistic regression
    result = reg.fit()
    betahat = result.params
    return np.array(betahat)
    
def predict_ols(x,beta): 
    if len(beta)>len(x.T): #Check if there is a constant term 
        s = np.dot(add_constant(x),beta)
    else: #If no constant term, it works fine
        s = np.dot(x,beta)
    return s         

def mrgeff_ols(x,beta):
    if len(beta)>len(x.T): #Check if there is a constant term 
         mrgeff = np.resize(beta[1:], (len(x),len(beta[1:])))
    else: #If no constant term, it works fine
         mrgeff = np.resize(beta, (len(x),len(beta)))
    return mrgeff 


##############################################################################
### Wrapper, which estimates and extracts results in the preferred format
def estimator_ols(data, 
                  est_kwargs={}, splits = ('Train', 'Test'),
                  **kwargs): #data must be dict with form data[x/y][train/test]
    # Estimate model 
    betahat = np.array(estimate_ols(data['x']['Train'], data['y']['Train']))
    
    #Extract results
    expect, mrgeff = {}, {}
    for split in splits:
        expect[split]     = predict_ols(x=data['x'][split], beta=betahat)
        mrgeff[split]     = mrgeff_ols(x=data['x'][split], beta=betahat) 

    return betahat, expect, mrgeff
    

##############################################################################
# OLS (polynomial)
##############################################################################
##############################################################################
  

##############################################################################
### Extract results
def mrgeff_ols_poly(x, x_transformed, beta, constant=True, interaction_only=False):
    if constant==True: 
#        s = np.dot(add_constant(x_transformed),beta)    
        beta = beta[1:] #remove constant term (after computing signal) because there is no marginal effect
#    else: #If no constant term, it works fine
#        s = np.dot(x_transformed,beta)
    mrgeff = dgp.g_linear_poly_prime(x,beta)
    return mrgeff

##############################################################################
### Wrapper/estimator 
def estimator_ols_poly(data, est_kwargs={}, splits=('Train', 'Test'), **kwargs):
    from sklearn.preprocessing import PolynomialFeatures     
    
    #Calculate quadratic features. 
    x_transformed = {}
    for split in splits: 
        x_transformed[split] = PolynomialFeatures(degree=2, interaction_only=False
                                 , include_bias=False).fit_transform(data['x'][split])
    
    # Estimate model (same as basic) 
    betahat = np.array(estimate_ols(x_transformed['Train'], data['y']['Train']))
    
    #Extract results
    expect, mrgeff = {}, {}
    for split in splits: 
        expect[split]   = predict_ols(x=x_transformed[split], beta=betahat) #(same)
        mrgeff[split]   = mrgeff_ols_poly(x=data['x'][split], x_transformed=x_transformed[split], beta=betahat) 

    return betahat, expect, mrgeff

##############################################################################
# Two-stage least squares (2SLS)
##############################################################################
##############################################################################
### Helper functions
def estimate_2sls_1st_ols(x_endog, z, x_exog={}, constant=True, 
                      z_test={}, x_exog_test={}, 
                      **kwargs):
    from statsmodels.regression.linear_model import OLS
    
    if len(x_exog)==0: #Are there no exogenous covariates?
        covariates = z
        if len(z_test)>0:
            covariates_test = z_test
    else: 
        covariates  = np.concatenate((x_exog, z), axis=1) #Also uses exogenous covariates
        if len(z_test)>0:
            covariates_test  = np.concatenate((x_exog_test, z_test), axis=1
                                         )
    if constant==True: 
        covariates = add_constant(covariates)
        if len(z_test)>0:
            covariates_test = add_constant(covariates_test)
    
    x_hat = np.zeros(np.shape(x_endog))
    if len(z_test)>0:
        x_hat_test = np.zeros((len(z), len(x_endog.T)))
    
    
    for p in range(0, len(x_endog.T)): 
            reg = OLS(x_endog.iloc[:,p],covariates) 
            result = reg.fit()
            betahat = result.params
            x_hat[:,p] = predict_ols(x=covariates, beta=betahat)
            if len(z_test)>0:
                x_hat_test[:,p] = predict_ols(x=covariates_test, beta=betahat)
    if len(z_test)>0:
        return x_hat, x_hat_test    
    else: 
        return x_hat 

def estimate_2sls_2nd_ols(y_train, x_hat, x_exog = {},
                         x_hat_test = {}, x_exog_test = {}, 
                         **kwargs): 
    covariates = {} #Collect predicted endogenous and exogenous
    if len(x_exog)>0: 
        covariates['Train'] = np.concatenate((x_hat, x_exog), axis=1)
        if len(x_exog_test)>0: 
            covariates['Test'] = np.concatenate((x_hat_test, x_exog_test), axis=1)
    else: 
        covariates['Train'] = x_hat
        if len(x_hat_test)>0: 
            covariates['Test'] = x_hat_test
    
    # Estimate model 
    betahat = np.array(estimate_ols(covariates['Train'], y_train))
    
    #Extract results
    expect, mrgeff = {}, {}
    for split in covariates.keys():
        expect[split]     = predict_ols(x=covariates[split], beta=betahat)
        mrgeff[split]     = mrgeff_ols(x=covariates[split], beta=betahat) 


    #Return output 
#    if len(x_hat_test)>0:             
    return betahat, expect, mrgeff
#    else: 
#        return betahat, expect['Train'], mrgeff['Train']
    
##############################################################################
### Control function approach (adds residual to second stage)    
def estimate_2sls_2nd_ols_control(y_train, x_hat, x_endog, x_exog = {},
                         x_hat_test = {}, x_exog_test = {}, 
                         x_endog_test={},
                         **kwargs): 
    #Calculate residuals from first stage
    res = x_endog - x_hat 
    if len(x_endog_test)>0: 
        res_test = x_endog_test - x_hat_test
    
    covariates = {} #Collect endogenous, exogenous and residuals (control function)
    if len(x_exog)>0: 
        covariates['Train'] = np.concatenate((x_endog, x_exog, res), axis=1)
        if len(x_exog_test)>0: 
            covariates['Test'] = np.concatenate((x_endog_test, x_exog_test, res_test), axis=1)
    else: 
        covariates['Train'] = np.concatenate((x_endog, res), axis=1)
        if len(x_hat_test)>0: 
            covariates['Test'] = np.concatenate((x_endog_test, res_test), axis=1)
    
    # Estimate model 
    betahat = np.array(estimate_ols(covariates['Train'], y_train))
    
    #Extract results
    expect, mrgeff = {}, {}
    for split in covariates.keys():
        expect[split]     = predict_ols(x=covariates[split], beta=betahat)
        mrgeff[split]     = mrgeff_ols(x=covariates[split], beta=betahat) 

        # Drop control function (coefficients of residuals are not interesting)
        if len(x_exog)>0: 
            mrgeff[split] = mrgeff[split][:, 0:(len(x_endog.T)+len(x_exog.T))]
        else: 
            mrgeff[split] = mrgeff[split][:, 0:(len(x_endog.T))]
    

    #Return output 
#    if len(x_hat_test)>0:             
    return betahat, expect, mrgeff
#    else: 
#        return betahat, expect, mrgeff['Train']    

##############################################################################
### Wrapper, which allows for any two estimation procedures 
def estimator_2sls(data, est_kwargs={}, mrg_kwargs={},
                      exog={}, est_kwargs_first={}, regression=True, constant=True,
                      estimate_2sls_1st=estimate_2sls_1st_ols, 
                      estimate_2sls_2nd=estimate_2sls_2nd_ols, 
                      splits = ('Train', 'Test'),
                      **kwargs):

    if est_kwargs_first == {}: #If no special settings for first stage, use second stage
        est_kwargs_first = est_kwargs.copy()
    
#    print(data['x']['Train'].head())
    
    x_endog, x_exog = {},{}    
    for split in splits:
        if len(exog)>0: #x_exog provides boolean of exogenous covariates. 
            x_exog[split] = data['x'][split].iloc[:,exog]
            x_endog[split] = data['x'][split].iloc[:,~exog]
        else: #If no boolean, all x variables are assumed endogenous. 
            x_exog[split] = {}    
            x_endog[split] = data['x'][split]

#    print(x_endog['Train'].head)
#    print(x_exog['Train'].head)

    # 1st stage
    x_hat = {}
    if 'Test' in splits: 
        x_hat['Train'], x_hat['Test'] = estimate_2sls_1st(x_endog=x_endog['Train'], z=data['z']['Train'], 
                                              x_exog=x_exog['Train'], constant=constant, regression=regression,  
                                              est_kwargs=est_kwargs_first,
                                              z_test=data['z']['Test'], x_exog_test=x_exog['Test'])
    else: 
        x_hat['Train'] = estimate_2sls_1st(x_endog=x_endog['Train'], z=data['z']['Train'], 
                                              x_exog=x_exog['Train'], constant=constant, regression=regression,  
                                              est_kwargs=est_kwargs_first)
    
    #Second stage 
    if 'Test' in splits: 
        betahat, expect, mrgeff = estimate_2sls_2nd(y_train=data['y']['Train'], 
                                                       x_hat=x_hat['Train'], x_hat_test=x_hat['Test'],
                                                       x_exog = x_exog['Train'], x_exog_test=x_exog['Test'],
                                                       x_endog = x_endog['Train'], x_endog_test = x_endog['Test'],
                                                       regression=regression, constant=constant, 
                                                       est_kwargs=est_kwargs, mrg_kwargs=mrg_kwargs, 
                                                       **kwargs)
    else: 
        betahat, expect, mrgeff = estimate_2sls_2nd(y_train=data['y']['Train'], 
                                                       x_hat=x_hat['Train'], #x_hat_test=x_hat['Test'],
                                                       x_exog = x_exog['Train'], #x_exog_test=x_exog['Test'],
                                                       x_endog = x_endog['Train'], #x_endog_test = x_endog['Test'],
                                                       regression=regression, constant=constant, 
                                                       est_kwargs=est_kwargs, mrg_kwargs=mrg_kwargs, 
                                                       **kwargs)
        
        
    return betahat, expect, mrgeff

##############################################################################
### Estimator
def estimator_2sls_ols(data, 
                       est_kwargs={}, exog={},
                       splits = ('Train', 'Test'),
                       **kwargs): #data must be dict with form data[x/y][train/test]
    # The main point is to specify first and second stage estimation techniques. 
    betahat, expect, mrgeff = estimator_2sls(data = data, 
                                             est_kwargs = est_kwargs, 
                                             exog = exog, 
                                             constant=True,
                                             estimate_2sls_1st=estimate_2sls_1st_ols, 
                                             estimate_2sls_2nd=estimate_2sls_2nd_ols,
                                             splits = splits,
                                             **kwargs)

    return betahat, expect, mrgeff

def estimator_2sls_ols_control(data, 
                               est_kwargs={}, exog={},
                               splits = ('Train', 'Test'),
                               **kwargs): #data must be dict with form data[x/y][train/test]
    # The main point is to specify first and second stage estimation techniques. 
    betahat, expect, mrgeff = estimator_2sls(data = data, 
                                             est_kwargs = est_kwargs, 
                                             exog = exog, 
                                             constant=True,
                                             estimate_2sls_1st=estimate_2sls_1st_ols, 
                                             estimate_2sls_2nd=estimate_2sls_2nd_ols_control, 
                                             splits=splits,
                                             **kwargs)

    return betahat, expect, mrgeff
    

##############################################################################
# MAXIMUM LIKELIHOOD ESTIMATION (BASED ON DGP)
##############################################################################
##############################################################################

##############################################################################
### Preliminary functions
    
# Loglikelihood functions for classification based logistic errors 
def logistic_loglikelihood(x,y,beta, g=dgp.g_logit, epsilon=10**-5):
    #Calculates loglikelihood for binary model with logistic output function
    F = dgp.logit_cdf(g(x,beta, dgp=False))
    y = np.ravel(y)
    ll = y*np.log(F+epsilon) + (1-y)*np.log(1-F+epsilon) #Epsilon added to prevent log(0)
#    ll = g(x,beta)
    return ll.sum(0) 
#print(logistic_loglikelihood(y,x,beta))

def logistic_loglikelihood_grad(x,y,beta, g=dgp.g_logit, 
                                         g_prime = dgp.g_logit_prime_beta): 
    #Calulates gradient of loglikelihood with logistic output function
    F = dgp.logit_cdf(g(x,beta, dgp=False))
    y = np.ravel(y)
    prime = np.reshape(y-F, (len(y),1))*g_prime(x,beta)
    return prime.sum(0).ravel()

#def logistic_loglikelihood_hess(x,y,beta, g=dgp.g_logit, 
#                                         g_prime = dgp.g_logit_prime_beta, 
#                                         g_hess = dgp.g_logit_hess_beta):
#    F = dgp.logit_cdf(g(x,beta))
#    F_prime = logit_cdf_prime(g(x,beta))
#    y = np.ravel(y)
#    pprime = np.reshape(y-F, (len(y),1))*g_hess(x,beta) \
#            - F_prime*g_prime(x,beta).T*g_prime(x,beta)
#    return pprime
    
def loglikehood_normal_reg(x,y,beta, g=dgp.g_logit): 
#    sigma2 = beta[-1]
#    beta = beta[:-1]
    
#    ll = 1/2*np.log(2*np.pi)-1/2*np.log(sigma2) - 1/(2*sigma2)*(y-g(x,beta, dgp=False))**2
    ll = - 1/2*(y-g(x,beta, dgp=False))**2
    return ll.sum(0)

def loglikehood_normal_reg_grad(x,y,beta, g=dgp.g_logit, 
                                g_prime = dgp.g_logit_prime_beta): 
#    sigma2 = beta[-1]
#    beta = beta[:-1]
    
#    grad_beta = 1/(sigma2)*(y-g(x,beta, dgp=False))*g_prime(x,beta)
#    grad_sigma = -1/(2*sigma2)+1/(2*sigma2**2)*(y-g(x,beta, dgp=False))**2
#    grad = np.concatenate((grad_beta, grad_sigma))
    grad = (y-g(x,beta, dgp=False)).reshape((len(y),1))*g_prime(x,beta)
    return grad.sum(0).ravel()


##############################################################################
### Estimate model 
def estimate_mle(x,y, g_function, additional_parameters=0,
                 loglikelihood=logistic_loglikelihood, 
                 loglikelihood_grad=logistic_loglikelihood_grad):
    #Define MLE method based on g(x,beta)
    from statsmodels.base.model import GenericLikelihoodModel
    class MLE_DGP(GenericLikelihoodModel): 
        #Custom likelihood
        def loglike(self, params): 
            exog = self.exog
            endog = self.endog 
            return loglikelihood(exog, endog, params, g=g_function['g_dgp'])
        
        if 'g_dgp_prime_beta' in g_function.keys(): 
            def score(self, params): 
                exog = self.exog
                endog = self.endog 
                return loglikelihood_grad(exog,endog,params,g=g_function['g_dgp'], 
                                                            g_prime=g_function['g_dgp_prime_beta'])
        
        # Estimation stuff
        def __init__(self, endog, exog=None, loglike=None, score=None,
                 hessian=None, missing='none', extra_params_names=None,
                 **kwds): #Source code from statsmodel http://www.statsmodels.org/dev/_modules/statsmodels/base/model.html#GenericLikelihoodModel
            if loglike is not None:
                self.loglike = loglike
            if score is not None:
                self.score = score
            if hessian is not None:
                self.hessian = hessian
    
            self.__dict__.update(kwds)

            super(GenericLikelihoodModel, self).__init__(endog, exog,
                                                         missing=missing)
            #CHANGE: Adjusted number of parameters
            if exog is not None:
                #self.nparams = (exog.shape[1] if np.ndim(exog) == 2 else 1)
                self.nparams = g_function['g_parameters'](exog.shape[1] if np.ndim(exog) == 2 else 1)\
                                + additional_parameters #Can add additionals (such as variance)
    
            if extra_params_names is not None:
                self._set_extra_params_names(extra_params_names)

#    #Estimation    
    max_iterations = 500
#    start_parameters = None
    if additional_parameters>0: 
        start_parameters = np.concatenate(np.random.normal(size=g_function['g_parameters'](x.shape[1] if np.ndim(x) == 2 else 1)), 
                                          np.ones(shape=additional_parameters))
    else: # (random uniform on [0.1,0.9])
        start_parameters = 1+0.1+(0.9-0.1)*np.random.random(size=g_function['g_parameters'](x.shape[1] if np.ndim(x) == 2 else 1))
    
#    k = len(x.T)
#    if g_function['g_name'] == 'Ackley': #First beta cannot be negative
#        start_parameters = np.concatenate((1+np.random.random(size=k), 
#                                         np.random.normal(size=g_function['g_parameters']-k)))
#    print(start_parameters)

#    if 'g_dgp_prime_beta' in g_function.keys(): 
#        mle = MLE_DGP(y, x).fit(method='bfgs', disp=None, 
#                 maxiter=max_iterations, 
#                 start_params = start_parameters) 
#    else: 
#        mle = MLE_DGP(y, x).fit(method='nm', disp=None, 
#                 maxiter=max_iterations, 
#                 start_params = start_parameters) 

    mle = MLE_DGP(y, x).fit(method='bfgs', disp=None, 
                 maxiter=max_iterations, 
                 start_params = start_parameters) 
#    print(mle.summary())
    
    #Return results
    if additional_parameters>0: #Only return betas from g. 
        betahat = mle.params[:-additional_parameters] 
    else: 
        betahat = mle.params
    return betahat

##############################################################################
### Wrapper, which estimates and extracts results in the preferred format

### Classification (based on logistic error term/squashing function)
def estimator_mle_dgp(data, est_kwargs={}, #data must be dict with form data[x/y][train/test]
                 loglikelihood=logistic_loglikelihood, 
                 loglikelihood_grad=logistic_loglikelihood_grad,                      
                 **kwargs): 
    g_function = est_kwargs['g_function'] #est_kwargs must include relevant g_function. 
    
    from statsmodels.tools.sm_exceptions import HessianInversionWarning
    warnings.filterwarnings("ignore", category=HessianInversionWarning)
    
    # Estimate model 
    betahat = np.array(estimate_mle(data['x']['Train'], data['y']['Train'], g_function, 
                 loglikelihood=loglikelihood, 
                 loglikelihood_grad=loglikelihood_grad))

    #Extract results
    prob, mrgeff = {}, {}
    for split in ('Train', 'Test'):
        g_est = g_function['g_dgp'](x=data['x'][split], beta=betahat, dgp=False)
        prob[split]     = dgp.logit_cdf(g_est)
        mrgeff[split]   = dgp.mrgeff_logit(g_est, g_function['g_dgp_prime'](x=data['x'][split], beta=betahat, dgp=False))

    return betahat, prob, mrgeff

### Classification (based on logistic error term/squashing function)
def estimator_mle_dgp_reg(data, est_kwargs={}, #data must be dict with form data[x/y][train/test]
                 loglikelihood=loglikehood_normal_reg, 
                 loglikelihood_grad=loglikehood_normal_reg_grad,                      
                 **kwargs): 
    g_function = est_kwargs['g_function'] #est_kwargs must include relevant g_function. 
    
    from statsmodels.tools.sm_exceptions import HessianInversionWarning
    warnings.filterwarnings("ignore", category=HessianInversionWarning)
    
    # Estimate model 
    betahat = np.array(estimate_mle(data['x']['Train'], data['y']['Train'], g_function, 
                 loglikelihood=loglikelihood, 
                 loglikelihood_grad=loglikelihood_grad))

    #Extract results
    expect, mrgeff = {}, {}
    for split in ('Train', 'Test'):
        expect[split]   = g_function['g_dgp'](x=data['x'][split], beta=betahat, dgp=False)
        mrgeff[split]   = g_function['g_dgp_prime'](x=data['x'][split], beta=betahat, dgp=False)

    return betahat, expect, mrgeff

##############################################################################
# Non-parametric kernel regression (Nadaraya-Watson)
##############################################################################
##############################################################################
    
#Estimate models 
    
def estimator_nw(data, est_kwargs={}, **kwargs):   
    from statsmodels.nonparametric.kernel_regression import KernelReg
    #http://www.statsmodels.org/dev/generated/statsmodels.nonparametric.kernel_density.EstimatorSettings.html
    from statsmodels.nonparametric.kernel_regression import EstimatorSettings
    k = len(data['x']['Train'].T)
#    n = len(data['x']['Train'])
    
    if 'reg_type' in est_kwargs.keys(): 
        reg_type = est_kwargs['reg_type'] #Allows for locally linear estimation
    else: reg_type = 'lc' #Default is local constant (Nadaraya-Watson). 
    
    #Estimate model
    nw = KernelReg(data['y']['Train'],data['x']['Train'], #Fits regression
                   var_type='c'*k, #Continuous variables
                   reg_type = reg_type, 
                   bw ='aic', #Least-squares cross val. Else aic for aic hurdwidth
                   defaults=EstimatorSettings(n_jobs=1, #No parallel
                                              efficient = True,
                                              randomize=True, #bw estimation random subsampling
                                              n_res = 25, #Number of resamples
                                              n_sub = 50, # Size of samples 
                                              ),
                   )
    betahat = np.array([]) #NP does not have coefficients
                   
    # Extract results
    prob, mrgeff = {}, {} 
    for split in ('Train', 'Test'): 
        prob[split], mrgeff[split] = nw.fit(data_predict=data['x'][split])

    return betahat, prob, mrgeff


##############################################################################
# Semi-parametric kernel regression (Semi-Linear)
##############################################################################
##############################################################################

def estimator_semiparametric_semilinear(data, est_kwargs={}, **kwargs): 
    from statsmodels.sandbox.nonparametric.kernel_extras import SemiLinear
    k = len(data['x']['Train'].T)
    spsl = SemiLinear(endog=data['y']['Train'], 
                      exog=data['x']['Train'], exog_nonparametric=data['x']['Train'],
                  var_type='c'*k,
                  k_linear = k,
                  )
    betahat = spsl.b
    
    prob, mrgeff = {}, {} 
    for split in ('Train', 'Test'): 
        prob[split], mrgeff[split] = spsl.fit(exog_predict=data['x'][split], 
                                              exog_nonparametric_predict= data['x'][split])
        mrgeff[split] += betahat
    
    return betahat, prob, mrgeff


##############################################################################
# Semi-parametric kernel regression (Single Index)
##############################################################################
##############################################################################    

def estimator_semiparametric_singleindex(data, est_kwargs={}, **kwargs): 
    from statsmodels.sandbox.nonparametric.kernel_extras import SingleIndexModel
    k = len(data['x']['Train'].T)
    spsi = SingleIndexModel(endog=data['y']['Train'], 
                      exog=data['x']['Train'],
                  var_type='c'*k,
                  )
    betahat = spsi.b
    
    prob, mrgeff = {}, {} 
    for split in ('Train', 'Test'): 
        prob[split], mrgeff[split] = spsi.fit(data_predict=data['x'][split])
    
    return betahat, prob, mrgeff