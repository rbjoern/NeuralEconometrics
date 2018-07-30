#Preliminaries
#import tensorflow as tf
import numpy as np
import pandas as pd

#Import own files 
import sys 
sys.path.append(r'F:\Documents\TheEnd\Code\Functions')
sys.path.append(r'C:\Users\rbjoe\Dropbox\Kugejl\10.semester\TheEnd\Code\Functions')
import monte_carlo_simulation as mc
import dgp_stuff as dgp
import neural_net as nn
import estimators as est
import summaries as smr
import figurez as figz
import tablez as tblz

#This file has the following sections: 
    # Helper functions 
    # Estimator (classifier)
    # Estimator (regressor)
    # IV estimators
    # Bootstrap 

##############################################################################
# Helper functions
##############################################################################
##############################################################################

# Activation functions
def relu(x):
    a= x.copy()
    a[a<0] = 0
    return a

def relu_prime(x): 
    a = x.copy()
    a[a>0] = 1
    a[a<=0] = 0 # Nondifferentiable zero arbitrarily assigned to zero. 
    return a 

# Functions used to feed forward from coefficients
def unpack_mlp(flat_beta, k, layers=(100,)): 
    #Purpose: Take flattened NN-weights, and restore them to their proper order. 
    #Output: Returns a dict of weighs, with an entry for each layer. 
    previous_layer = np.hstack((k,layers))+1 #+1 adds intercept
    all_layers = np.hstack((layers,1))
    layer_count = previous_layer*all_layers #expected observations 
    start=0
    beta_hat = {}
    for i in range(0,len(layer_count)):
        beta_hat[i] = flat_beta[start:start+layer_count[i]].reshape((previous_layer[i], all_layers[i]))
        start += layer_count[i]
    return beta_hat

# Feed-forward based coefficients produced in above unpacker
def feed_forward_mlp(x,beta_hat, layers=(100,), activation = relu, output=dgp.logit_cdf):
    s,h = {}, {}
    s[0] = est.add_constant(x) 
    h[0] = s[0]
    
    for l in range(1, len(layers)+2):
        s[l] = h[l-1]@beta_hat[l-1]
        if l <len(layers)+1:
            h[l] = est.add_constant(activation(s[l]))
    h[len(layers)+1] = output(s[len(layers)+1])   
    
    return s,h

### How many parameters to expect for a given layer size?
def number_of_coefficients(k, layers=(100,)):
    previous_layer = np.hstack((k,layers))+1 #+1 adds intercept
    all_layers = np.hstack((layers,1))
    coefficients = np.sum(previous_layer*all_layers) 
    return coefficients


##############################################################################
# Estimator (classifier)
##############################################################################
##############################################################################

### Estimate model 
def estimate_mlp(x_train,y_train, x_test={}, 
                 layers=(100,), 
                 activation = 'relu', 
                 alpha=10**(-7), 
                 early_stopping=False, 
                 solver='adam',
                 max_iter=500, 
                 **kwargs): 
    
    # Define estimator 
    from sklearn.neural_network  import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes = layers, 
                        activation = activation, 
                        alpha=alpha, #L2 regularization parameter
                        early_stopping=early_stopping, #DO NOT USE EARLY STOPPING
                        max_iter = max_iter, 
                        **kwargs) 
    
    # Fit data. 
    mlp.fit(x_train,np.ravel(y_train)) 
    
    # Extract coefficients from estimated model, and save them in preferred format. 
    beta_hat = [np.ravel(np.vstack((a,b))) for a, b in zip(mlp.intercepts_, mlp.coefs_)]
    beta_hat = np.concatenate(beta_hat)
    
    #Predict yhat based on fitted model. 
    prob_train = mlp.predict_proba(x_train)[:,1] # Column 1 is prob of 1
    if len(x_test)>0:
        prob_test = mlp.predict_proba(x_test)[:,1] # Column 1 is prob of 1
    if len(x_test)>0:
        return beta_hat, prob_train, prob_test
    else: 
        return beta_hat, prob_train
        
##############################################################################
### Extract results

# Gets marginal effects for arbitrary number of layers
def mrgeff_multlayer(x,beta, k, layers=(100,), activation = relu, activation_prime = relu_prime, 
                     output = dgp.logit_cdf, output_prime=dgp.logit_cdf_prime): 
    beta = unpack_mlp(flat_beta=beta, k=k, layers=layers)
    s, h = feed_forward_mlp(x=x, beta_hat=beta, layers=layers, 
                            activation=activation, output=output)
    
    theta_primes = {}
    for l in range(1,len(layers)+1): 
        theta_primes[l] = activation_prime(s[l]) #theta_primes
    sigma_prime = output_prime(s[len(layers)+1]) #theta prime
      
    mrgeff = np.zeros(np.shape(x))
    
## First
#    for i in range (0, k): 
#         mrg = np.diag(beta[0][i+1, :])
#         for l in range(1,len(layers)+1):
#             if l==1: 
#                mrg = theta_primes[l]@mrg@beta[l][1:]
#             else: 
#                 mrg = theta_primes[l]*mrg@beta[l][1:]
#         mrgeff[:,i] = (sigma_prime*mrg).ravel()    
#    
## Element-wise (slightly simpler )
    #Note: Element-wise multiplication corresponds to multiplying diagonal
    for i in range (0, k): 
         mrg = beta[0][i+1, :]
         for l in range(1,len(layers)+1):
             mrg = theta_primes[l]*mrg@beta[l][1:]
         mrgeff[:,i] = (sigma_prime*mrg).ravel()

#Matrix (gon' get me some diagonals)
    # Matrix nice for one obs, but dimensions mismatch for entire sample. 
    # Inefficient, because loop over observations. No use plz 
#    for i in range (0, k): 
#        for j in range(0,len(sigma_prime)):
#             mrg = beta[0][i+1, :]
#             for l in range(1,len(layers)+1):
#                 mrg = theta_primes[l][j]@np.diag(mrg)@beta[l][1:]
#             mrgeff[j,i] = (sigma_prime[j]*mrg).ravel()

    return mrgeff   

# Predict from coefficients. Now I just predict from probabilities. 
def predict_nn(x,y, beta,k, layers=(100,), activation = relu): 
    beta = unpack_mlp(flat_beta=beta, k=k, layers=layers)
    s, h = feed_forward_mlp(x=x, beta_hat=beta, layers=layers, activation=activation)
    F = h[max(h.keys())]
    y_hat = pd.DataFrame(F>=0.5)
    accuracy = float(np.mean((y_hat==y)))
    return y_hat, accuracy

# Marginal effect for just one layer. I made this one first. I was proud. 
def mrgeff_1layer(x,beta, k, layers=(100,), activation = relu, activation_prime = relu_prime): 
    beta = unpack_mlp(flat_beta=beta, k=k, layers=layers)
    s, h = feed_forward_mlp(x=x, beta_hat=beta, layers=layers, activation=activation)
    
    sigma_prime = dgp.logit_cdf_prime(s[2])
    theta_prime = activation_prime(s[1])
    
    mrgeff = np.zeros(np.shape(x))
    for i in range (0, k): 
        mrgeff[:,i] = sigma_prime*theta_prime@np.diag(beta[0][i+1, :])@beta[1][1:].ravel()
    
    return mrgeff 

##############################################################################
### Wrapper, which does it all
def estimator_nn(data, 
                 est_kwargs={}, mrg_kwargs={}, 
                 splits = ('Train', 'Test'), 
                 **kwargs): #data must be dict with form data[x/y][train/test]
    prob, mrgeff = {}, {}
    
    if 'Test' in splits: 
        betahat, prob['Train'], prob['Test'] \
            = estimate_mlp(x_train=data['x']['Train'], y_train=data['y']['Train'],
                           x_test=data['x']['Test'],
                           **est_kwargs)
    else: 
        betahat, prob['Train'] \
            = estimate_mlp(x_train=data['x']['Train'], y_train=data['y']['Train'],
                           x_test={},
                           **est_kwargs)

    for split in splits:
        mrgeff[split] = mrgeff_multlayer(x=data['x'][split], beta=betahat, k=len(data['x'][split].T), 
                                          output =dgp.logit_cdf, output_prime=dgp.logit_cdf_prime, 
                                          **mrg_kwargs)

    return betahat, prob, mrgeff


##############################################################################
# Estimator (regressor)
##############################################################################
##############################################################################

### Estimate model 
def estimate_mlp_reg(x_train,y_train, x_test={}, 
                 layers=(100,), 
                 activation = 'relu', 
                 alpha=10**(-7), 
                 early_stopping=False, 
                 max_iter=500,
                 solver='adam',
                 **kwargs): 
    
    #Define estimator 
    from sklearn.neural_network  import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes = layers, 
                        activation = activation, 
                        alpha=alpha, #L2 regularization parameter
                        early_stopping=early_stopping, #DO NOT USE EARLY STOPPING
                        max_iter = max_iter, 
                        **kwargs) 
    
    # Fit model
    mlp.fit(x_train,np.ravel(y_train)) 
    
    # Extract coefficient
    beta_hat = [np.ravel(np.vstack((a,b))) for a, b in zip(mlp.intercepts_, mlp.coefs_)]
    beta_hat = np.concatenate(beta_hat)
    
    #Predict yhat based on fitted model. 
    expect_train = mlp.predict(x_train) 
    if len(x_test)>0: #Check if a test set is provied
        expect_test = mlp.predict(x_test) 
    
    if len(x_test)>0:
        return beta_hat, expect_train, expect_test
    else: 
        return beta_hat, expect_train

         

##############################################################################
### Extract results
    # Uses the functions defined for classifier (but with linear output)
    
##############################################################################
### Wrapper, which does it all
def estimator_nn_reg(data, 
                     est_kwargs={}, mrg_kwargs={}, 
                     splits=('Train', 'Test'), 
                     **kwargs): #data must be dict with form data[x/y][train/test]
    expect, mrgeff = {}, {}
    
    if 'Test' in splits: 
        betahat, expect['Train'], expect['Test'] \
            = estimate_mlp_reg(x_train=data['x']['Train'], y_train=data['y']['Train'],
                           x_test=data['x']['Test'],
                           **est_kwargs)
    else: 
        betahat, expect['Train'] \
            = estimate_mlp_reg(x_train=data['x']['Train'], y_train=data['y']['Train'],
                           x_test={},
                           **est_kwargs)

    for split in splits:
        mrgeff[split] = mrgeff_multlayer(x=data['x'][split], beta=betahat, k=len(data['x'][split].T), 
                                          output =dgp.linear_output, output_prime=dgp.linear_output_prime, 
                                          **mrg_kwargs)

    return betahat, expect, mrgeff



##############################################################################
# IV estimators (naive)
##############################################################################
##############################################################################
### First stage 
def estimate_2sls_nn_1st(x_endog, z, x_exog={}, 
                   z_test={}, x_exog_test={}, 
                   regression = True, 
                   est_kwargs={},
                   **kwargs):
    
    if len(x_exog)==0: #Are there no exogenous covariates?
        covariates = z
        if len(z_test)>0:
            covariates_test = z_test
    else: 
        covariates  = np.concatenate((x_exog, z), axis=1) #Also uses exogenous covariates
        if len(z_test)>0:
            covariates_test  = np.concatenate((x_exog_test, z_test), axis=1)
    
    x_hat = np.zeros(np.shape(x_endog))
    if len(z_test)>0:
        x_hat_test = np.zeros((len(z), len(x_endog.T)))                                              
    for p in range(0, len(x_endog.T)): 
        if regression == True: 
            if len(z_test)>0:
                betahat, x_hat[:,p], x_hat_test[:,p] \
                    = estimate_mlp_reg(x_train=covariates, y_train=x_endog.iloc[:,p],
                                   x_test=covariates_test,
                                   **est_kwargs)
            else: 
                betahat, x_hat[:,p] \
                    = estimate_mlp_reg(x_train=covariates, y_train=x_endog.iloc[:,p],
                                   **est_kwargs)
        else: 
            if len(z_test)>0:
                betahat, x_hat[:,p], x_hat_test[:,p] \
                    = estimate_mlp(x_train=covariates, y_train=x_endog.iloc[:,p],
                                   **est_kwargs)
            else: 
                betahat, x_hat[:,p] \
                    = estimate_mlp(x_train=covariates, y_train=x_endog.iloc[:,p],
                                   **est_kwargs)
               
    if len(z_test)>0:
        return x_hat, x_hat_test    
    else: 
        return x_hat    
    
## Second stage 
def estimate_2sls_nn_2nd(y_train, x_hat, x_exog = {},
                         x_hat_test = {}, x_exog_test = {}, 
                         regression=True, est_kwargs={}, mrg_kwargs={},
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
            
    # Second stage 
    expect, mrgeff = {}, {}
    if regression == True: 
        if len(x_hat_test)>0: 
            betahat, expect['Train'], expect['Test'] \
                = estimate_mlp_reg(x_train=covariates['Train'], y_train=y_train,
                               x_test=covariates['Test'],
                               **est_kwargs)
        else: 
            betahat, expect['Train'] \
                = estimate_mlp_reg(x_train=covariates['Train'], y_train=y_train,
                               **est_kwargs)
                
        for split in expect.keys():
            mrgeff[split] = mrgeff_multlayer(x=covariates[split], beta=betahat, k=len(covariates[split].T), 
                                              output =dgp.linear_output, output_prime=dgp.linear_output_prime, 
                                              **mrg_kwargs)
    else: #Binary case
        if len(x_hat_test)>0: 
            betahat, expect['Train'], expect['Test'] \
                = estimate_mlp(x_train=covariates['Train'], y_train=y_train,
                               x_test=covariates['Test'],
                               **est_kwargs)

        else: 
            betahat, expect['Train'] \
                = estimate_mlp(x_train=covariates['Train'], y_train=y_train,
                               **est_kwargs)
                
        for split in expect.keys():
            mrgeff[split] = mrgeff_multlayer(x=covariates[split], beta=betahat, k=len(covariates[split].T), 
                                              output =dgp.logit_cdf, output_prime=dgp.logit_cdf_prime, 
                                              **mrg_kwargs)

    #Return output 
#    if len(x_hat_test)>0:             
    return betahat, expect, mrgeff
#    else: 
#        return betahat, expect['Train'], mrgeff['Train']

##############################################################################
### Control function approach 
    # Instead of using the prediction of the first stage in second stage, use 
    # same x but add residuals from first stage as controls.     
def estimate_2sls_nn_2nd_control(y_train, x_hat, x_endog, x_exog = {},
                         x_hat_test = {}, x_exog_test = {}, x_endog_test={},
                         regression=True, est_kwargs={}, mrg_kwargs={},
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
            
    # Second stage 
    expect, mrgeff = {}, {}
    if regression == True: 
        if len(x_hat_test)>0: 
            betahat, expect['Train'], expect['Test'] \
                = estimate_mlp_reg(x_train=covariates['Train'], y_train=y_train,
                               x_test=covariates['Test'],
                               **est_kwargs)
        else: 
            betahat, expect['Train'] \
                = estimate_mlp_reg(x_train=covariates['Train'], y_train=y_train,
                               **est_kwargs)
                
        for split in expect.keys():
            mrgeff[split] = mrgeff_multlayer(x=covariates[split], beta=betahat, k=len(covariates[split].T), 
                                              output =dgp.linear_output, output_prime=dgp.linear_output_prime, 
                                              **mrg_kwargs)
    else: #Binary case
        if len(x_hat_test)>0: 
            betahat, expect['Train'], expect['Test'] \
                = estimate_mlp(x_train=covariates['Train'], y_train=y_train,
                               x_test=covariates['Test'],
                               **est_kwargs)

        else: 
            betahat, expect['Train'] \
                = estimate_mlp(x_train=covariates['Train'], y_train=y_train,
                               **est_kwargs)
                
        for split in expect.keys():
            mrgeff[split] = mrgeff_multlayer(x=covariates[split], beta=betahat, k=len(covariates[split].T), 
                                              output =dgp.logit_cdf, output_prime=dgp.logit_cdf_prime, 
                                              **mrg_kwargs)

    # Drop control function (coefficients of residuals are not interesting)
    for split in covariates.keys(): 
        if len(x_exog)>0: 
            mrgeff[split] = mrgeff[split][:, 0:(len(x_endog.T)+len(x_exog.T))]
        else: 
            mrgeff[split] = mrgeff[split][:, 0:(len(x_endog.T))]    

    #Return output 
#    if len(x_hat_test)>0:             
    return betahat, expect, mrgeff
#    else: 
#        return betahat, expect['Train'], mrgeff['Train']    
    
##############################################################################
### Estimators (naive approach)
def estimator_2sls_nn(data, est_kwargs={}, mrg_kwargs={},
                      exog={}, est_kwargs_first={}, regression=True,
                      splits = ('Train', 'Test'),
                      **kwargs):
    # Call 2SLS estimator
    # The main point is to specify first and second stage estimation techniques. 
    betahat, expect, mrgeff = est.estimator_2sls(data = data, 
                                             est_kwargs = est_kwargs, est_kwargs_first=est_kwargs_first,
                                             mrg_kwargs=mrg_kwargs, 
                                             exog = exog, 
                                             regression=regression,
                                             estimate_2sls_1st=estimate_2sls_nn_1st, 
                                             estimate_2sls_2nd=estimate_2sls_nn_2nd,
                                             splits=splits,
                                             **kwargs)    
    
    #Return results 
    return betahat, expect, mrgeff


def estimator_2sls_ols_nn(data, est_kwargs={}, mrg_kwargs={},
                      exog={}, est_kwargs_first={}, regression=True,
                      splits = ('Train', 'Test'),
                      **kwargs):
    # Call 2SLS estimator
    betahat, expect, mrgeff = est.estimator_2sls(data = data, 
                                             est_kwargs = est_kwargs, est_kwargs_first=est_kwargs_first,
                                             mrg_kwargs=mrg_kwargs, 
                                             exog = exog, 
                                             regression=regression, constant=True,
                                             estimate_2sls_1st=est.estimate_2sls_1st_ols, 
                                             estimate_2sls_2nd=estimate_2sls_nn_2nd, 
                                             splits=splits,
                                             **kwargs)         

    
    #Return results 
    return betahat, expect, mrgeff


def estimator_2sls_nn_ols(data, est_kwargs={}, mrg_kwargs={},
                      exog={}, est_kwargs_first={}, regression=True,
                      splits = ('Train', 'Test'),
                      **kwargs):
    # Call 2SLS estimator
    betahat, expect, mrgeff = est.estimator_2sls(data = data, 
                                             est_kwargs = est_kwargs, est_kwargs_first=est_kwargs_first,
                                             mrg_kwargs=mrg_kwargs, 
                                             exog = exog, 
                                             regression=regression, constant=True,
                                             estimate_2sls_1st=estimate_2sls_nn_1st, 
                                             estimate_2sls_2nd=est.estimate_2sls_2nd_ols,
                                             splits=splits,
                                             **kwargs)           
    
    #Return results 
    return betahat, expect, mrgeff          

##############################################################################
# Estimators (control function approach)
def estimator_2sls_nn_control(data, est_kwargs={}, mrg_kwargs={},
                      exog={}, est_kwargs_first={}, regression=True,
                      splits = ('Train', 'Test'),
                      **kwargs):
    # Call 2SLS estimator
    # The main point is to specify first and second stage estimation techniques. 
    betahat, expect, mrgeff = est.estimator_2sls(data = data, 
                                             est_kwargs = est_kwargs, est_kwargs_first=est_kwargs_first,
                                             mrg_kwargs=mrg_kwargs, 
                                             exog = exog, 
                                             regression=regression,
                                             estimate_2sls_1st=estimate_2sls_nn_1st, 
                                             estimate_2sls_2nd=estimate_2sls_nn_2nd_control, 
                                             splits = splits,
                                             **kwargs)    
    
    #Return results 
    return betahat, expect, mrgeff

def estimator_2sls_ols_nn_control(data, est_kwargs={}, mrg_kwargs={},
                      exog={}, est_kwargs_first={}, regression=True,
                      splits = ('Train', 'Test'),
                      **kwargs):
    # Call 2SLS estimator
    betahat, expect, mrgeff = est.estimator_2sls(data = data, 
                                             est_kwargs = est_kwargs, est_kwargs_first=est_kwargs_first,
                                             mrg_kwargs=mrg_kwargs, 
                                             exog = exog, 
                                             regression=regression, constant=True,
                                             estimate_2sls_1st=est.estimate_2sls_1st_ols, 
                                             estimate_2sls_2nd=estimate_2sls_nn_2nd_control, 
                                             splits=splits,
                                             **kwargs)         

    
    #Return results 
    return betahat, expect, mrgeff


def estimator_2sls_nn_ols_control(data, est_kwargs={}, mrg_kwargs={},
                      exog={}, est_kwargs_first={}, regression=True,
                      splits = ('Train', 'Test'),
                      **kwargs):
    # Call 2SLS estimator
    betahat, expect, mrgeff = est.estimator_2sls(data = data, 
                                             est_kwargs = est_kwargs, est_kwargs_first=est_kwargs_first,
                                             mrg_kwargs=mrg_kwargs, 
                                             exog = exog, 
                                             regression=regression, constant=True,
                                             estimate_2sls_1st=estimate_2sls_nn_1st, 
                                             estimate_2sls_2nd=est.estimate_2sls_2nd_ols_control, 
                                             splits=splits,
                                             **kwargs)           
    
    #Return results 
    return betahat, expect, mrgeff 


##############################################################################
# Boostrap
##############################################################################
##############################################################################

# Sample data with replacement 
def bootstrap_sample_single(y,x):
    #Inspiration: https://gist.github.com/aflaxman/6871948
    n = len(y)
    
    boot_sample = np.random.randint(low=0, high=n, size=n)
    x_boot = x[boot_sample]
    y_boot = y[boot_sample]
    
    
    return y_boot, x_boot, bootstrap_sample

def bootstrap_sample(data, splits=('Train','Test')):
    #Inspiration: https://gist.github.com/aflaxman/6871948
    
    data_boot = {}
    data_boot['x'], data_boot['y'] = {},{}
    if 'z' in data.keys(): data_boot['z'] = {}
    
    for split in splits:
        n = len(data['y'][split])
        boot_sample = np.random.randint(low=0, high=n, size=n)
        data_boot['x'][split] =  data['x'][split].iloc[boot_sample, :]     
        data_boot['y'][split] =  data['y'][split].iloc[boot_sample] 
        if 'z' in data_boot.keys(): 
            data_boot['z'][split] =  data['z'][split].iloc[boot_sample] 
    
    return data_boot, boot_sample

def bootstrap_estimator(estimator, data, 
                        B=100, splits=('Train', 'Test'), exog={},
                        get_averages=False, 
                        ): 
        # Estimators are of the format above
        # Data is of the format used (dict with train/test)
#    n = len(data['y']['Test'])
    boot_expects, boot_mrgeff = {}, {}
    
    for b in range (0,B):
        #Draw bootstrap sample
        data_boot, boot_sample = bootstrap_sample(data, splits=splits)
        
        # Retrain models and recalculate statistics of interest
        betahat, expect, mrgeff \
        = estimator['estimator'](data_boot, est_kwargs = estimator['est_kwargs'],
                                            mrg_kwargs=estimator['mrg_kwargs'], 
                                            splits=splits, exog=exog) 
        
        # Save statistics for included observations 
        expect_boot, mrgeff_boot = {}, {}
        for split in splits:     
            n = len(data['y'][split])
#            print(type(np.array(expect[split])))
#            print(1/0)
            expect_boot[split] = pd.DataFrame(np.array(expect[split]), index=boot_sample) #Create df with index
        #    expect_boot[split] = expect_boot[split].sort_index() # Sort it
        #    expect_boot[split] = expect_boot[split].drop_duplicates(keep='first') #Drop duplicates
            expect_boot[split] = expect_boot[split][~expect_boot[split].index.duplicated(keep='first')] #Drop duplicates
            expect_boot[split] = expect_boot[split].reindex(index=pd.Index(range(0,n))) #Insert empty observations
            
            mrgeff_boot[split] = pd.DataFrame(np.array(mrgeff[split]), index=boot_sample) #Create df with index
            mrgeff_boot[split] = mrgeff_boot[split][~mrgeff_boot[split].index.duplicated(keep='first')] #Drop duplicates
            mrgeff_boot[split] = mrgeff_boot[split].reindex(index=pd.Index(range(0,n))) #Insert empty observations
    
        #Store results 
        if b==0: #If first iteration, create storage units
            for split in splits:
                boot_expects[split] = np.array(expect_boot[split])
                boot_mrgeff[split] = np.expand_dims(mrgeff_boot[split], axis=1)
        else: # Later on, just fill out
            for split in splits:
                boot_expects[split] = np.concatenate((boot_expects[split], expect_boot[split]), axis=1)
                boot_mrgeff[split] = np.concatenate((boot_mrgeff[split], np.expand_dims(mrgeff_boot[split], axis=1)), axis=1)
        
    if get_averages == True: #Calculate averages within each bootstrap
        for split in splits:
            boot_expects[split] = np.nanmean(boot_expects[split], axis=0)
            boot_mrgeff[split] = np.nanmean(boot_mrgeff[split], axis=0)            
        
    return boot_expects, boot_mrgeff
    