
# IMPORT DEPENDENCIES
import numpy as np
import pandas as pd
from datetime import datetime 
import pickle
import os 
import warnings
import multiprocessing as mp
from functools import partial 
from itertools import repeat


#Import own files 
import sys 
sys.path.append(r'F:\Documents\TheEnd\Code\Functions')
sys.path.append(r'C:\Users\rbjoe\Dropbox\Kugejl\10.semester\TheEnd\Code\Functions')
import dgp_stuff as dgp
import neural_net as nn
import estimators as est

#This file has the following sections: 
    ### Basic simulation
    ### Wrappers for running multiple simulations
    ### Loaders (gets simulation results from files)

##########################################################################
##########################################################################
##########################################################################
### Basic simulation

### Define one iteration of the simulation
def MC_iteration(i_m, #iterated for simulations
                parameters, #Parameter dictionary. Defined below
                estimators, #Dictionary of models to estimate
                g_function, #Dictionary with DGP g function
                **kwargs): #Mainly to allow for odd arguments passed and ignored
    
    ##########################################################################
    #2.1. Preliminaries
    np.random.seed(parameters['seed']+i_m) #Needs to be explicitly different when parallel
    
    ##########################################################################
    #2.2.Generate data from dgp
    data, betahat, temp         = {}, {}, {} #temp stores data not returned by function. 
    if parameters['redraw']==True:  #Re-draw betas for every iteration
            betahat['DGP'] = parameters['beta_distribution'](k=g_function['g_parameters'](parameters['k']+parameters['C']), 
                               mu = parameters['beta_mean'], stdev = parameters['beta_scale']) 
    else:   betahat['DGP'] = parameters['beta_dgp'] #If no redraw, use the one drawn above
    
    # Draw data from distributions
    data['x'], data['y'], temp['g'], temp['u'] = {}, {}, {}, {}
    if parameters['Z'] > 0: data['z'] = {}
    if parameters['V'] > 0: temp['v'] = {}
    
    for split in ('Train', 'Test'):
        if parameters['Z'] == 0: #Default: 
            data['x'][split] = parameters['x_distribution'](mu = parameters['xmu'], cov=parameters['xcov'],n=parameters['n'])            
        elif parameters['Z'] > 0: #Special case: IV estimation splits into instruments (z) and regressors (x)
            data['x'][split] = parameters['x_distribution'](mu = parameters['xmu'], cov=parameters['xcov'],n=parameters['n'], 
                                            k = parameters['k']+parameters['V'], c = parameters['C'], z=parameters['Z'], 
                                            g_function = g_function)            
            data['z'][split] = data['x'][split].iloc[:, parameters['k']+parameters['V']+parameters['C']:] #Split out instruments
            data['x'][split] = data['x'][split].iloc[:, :parameters['k']+parameters['V']+parameters['C']] #Data is what remains

        if parameters['V']>0: #Remove irrelevant variables before computing true properties
            temp['v'][split] = data['x'][split].iloc[:, parameters['k']:parameters['k']+parameters['V']]   #Middle is irrelevant
            data['x'][split] = data['x'][split].iloc[:, np.r_[0:parameters['k'],-parameters['C']:0]]  # Regressors and confounders in the ends
        
        temp['u'][split]     = parameters['u_distribution'](n=parameters['n'], stdev=parameters['u_scale'])
        
        
    
    #Compute variables based on drawws 
    for split in ('Train', 'Test'):
        temp['g'][split]     = g_function['g_dgp'](x= data['x'][split], beta=betahat['DGP'])
        data['y'][split]     = parameters['y_generate'](g=temp['g'][split], u=temp['u'][split])
    
    
    ##########################################################################
    #2.3. Compute true properties. 
    mrgeff, prob = {}, {}
    prob['DGP'] = {}
    for split in ('Train', 'Test'):
        prob['DGP'][split]   = parameters['y_squashing'](temp['g'][split])
    
    temp['g_prime'], mrgeff['DGP'] = {}, {}
    for split in ('Train', 'Test'):
        temp['g_prime'][split]    = g_function['g_dgp_prime'](x=data['x'][split], beta=betahat['DGP'])
        mrgeff['DGP'][split]        = np.array(dgp.mrgeff_dgp(g=temp['g'][split], g_prime=temp['g_prime'][split], 
                                                      y_squashing_prime=parameters['y_squashing_prime']))
    
    ##########################################################################
    #2.4. Estimate models (well-specified /true data)
    # Run this section if 1) observable case is wellspecified or 2) we want both wellspecified and observables
    if parameters['run_wellspecified'] == True: 
        #Estimate models
        for estimator in estimators: 
                betahat[estimator], prob[estimator], mrgeff[estimator] \
                = estimators[estimator]['estimator'](data, est_kwargs= estimators[estimator]['est_kwargs'],
                                                               mrg_kwargs=estimators[estimator]['mrg_kwargs'])      
        
        #Bootstrap results (optional)
        if parameters['B'] > 0:  #If no replications, bootstrap is not desired 
            boot_expect, boot_mrgeff = {}, {}
            for estimator in estimators: 
#                if 'bootstrapper' in estimators[estimator].keys():
                boot_expect[estimator], boot_mrgeff[estimator] \
                    = nn.bootstrap_estimator(estimator=estimators[estimator],
                                             data=data, B=parameters['B'], 
                                             get_averages=parameters['bootstrap_averages'])
            

    ###########################################################################
    # 2.5. Restrict to observable data (optional)
    # Prepare storage for observables 
    if parameters['run_observables'] == True:     
        betahat_obs, mrgeff_obs, prob_obs = {}, {}, {}
        betahat_obs['DGP'] = betahat['DGP'].copy() # No differences for DGP. 
        mrgeff_obs['DGP'] = mrgeff['DGP'].copy()
        prob_obs['DGP'] = prob['DGP'].copy()
    
    # Remove confounders
    if parameters['C'] > 0:
        for split in ('Train', 'Test'):
            data['x'][split] = data['x'][split].iloc[:,0:parameters['k']]
            mrgeff_obs['DGP'][split] =  mrgeff_obs['DGP'][split][:,0:parameters['k']]
    
    # Reinsert irrelevant variables
    if parameters['V']>0:
        for split in ('Train', 'Test'):
            data['x'][split] = pd.concat([data['x'][split], temp['v'][split]], axis=1)
            #Marginal effect of irrelevant is zero
            mrgeff_obs['DGP'][split] = np.concatenate((mrgeff_obs['DGP'][split], np.zeros(np.shape(temp['v'][split]))), axis=1)
    
    # Add measurement error 
    if parameters['add_error'] == True: 
        for split in ('Train', 'Test'):
            #Draw measurement error 
            error = parameters['error_distribution'](mu = parameters['error_mu'], 
                                                    cov=parameters['error_cov'],n=parameters['n'])            
            # Add it to observable data 
            data['x'][split] = data['x'][split] + error 
        
    ##########################################################################
    #2.6. Estimate models (actual / observable data)                    
    if parameters['run_observables'] == True: 
        #Estimate models
        for estimator in estimators: 
                betahat_obs[estimator], prob_obs[estimator], mrgeff_obs[estimator] \
                = estimators[estimator]['estimator'](data, est_kwargs= estimators[estimator]['est_kwargs'],
                                                               mrg_kwargs=estimators[estimator]['mrg_kwargs'])      
        
        #Bootstrap results (optional)
        if parameters['B'] > 0:  #If no replications, bootstrap is not desired 
            boot_expect_obs, boot_mrgeff_obs = {}, {}
            for estimator in estimators: 
#                if 'bootstrapper' in estimators[estimator].keys():
                boot_expect_obs[estimator], boot_mrgeff_obs[estimator] \
                    = nn.bootstrap_estimator(estimator=estimators[estimator],
                                             data=data, B=parameters['B'], 
                                             get_averages=parameters['bootstrap_averages'])        

        # If wellspecified case wasnt run, just replace it and return standards
        if parameters['run_wellspecified'] == False: 
            betahat, mrgeff, prob = betahat_obs, mrgeff_obs, prob_obs
            if parameters['B'] > 0:
                boot_expect, boot_mrgeff = boot_expect_obs, boot_mrgeff_obs
    
    ##########################################################################
    #2.7. Finalize 
    #Reduce size 
    if parameters['reduce_size'] == True: 
        for split in ('Train', 'Test'):
            for var in ('x', 'y'):
                data[var][split] = data[var][split].astype(np.float16)
        
        for model in ['DGP'] + [estimator for estimator in estimators]: 
            betahat[model] = betahat[model].astype(np.float16)
            for split in ('Train', 'Test'):
                mrgeff[model][split] = mrgeff[model][split].astype(np.float16)
                prob[model][split] = prob[model][split].astype(np.float16)
                if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
                    mrgeff_obs[model][split] = mrgeff_obs[model][split].astype(np.float16)
                    prob_obs[model][split] = prob_obs[model][split].astype(np.float16)
        
        if parameters['B'] > 0:  # Bootstrap is optional
            for model in boot_expect.keys(): 
                for split in ('Train', 'Test'):
                   boot_expect[model][split] = boot_expect[model][split].astype(np.float16)
                   boot_mrgeff[model][split] = boot_mrgeff[model][split].astype(np.float16)
                   if parameters['run_wellspecified'] == True and parameters['run_observables'] == True: 
                       boot_expect_obs[model][split] = boot_expect_obs[model][split].astype(np.float16)
                       boot_mrgeff_obs[model][split] = boot_mrgeff_obs[model][split].astype(np.float16)
    
    ##########################################################################
    #2.8.  Return output
    if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
        return betahat, mrgeff, prob, data, betahat_obs, mrgeff_obs, prob_obs 
    elif parameters['B'] > 0:
        return betahat, mrgeff, prob, data, boot_expect, boot_mrgeff
    else: # Default: Most basic outcome 
        return betahat, mrgeff, prob, data

##############################################################################
### Run a single set of M simulations    
def MC_simulate(parameters={}, #Parameter dictionary. Defined below
                estimators={}, #Dictionary of models to estimate
                g_function={}, #Dictionary with DGP g function
                **kwargs): #Mainly to allow for odd arguments passed and ignored
    
    ###########################################################################
    #1. PRELIMINARIES 
    
    ##########################################################################
    #Set default parameters 
    #General simulation set-up
    if 'seed' not in parameters.keys(): parameters['seed'] = 33 # Seed for pseudo-random draws  
    if 'M' not in parameters.keys(): parameters['M'] = 10       # Simulation repetitions
    if 'n' not in parameters.keys(): parameters['n'] = 10**4    # Observations (sample)
    if 'k' not in parameters.keys(): parameters['k'] = 2        # Regressors (features)
    if 'V' not in parameters.keys(): parameters['V'] = 0        # Irrelevant regressors
    if 'C' not in parameters.keys(): parameters['C'] = 0        # Confounding variables
    if 'Z' not in parameters.keys(): parameters['Z'] = 0        # Instruments 
    if 'B' not in parameters.keys(): parameters['B'] = 0        # Bootstrap replications
    if 'add_error' not in parameters.keys(): parameters['add_error'] = False # Measurement error on regressors
    
    #Technical stuff
    if 'parallel' not in parameters.keys(): parameters['parallel']       = False        # Simulations run parallel or serial   
    if 'reduce_size' not in parameters.keys(): parameters['reduce_size'] = False        # Saves simulations as 16 insted of 64 
    if 'save_file' not in parameters.keys(): parameters['save_file']     = False        # Save results in files 
    if 'filename' not in parameters.keys(): parameters['filename']       = 'V1'        
    if 'decimals' not in parameters.keys(): parameters['decimals']       = 2            # Decimals (supposedly) used for figures
    if 'start_time' not in parameters.keys(): parameters['start_time']   = datetime.now() # Time measure 
    if 'bootstrap_averages' not in parameters.keys(): parameters['bootstrap_averages'] = True #Only get averages from bootstrap
    
    #How to draw data
    if 'beta_distribution' not in parameters.keys(): parameters['beta_distribution'] = dgp.draw_beta_normal    #Draw betas from this distribution  
    if 'beta_mean' not in parameters.keys(): parameters['beta_mean'] = 1                                       #Mean and scale for beta distribution
    if 'beta_scale' not in parameters.keys(): parameters['beta_scale'] = 1
    if 'redraw' not in parameters.keys(): parameters['redraw'] = False                                         #Redraw beta in each simulation? 
    if 'x_distribution' not in parameters.keys(): parameters['x_distribution'] = dgp.draw_x_normal             #Draw x from this distribution
    if 'x_distribution_parameters' not in parameters.keys(): parameters['x_distribution_parameters'] = dgp.gen_x_normal_unitvariance_randommean #Draw mean and correlation for x 
    if 'x_mean' not in parameters.keys(): parameters['x_mean'] = 0                                             #Average expected value of x variables
    if 'u_distribution' not in parameters.keys(): parameters['u_distribution'] = dgp.draw_u_logit              #Define error term for y.       
    if 'u_scale' not in parameters.keys(): parameters['u_scale'] = 1                                           # Scale for error term
    if 'y_generate' not in parameters.keys(): parameters['y_generate'] = dgp.gen_y_latent                      #Compute y from errorn and g
    if 'y_squashing' not in parameters.keys(): parameters['y_squashing'] = dgp.logit_cdf                       # Squashing
    if 'y_squashing_prime' not in parameters.keys(): parameters['y_squashing_prime'] =  dgp.logit_cdf_prime    # Squashing outer derivative 

    
    # Sometimes data is restricted from that used in DGP true vs. observable data)
    if parameters['V'] > 0 or parameters['C'] > 0 or parameters ['add_error']==True:
        if 'run_wellspecified' not in parameters.keys():  # Default: One case (observables)
            parameters['run_observables'], parameters['run_wellspecified'] = True, False   
        else: parameters['run_observables'] = True #Well-specified specified in parameters 
    else: #Default: One case (wellspecified, since observable data is the same )
        parameters['run_observables'], parameters['run_wellspecified'] = False, True 
    
    # Exception: Bootstrap only implemented for estimating one set of models for simplicity in returning output. 
    if parameters['B'] > 0 and parameters['run_observables'] == True and parameters['run_wellspecified'] == True: 
        raise Exception('Bootstrap only implemented for estimating single set of models')
    
   #If no g_function is provided, run basic linear case
    if g_function == {}: 
      g_function = {'g_name': 'Linear', 'g_dgp': dgp.g_logit, 'g_dgp_prime': dgp.g_logit_prime, 'g_parameters': parameters['k']}
    
    # If no estimators are provided, just run logistic and neural net
    if estimators == {}: 
        estimators['Logit'] = {'name': 'Logit',     'estimator': est.estimator_logit, 'est_kwargs':{}, 'mrg_kwargs':{}}
        estimators['NN']    = {'name': 'NN',        'estimator': nn.estimator_nn, 'est_kwargs':{}, 'mrg_kwargs':{}}
    
    #Control randomness 
    np.random.seed(parameters['seed']) #Set random seed
    
    if parameters['redraw']==False:  #Draw DGP once, and don't change it. 
        parameters['beta_dgp'] = parameters['beta_distribution'](k=g_function['g_parameters']\
                                  (parameters['k']+parameters['C']), #Generate parameters for DGP
                                   mu = parameters['beta_mean'], stdev = parameters['beta_scale']) 
    #Draw distribution parameters for x. 
    if parameters['Z'] == 0: #Default
        parameters['xmu'], parameters['xcov'] = parameters['x_distribution_parameters']\
                                                (k=parameters['k']+parameters['V']+parameters['C'], 
                                                 mean=parameters['x_mean'])
    else: # IV draws covariance matrix which follows exclusion restriction. 
        cov_max_tries = 100    
        for i in range(0,cov_max_tries): #Cov matrix is sometimes not pos. def. when we impose no corr. Try until we get one right.     
            parameters['xmu'], parameters['xcov'] = parameters['x_distribution_parameters']\
                                                    (k=parameters['k']+parameters['V'], 
                                                     c=parameters['C'], z=parameters['Z'],
                                                     mean=parameters['x_mean'])
            #Check if cov matrix is positive semidef
            
            try: 
                np.linalg.cholesky(parameters['xcov']) #Cholesky decomposition fails if not pos def
                break #Stop loop if it works
            except Exception: #Technically a LinAlgError, but would need to be imported 
                pass #Continue loop until we hopefully finds something. 
                if i == cov_max_tries-1: 
                    raise Exception('No semi-positive definite covariance matrix found. Try different seed.')
    
    if parameters['add_error'] == True: # Optional measurement error 
        if 'error_distribution_parameters' not in parameters.keys(): 
                parameters['error_distribution_parameters'] = dgp.gen_error_normal
        if 'error_scale' not in parameters.keys(): parameters['error_scale'] = 1
        if 'error_distribution' not in parameters.keys(): parameters['error_distribution'] = dgp.draw_x_normal
        
        parameters['error_mu'], parameters['error_cov'] = parameters['error_distribution_parameters']\
                                                            (k=parameters['k']+parameters['V'], scale=parameters['error_scale'])
        
    warnings.filterwarnings("ignore", category=FutureWarning) #Statsmodel uses deprecated features of pandas, which yields warnings. This is not problematic, as you can't update the packages anyways. If packages are updated, this line should be removed  
    # Only for testing - Should possibly be disabled in final version, where convergence should work. 
    from statsmodels.tools.sm_exceptions import ConvergenceWarning 
    warnings.filterwarnings("ignore", category=ConvergenceWarning)   
    from sklearn.exceptions import ConvergenceWarning 
    warnings.filterwarnings("ignore", category=ConvergenceWarning) 
    
    ###########################################################################
    #2. RUN SIMULATION
    iteration_keywords =    {'parameters': parameters,
                             'estimators':estimators, 
                             'g_function': g_function,
                             }
    if parameters['parallel'] == False: 
        #results = [MC_iteration(i, **iteration_keywords) for i in range(0,M)]
        results = list(map(partial(MC_iteration, **iteration_keywords), range(0,parameters['M'])))
        #list(map(MC_iteration, range(0,M))) #Runs function just like a for loop
        
    else: #Runs the simulations in parallel for each cpu on the computer
        pool = mp.Pool(processes = min(mp.cpu_count(),10))
        results = pool.map(partial(MC_iteration, **iteration_keywords), range(0,parameters['M']))

    ###########################################################################
    #3. UNPACK RESULTS
    betahats, mrgeffs, probs, data      = {}, {}, {}, {}
    for model in ['DGP'] + [estimator for estimator in estimators]: 
        betahats[model]                 = [result[0][model] for result in results] 
        probs[model], mrgeffs[model]    = {}, {}
        for split in ('Train', 'Test'):
            mrgeffs[model][split]       = [result[1][model][split] for result in results] 
            probs[model][split]         = [result[2][model][split] for result in results]
    for key in results[0][3].keys(): 
        data[key] = {}
        for split in ('Train','Test'):    
            data[key][split]            = [result[3][key][split] for result in results]
    
    if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
        betahats_obs, mrgeffs_obs, probs_obs = {}, {}, {}
        for model in ['DGP'] + [estimator for estimator in estimators]: 
            betahats_obs[model]                 = [result[4][model] for result in results] 
            probs_obs[model], mrgeffs_obs[model]    = {}, {}
            for split in ('Train', 'Test'):
                mrgeffs_obs[model][split]       = [result[5][model][split] for result in results] 
                probs_obs[model][split]         = [result[6][model][split] for result in results]
            
    if parameters['B'] > 0:
        boot_expects, boot_mrgeffs = {}, {}
        for model in results[0][4].keys(): 
            boot_expects[model], boot_mrgeffs[model] = {},{},
            for split in ('Train','Test'):    
                boot_expects[model][split]     = [result[4][model][split] for result in results]
                boot_mrgeffs[model][split]     = [result[5][model][split] for result in results]
    
    ###########################################################################
    # 4. SAVE RESULTS (POSSIBLY)
    if parameters['save_file']==True: 
        outputs = {'res_betahats': betahats, 'res_mrgeffs': mrgeffs, 'data': data, 'res_probs':probs, 
                   'parameters': parameters, 'estimators': estimators, 'g_function':g_function}
        
        if parameters['B'] > 0:
                outputs.update({'res_boot_expects': boot_expects, 'res_boot_mrgeffs': boot_mrgeffs})
        
        if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
                outputs.update({'res_betahats_obs': betahats_obs, 'res_mrgeffs_obs': mrgeffs_obs, 
                                'res_probs_obs':probs_obs})
        for output in outputs: 
            with open(os.getcwd() + '\\simulation_results\\'+'%s_%s.txt' % (parameters['filename'], output), "wb") as f: 
                pickle.dump(outputs[output], f)
    
    #5. RETURN OUTPUT
    if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
        return betahats, mrgeffs, data, probs, betahats_obs, mrgeffs_obs, probs_obs 
    elif parameters['B'] > 0:
        return betahats, mrgeffs, data, probs, boot_expects, boot_mrgeffs 
    else: 
        return betahats, mrgeffs, data, probs


##########################################################################
##########################################################################
##########################################################################
### WRAPPERS FOR RUNNING MULTIPLE SIMULATIONS
# These functions repeat the simulations for various data setups and parameters. 

##########################################################################
### V2: Run simulation for various DGP's (specified through g function, see paper)
def MC_simulate_dgps(parameters, estimators, g_functions, 
                     save_file=False, filename='V2'): 
    ##########################################################################
    # Preliminaries 
    if 'save_file' in parameters.keys(): 
        save_file = parameters['save_file']
    parameters['save_file'] = False # Don't save files in subfunctions
    
    if 'filename' in parameters.keys(): 
        filename = parameters['filename']
    else: filename = 'V2'
    
    if 'B' not in parameters.keys(): parameters['B'] = 0 #No bootstrap as default

    if parameters['V'] > 0 or parameters['C'] > 0 or parameters ['add_error']==True:
        if 'run_wellspecified' not in parameters.keys():  # Default: One case (observables)
            parameters['run_observables'], parameters['run_wellspecified'] = True, False   
        else: parameters['run_observables'] = True #Well-specified specified in parameters 
    else: #Default: One case (wellspecified, since observable data is the same )
        parameters['run_observables'], parameters['run_wellspecified'] = False, True     
    
    #Prepare storage 
    res_betahats, res_mrgeffs, data, res_probs = {},{},{},{}
    if parameters['B']>0: res_boot_expects, res_boot_mrgeffs = {},{}
    if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
        res_betahats_obs, res_mrgeffs_obs, res_probs_obs = {},{},{}
    
    ##########################################################################
    # Loop over dgp functions    
    for g_function in g_functions.keys(): 
        print('Runtime:', datetime.now() - parameters['start_time'], '\tStatus: Began simulation for', g_functions[g_function]['g_name'])
        
        #Update estimators to fit with the given g(x,beta)
        if 'MLE' in estimators.keys(): #MLE uses g(x,beta)
            estimators['MLE']['est_kwargs'].update({'g_function': g_functions[g_function]})
        
        if 'g_hyper_nn' in  g_functions[g_function].keys():  #Allows for various layer sizes etc based on g(x,beta) complexity
            if 'NN (I)' in estimators: 
                estimators['NN (I)']['est_kwargs'].update(g_functions[g_function]['g_hyper_nn'])
                estimators['NN (I)']['mrg_kwargs'].update(g_functions[g_function]['g_hyper_nn'])
            if 'NN (II)' in estimators: 
                estimators['NN (II)']['est_kwargs'].update(g_functions[g_function]['g_hyper_nn'])
                estimators['NN (II)']['mrg_kwargs'].update(g_functions[g_function]['g_hyper_nn'])
                #2layer net simply replicates first layer again, increasing representational capacity
                estimators['NN (II)']['est_kwargs']['layers'] = 2*estimators['NN (II)']['est_kwargs']['layers'] #2 layers
                estimators['NN (II)']['mrg_kwargs']['layers'] = 2*estimators['NN (II)']['mrg_kwargs']['layers'] #2 layers
        
        #Run basic simulation
        if parameters['B'] == 0: #Standard (no bootstrap)
            if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
                res_betahats[g_function], res_mrgeffs[g_function], data[g_function], res_probs[g_function], \
                res_betahats_obs[g_function], res_mrgeffs_obs[g_function], res_probs_obs[g_function]= \
                    MC_simulate(parameters, estimators, g_functions[g_function])
            else: 
                res_betahats[g_function], res_mrgeffs[g_function], data[g_function], res_probs[g_function] = \
                    MC_simulate(parameters, estimators, g_functions[g_function])
        else: #Bootstrap
            res_betahats[g_function], res_mrgeffs[g_function], data[g_function], res_probs[g_function], \
                res_boot_expects[g_function], res_boot_mrgeffs[g_function] = \
                    MC_simulate(parameters, estimators, g_functions[g_function])         
            
        #print('Runtime:', datetime.now() - parameters['start_time'], '\tStatus: Finished the simulation for', g_functions[g_function]['g_name'])
#    print('Runtime:', datetime.now() - parameters['start_time'], '\tStatus: Finished simulations for DGP\'s.')
    
    
    ##########################################################################
    # Save output (possibly)
    if save_file==True: 
        outputs = {'res_betahats': res_betahats, 'res_mrgeffs': res_mrgeffs, 'data': data, 'res_probs':res_probs, 
                   'parameters': parameters, 'estimators': estimators, 'g_functions':g_functions}
        
        if parameters['B'] > 0:
                outputs.update({'res_boot_expects': res_boot_expects, 'res_boot_mrgeffs': res_boot_mrgeffs})
        
        if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
                outputs.update({'res_betahats_obs': res_betahats_obs, 'res_mrgeffs_obs': res_mrgeffs_obs, 
                                'res_probs_obs':res_probs_obs})
        for output in outputs: 
            with open(os.getcwd() + '\\simulation_results\\'+'%s_%s.txt' % (filename, output), "wb") as f: 
                pickle.dump(outputs[output], f)
    
    ##########################################################################
    # Return output    
    if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
        return res_betahats, res_mrgeffs, data, res_probs, res_betahats_obs, res_mrgeffs_obs, res_probs_obs
    elif parameters['B'] > 0:
        return res_betahats, res_mrgeffs, data, res_probs, res_boot_expects,  res_boot_mrgeffs
    else: 
        return res_betahats, res_mrgeffs, data, res_probs

##########################################################################
### V2_2: Run simulation for various DGP's (specified through g function, see paper)
        # Saves each iteration by itself, to prevent memory issues. 
def MC_simulate_dgps_indfiles(parameters, estimators, g_functions, 
                     save_file=False, filename='V2'): 
    ##########################################################################
    # Preliminaries 
#    res_betahats, res_mrgeffs, data, res_probs = {},{},{},{}
    if 'save_file' in parameters.keys(): 
        save_file = parameters['save_file']
    parameters['save_file'] = False # Don't save files in subfunctions
    
    if 'filename' in parameters.keys(): 
        filename = parameters['filename']
    else: filename = 'V2'
    
    if 'B' not in parameters.keys(): parameters['B'] = 0 #No bootstrap as default
    
    if parameters['V'] > 0 or parameters['C'] > 0 or parameters ['add_error']==True:
        if 'run_wellspecified' not in parameters.keys():  # Default: One case (observables)
            parameters['run_observables'], parameters['run_wellspecified'] = True, False   
        else: parameters['run_observables'] = True #Well-specified specified in parameters 
    else: #Default: One case (wellspecified, since observable data is the same )
        parameters['run_observables'], parameters['run_wellspecified'] = False, True      
    
    if save_file == True: #No reason to resave parameters if looping through parseries 
        #Store parameters (once, rather than doing it for each parameter)
        outputs = {'parameters': parameters, 'estimators': estimators, 'g_functions':g_functions}
        for output in outputs: 
            with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_pars.txt' % (filename, output), "wb") as f: 
                pickle.dump(outputs[output], f)
    
    ##########################################################################
    # Loop over dgp functions    
    for g_function in g_functions.keys(): 
        print('Runtime:', datetime.now() - parameters['start_time'], '\tStatus: Began simulation for', g_functions[g_function]['g_name'])
        
        #Update estimators to fit with the given g(x,beta)
        if 'MLE' in estimators.keys(): #MLE uses g(x,beta)
            estimators['MLE']['est_kwargs'].update({'g_function': g_functions[g_function]})
        
        if 'g_hyper_nn' in  g_functions[g_function].keys():  #Allows for various layer sizes etc based on g(x,beta) complexity
            if 'NN (I)' in estimators: 
                estimators['NN (I)']['est_kwargs'].update(g_functions[g_function]['g_hyper_nn'])
                estimators['NN (I)']['mrg_kwargs'].update(g_functions[g_function]['g_hyper_nn'])
            if 'NN (II)' in estimators: 
                estimators['NN (II)']['est_kwargs'].update(g_functions[g_function]['g_hyper_nn'])
                estimators['NN (II)']['mrg_kwargs'].update(g_functions[g_function]['g_hyper_nn'])
                #2layer net simply replicates first layer again, increasing representational capacity
                estimators['NN (II)']['est_kwargs']['layers'] = 2*estimators['NN (II)']['est_kwargs']['layers'] #2 layers
                estimators['NN (II)']['mrg_kwargs']['layers'] = 2*estimators['NN (II)']['mrg_kwargs']['layers'] #2 layers
        
        
        #Run basic simulation
        if parameters['B'] == 0: #Standard (no bootstrap)
            if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
                res_betahats, res_mrgeffs, data, res_probs, \
                res_betahats_obs, res_mrgeffs_obs, res_probs_obs = \
                    MC_simulate(parameters, estimators, g_functions[g_function])    
            else: 
                res_betahats, res_mrgeffs, data, res_probs = \
                    MC_simulate(parameters, estimators, g_functions[g_function]) 
        else: #Bootstrap
            res_betahats, res_mrgeffs, data, res_probs, res_boot_expects, res_boot_mrgeffs = \
                MC_simulate(parameters, estimators, g_functions[g_function]) 
        
        ##########################################################################
        #Save stage files
        outputs = {'res_betahats': res_betahats, 'res_mrgeffs': res_mrgeffs, 'data': data, 'res_probs':res_probs}

        if parameters['B'] > 0:
                outputs.update({'res_boot_expects': res_boot_expects, 'res_boot_mrgeffs': res_boot_mrgeffs})

        if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
                outputs.update({'res_betahats_obs': res_betahats_obs, 'res_mrgeffs_obs': res_mrgeffs_obs, 
                                'res_probs_obs':res_probs_obs})        
        for output in outputs: 
            with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_%s.txt' % (filename, g_function, output), "wb") as f: 
                pickle.dump(outputs[output], f)
    
    ##########################################################################
    # Return output
    if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
        return res_betahats, res_mrgeffs, data, res_probs, res_betahats_obs, res_mrgeffs_obs, res_probs_obs
    elif parameters['B'] > 0:
        return res_betahats, res_mrgeffs, data, res_probs, res_boot_expects,  res_boot_mrgeffs
    else: 
        return res_betahats, res_mrgeffs, data, res_probs

##########################################################################
### V3: Run simulation for various parameter values (and dgps)
def MC_simulate_chgpar(parameters, estimators, g_functions, changing_parameter,
                     save_file=False, filename='V3'): 
    ##########################################################################
    # Preliminaries 
    if 'save_file' in parameters.keys(): 
        save_file = parameters['save_file']
    else: save_file = False
    parameters['save_file'] = False #Don't save files in subfunctions
    
    if 'filename' in parameters.keys(): 
        filename = parameters['filename']
    else: filename = 'V3'
    
    if 'B' not in parameters.keys(): parameters['B'] = 0 #No bootstrap as default
    
    if parameters['V'] > 0 or parameters['C'] > 0 or parameters ['add_error']==True:
        if 'run_wellspecified' not in parameters.keys():  # Default: One case (observables)
            parameters['run_observables'], parameters['run_wellspecified'] = True, False   
        else: parameters['run_observables'] = True #Well-specified specified in parameters 
    else: #Default: One case (wellspecified, since observable data is the same )
        parameters['run_observables'], parameters['run_wellspecified'] = False, True  
    
    #Prepare storage
    res_betahats, res_mrgeffs, data, res_probs = {},{},{},{}
    
    if parameters['B']>0: res_boot_expects, res_boot_mrgeffs = {},{}
    
    if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
        res_betahats_obs, res_mrgeffs_obs, res_probs_obs = {},{},{}
        
    ##########################################################################
    #Repeat simulation for various parameters over a specified parameter space
    for par in changing_parameter['parameter_space']: # e.g. n in [1000,2000,....]
        #Status message
        print('Runtime:', datetime.now() - parameters['start_time'], '\tStatus: Began simulation for', 
              changing_parameter['parameter'], '=', par)    
       
        #Update the parameter we are exploring    
        parameters[changing_parameter['parameter']] = par #Specifies which parameter (e.g. n)
        
        #Run simulation with that parameter for each g. 
        if parameters['B'] == 0: #No bootstrap is default
            if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
                res_betahats[par], res_mrgeffs[par], data[par], res_probs[par], \
                res_betahats_obs[par], res_mrgeffs_obs[par], res_probs_obs[par] = \
                    MC_simulate_dgps(parameters, estimators, g_functions)
            else:         
                res_betahats[par], res_mrgeffs[par], data[par], res_probs[par] = \
                    MC_simulate_dgps(parameters, estimators, g_functions) 
        else: 
            res_betahats[par], res_mrgeffs[par], data[par], res_probs[par], \
                res_boot_expects[par], res_boot_mrgeffs[par] = \
                    MC_simulate_dgps(parameters, estimators, g_functions) 
            
    print('Runtime:', datetime.now() - parameters['start_time'], 
              '\tStatus: Finished simulations for parameters.')
    
    ##########################################################################
    # Save output (possibly)
    if save_file==True: 
        outputs = {'res_betahats': res_betahats, 'res_mrgeffs': res_mrgeffs, 'data': data, 'res_probs':res_probs, 
                   'parameters': parameters, 'estimators': estimators, 'g_functions':g_functions, 
                   'changing_parameter':changing_parameter}
        
        if parameters['B'] > 0:
                outputs.update({'res_boot_expects': res_boot_expects, 'res_boot_mrgeffs': res_boot_mrgeffs})
        
        if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
                outputs.update({'res_betahats_obs': res_betahats_obs, 'res_mrgeffs_obs': res_mrgeffs_obs, 
                                'res_probs_obs':res_probs_obs})        
        
        for output in outputs: 
            with open(os.getcwd() + '\\simulation_results\\'+'%s_%s.txt' % (filename, output), "wb") as f: 
                pickle.dump(outputs[output], f)
                        
    ##########################################################################
    # Return output
    if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
        return res_betahats, res_mrgeffs, data, res_probs, res_betahats_obs, res_mrgeffs_obs, res_probs_obs
    elif parameters['B'] > 0:
        return res_betahats, res_mrgeffs, data, res_probs, res_boot_expects,  res_boot_mrgeffs
    else: 
        return res_betahats, res_mrgeffs, data, res_probs

##########################################################################
### V3_2: Run simulation for various parameter values (and dgps)
        # Saves each iteration by itself, to prevent memory issues. 
def MC_simulate_chgpar_indfiles(parameters, estimators, g_functions, changing_parameter,
                     save_file=False, filename='V3'): 
    ##########################################################################
    #Preliminaries
    if 'save_file' in parameters.keys(): 
        save_file = parameters['save_file']
    parameters['save_file'] = False #Don't save files in subfunctions
    
    if 'filename' in parameters.keys(): 
        filename = parameters['filename']
    else: filename = 'V3'
    
    if 'B' not in parameters.keys(): parameters['B'] = 0 #No bootstrap as default
    
    if parameters['V'] > 0 or parameters['C'] > 0 or parameters ['add_error']==True:
        if 'run_wellspecified' not in parameters.keys():  # Default: One case (observables)
            parameters['run_observables'], parameters['run_wellspecified'] = True, False   
        else: parameters['run_observables'] = True #Well-specified specified in parameters 
    else: #Default: One case (wellspecified, since observable data is the same )
        parameters['run_observables'], parameters['run_wellspecified'] = False, True      
    
    #Store parameters (once, rather than doing it for each parameter)
    outputs = {'parameters': parameters, 'estimators': estimators, 'g_functions':g_functions, 
               'changing_parameter':changing_parameter}
    for output in outputs: 
        with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_pars.txt' % (filename, output), "wb") as f: 
            pickle.dump(outputs[output], f)
    
    
    ##########################################################################    
    #Repeat simulation for various parameters over a specified parameter space
    for par in changing_parameter['parameter_space']: # e.g. n in [1000,2000,....]
        #Status message
        print('Runtime:', datetime.now() - parameters['start_time'], '\tStatus: Began simulation for', 
              changing_parameter['parameter'], '=', par)    
       
        #Update the parameter we are exploring    
        parameters[changing_parameter['parameter']] = par #Specifies which parameter (e.g. n)
        
        #Run simulation with that parameter for each g. 
        if parameters['B'] == 0: #No bootstrap is default
            if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
                res_betahats, res_mrgeffs, data, res_probs, \
                res_betahats_obs, res_mrgeffs_obs, res_probs_obs = \
                    MC_simulate_dgps(parameters, estimators, g_functions) 
            else: 
                res_betahats, res_mrgeffs, data, res_probs = \
                    MC_simulate_dgps(parameters, estimators, g_functions) 
        else: 
            res_betahats, res_mrgeffs, data, res_probs, res_boot_expects,  res_boot_mrgeffs = \
                MC_simulate_dgps(parameters, estimators, g_functions) 
        
        ##########################################################################
        #Save stage files
        outputs = {'res_betahats': res_betahats, 'res_mrgeffs': res_mrgeffs, 'data': data, 'res_probs':res_probs}
        
        if parameters['B'] > 0:
                outputs.update({'res_boot_expects': res_boot_expects, 'res_boot_mrgeffs': res_boot_mrgeffs})

        if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
                outputs.update({'res_betahats_obs': res_betahats_obs, 'res_mrgeffs_obs': res_mrgeffs_obs, 
                                'res_probs_obs':res_probs_obs})        
        
        for output in outputs: 
            with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_%s.txt' % (filename, output, par), "wb") as f: 
                pickle.dump(outputs[output], f)

            
    print('Runtime:', datetime.now() - parameters['start_time'], 
              '\tStatus: Finished simulations for parameters.')
    
#    ##########################################################################
#    #Compile stage files 
#    for output in ('res_betahats', 'res_mrgeffs', 'data', 'res_probs'): 
#        try: 
#            temp = {}
#            for par in changing_parameter['parameter_space']:
#                with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_%s.txt' % (output, filename, par), "rb") as f: 
#                    temp[par] = pickle.loads(f.read())
#                    
#            with open(os.getcwd() + '\\simulation_results\\'+'%s_%s.txt' % (output, filename), "wb") as f: 
#                pickle.dump(temp, f)
#        except MemoryError: 
#            print('Memory error for ', output,'. Result files were not compiled, so only single iterations are available', sep='')
#        
#    #Load and return final files 
#    try: 
#        with open(os.getcwd() + '\\simulation_results\\'+'res_betahats_%s.txt' % filename, "rb") as f: 
#            res_betahats = pickle.loads(f.read())
#        with open(os.getcwd() + '\\simulation_results\\'+'res_mrgeffs_%s.txt' % filename, "rb") as f: 
#            res_mrgeffs = pickle.loads(f.read())
#        with open(os.getcwd() + '\\simulation_results\\'+'data_%s.txt' % filename, "rb") as f: 
#            data = pickle.loads(f.read())
#        with open(os.getcwd() + '\\simulation_results\\'+'res_probs_%s.txt' % filename, "rb") as f: 
#            res_probs = pickle.loads(f.read())
#    except MemoryError: 
#        print('Memory error. Results were not loaded, but are available in files.')
        
     ##########################################################################       
#    # Return output
    if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
        return res_betahats, res_mrgeffs, data, res_probs, res_betahats_obs, res_mrgeffs_obs, res_probs_obs
    elif parameters['B'] > 0:
        return res_betahats, res_mrgeffs, data, res_probs, res_boot_expects,  res_boot_mrgeffs
    else: 
        return res_betahats, res_mrgeffs, data, res_probs

##########################################################################
### V3_3: Run simulation for various parameter values (and dgps)
        # Saves each iteration by itself, to prevent memory issues. 
        #  Iterations are saved for each g fu8nction
def MC_simulate_chgpar_indfiles_g(parameters, estimators, g_functions, changing_parameter,
                     save_file=False, filename='V3'): 
    ##########################################################################
    #Preliminaries
    if 'save_file' in parameters.keys(): 
        save_file = parameters['save_file']
    parameters['save_file'] = False #Don't save files in subfunctions
    
    if 'filename' in parameters.keys(): 
        filename = parameters['filename']
    else: filename = 'V3'
    
    if 'B' not in parameters.keys(): parameters['B'] = 0 #No bootstrap as default
    
    if parameters['V'] > 0 or parameters['C'] > 0 or parameters ['add_error']==True:
        if 'run_wellspecified' not in parameters.keys():  # Default: One case (observables)
            parameters['run_observables'], parameters['run_wellspecified'] = True, False   
        else: parameters['run_observables'] = True #Well-specified specified in parameters 
    else: #Default: One case (wellspecified, since observable data is the same )
        parameters['run_observables'], parameters['run_wellspecified'] = False, True      
    
    #Store parameters (once, rather than doing it for each parameter)
    outputs = {'parameters': parameters, 'estimators': estimators, 'g_functions':g_functions, 
               'changing_parameter':changing_parameter}
    for output in outputs: 
        with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_pars.txt' % (filename, output), "wb") as f: 
            pickle.dump(outputs[output], f)
        
    ##########################################################################
    #Repeat simulation for various parameters over a specified parameter space
    for par in changing_parameter['parameter_space']: # e.g. n in [1000,2000,....]
        #Status message
        print('Runtime:', datetime.now() - parameters['start_time'], '\tStatus: Began simulation for', 
              changing_parameter['parameter'], '=', par)    
       
        #Update the parameter we are exploring    
        parameters[changing_parameter['parameter']] = par #Specifies which parameter (e.g. n)
        parameters['filename'] = filename + '_' + str(par)
        
        
        #Run simulation with that parameter for each g. 
        if parameters['B'] == 0: #No bootstrap is default
            if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
                res_betahats, res_mrgeffs, data, res_probs, \
                res_betahats_obs, res_mrgeffs_obs, res_probs_obs = \
                    MC_simulate_dgps_indfiles(parameters, estimators, g_functions)
            else: 
                res_betahats, res_mrgeffs, data, res_probs = \
                    MC_simulate_dgps_indfiles(parameters, estimators, g_functions) 
        else: 
            res_betahats, res_mrgeffs, data, res_probs, res_boot_expects,  res_boot_mrgeffs = \
                MC_simulate_dgps_indfiles(parameters, estimators, g_functions) 
                    
            
#        ##########################################################################
#        #Save stage files
#        outputs = {'res_betahats': res_betahats, 'res_mrgeffs': res_mrgeffs, 'data': data, 'res_probs':res_probs}
#        for output in outputs: 
#            with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_%s.txt' % (filename, output, par), "wb") as f: 
#                pickle.dump(outputs[output], f)

            
    print('Runtime:', datetime.now() - parameters['start_time'], 
              '\tStatus: Finished simulations for parameters.')
            
    ##########################################################################
    # Return output
    if parameters['run_wellspecified'] == True and parameters['run_observables'] == True:
        return res_betahats, res_mrgeffs, data, res_probs, res_betahats_obs, res_mrgeffs_obs, res_probs_obs
    elif parameters['B'] > 0:
        return res_betahats, res_mrgeffs, data, res_probs, res_boot_expects,  res_boot_mrgeffs
    else: 
        return res_betahats, res_mrgeffs, data, res_probs

##########################################################################
##########################################################################
##########################################################################
### Loaders 
    # These functions load previously generated simulation results, which were saved to files. 
    
##########################################################################
### Basic loader
    
def MC_load_results(filename): 
    #Load files 
    with open(os.getcwd() + '\\simulation_results\\'+'%s_res_betahats.txt' % filename, "rb") as f: 
        res_betahats = pickle.loads(f.read())
    with open(os.getcwd() + '\\simulation_results\\'+'%s_res_mrgeffs.txt' % filename, "rb") as f: 
        res_mrgeffs = pickle.loads(f.read())
    with open(os.getcwd() + '\\simulation_results\\'+'%s_data.txt' % filename, "rb") as f: 
        data = pickle.loads(f.read())
    with open(os.getcwd() + '\\simulation_results\\'+'%s_res_probs.txt' % filename, "rb") as f: 
        res_probs = pickle.loads(f.read())
    try:
        with open(os.getcwd() + '\\simulation_results\\'+'%s_parameters.txt' % filename, "rb") as f: 
            parameters = pickle.loads(f.read())
    except Exception: 
        parameters = None
    try:
        with open(os.getcwd() + '\\simulation_results\\'+'%s_estimators.txt' % filename, "rb") as f: 
            estimators = pickle.loads(f.read())
    except Exception: 
        estimators = None
    try: 
        with open(os.getcwd() + '\\simulation_results\\'+'%s_g_functions.txt' % filename, "rb") as f: 
            g_functions = pickle.loads(f.read())
    except Exception: 
        g_functions = None
    try: 
        with open(os.getcwd() + '\\simulation_results\\'+'%s_changing_parameter.txt' % filename, "rb") as f: 
            changing_parameter = pickle.loads(f.read())
    except Exception: 
        changing_parameter = None
        
    return res_betahats, res_mrgeffs, data, res_probs, parameters, estimators, g_functions, changing_parameter

##########################################################################
### Load parameters 
def MC_load_pars(filename, load_individually=True): 
    # All imulations stored together
    if load_individually == False: 
        with open(os.getcwd() + '\\simulation_results\\'+'%s_parameters.txt' % filename, "rb") as f: 
            parameters = pickle.loads(f.read())
        with open(os.getcwd() + '\\simulation_results\\'+'%s_estimators.txt' % filename, "rb") as f: 
            estimators = pickle.loads(f.read())
        try: 
            with open(os.getcwd() + '\\simulation_results\\'+'%s_g_functions.txt' % filename, "rb") as f: 
                g_functions = pickle.loads(f.read())
        except Exception: 
                g_functions = None
        try: 
            with open(os.getcwd() + '\\simulation_results\\'+'%s_changing_parameter.txt' % filename, "rb") as f: 
                changing_parameter = pickle.loads(f.read())
        except Exception: 
                changing_parameter = None
    
    # Simulations for each parameter in parameter space stored seperately due to memory issues
    if load_individually == True: 
        with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_parameters_pars.txt' % filename, "rb") as f: 
            parameters = pickle.loads(f.read())
        with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_estimators_pars.txt' % filename, "rb") as f: 
            estimators = pickle.loads(f.read())
        try: 
            with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_g_functions_pars.txt' % filename, "rb") as f: 
                g_functions = pickle.loads(f.read())
        except Exception: 
                g_functions = None
        try: 
            with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_changing_parameter_pars.txt' % filename, "rb") as f: 
                changing_parameter = pickle.loads(f.read())
        except Exception: 
                changing_parameter = None
    
                
        if changing_parameter != None: 
            return parameters, estimators, g_functions, changing_parameter
        elif g_functions != None: 
            return parameters, estimators, g_functions
        else: 
            return parameters, estimators