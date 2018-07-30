#Preliminaries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy as sp
from collections import defaultdict
import pickle
import statistics

#Import own files 
import sys
sys.path.append(r'F:\Documents\_Speciale\Code\Functions')
sys.path.append(r'C:\Users\rbjoe\Dropbox\Kugejl\10.semester\TheEnd\Code\Functions')
import monte_carlo_simulation as mc
import dgp_stuff as dgp
import neural_net as nn
import estimators as est
import tablez as tblz

#This file has the following sections: 
    # Wrappers 
    # Bottom level functions. 
    
# Design philosophy: The "bottom level" function is designed to apply to a list of simulations. 
# Data objects which cover a number of such lists (for various models, functions etc.) are
# then handled through general "wrappers", which splits the object into the various lists 
# and applies the bottom level functions.

###############################################################################
###############################################################################
###############################################################################
### WRAPPERS
    
### Wrapper for computing function on each simulation'
#def comp_wrapper_sim(function, series, y = est.dd_inf(), dgp_series=est.dd_inf(), **kwargs): 
#    comp = [function(serie, y, dgp_series) for serie in series ]    


###############################################################################    
#Wrapper for simulations with different estimated models on train/test splits
def comp_wrapper_model(function, *model_series, 
                       y = est.dd_inf(), dgp_series=est.dd_inf(), comp_kws={}): 
    #Example: function takes average (so comp is average) and series are the probabilities for each model
    comp = {}
    for model in model_series[0].keys(): #zero implies first series in variable number of series.
        comp[model] = {} #Store a value for each input model
        for split in model_series[0][model].keys():  #Repeat for train and test
            series = [serie[model][split] for serie in model_series] #Extract the data for the specific case
            #print(model, split)
            comp[model][split] = function(*series, #Perform computation in input function
                                            y=y['y'][split], dgp_series=dgp_series['DGP'][split], 
                                            **comp_kws) 
    return comp 

def comp_wrapper_addmodel(function, model_series, new_series='Mode', y=est.dd_inf(), dgp_series=est.dd_inf()):
    model_series[new_series] = {}
    for split in model_series['DGP'].keys(): 
        model_series[new_series][split] = function(y=y['y'][split])
    return model_series
    
###############################################################################
### Applies a function designed for one simulation to all the g functions
def comp_wrapper_g(function, *g_series, wrapper_model=comp_wrapper_model, 
                   y = est.dd_inf(), dgp_series=est.dd_inf(), comp_kws={}): 
    #Example: g_series as probabilities, function as predict from probabilities. 
    comp = {}
    for g in g_series[0].keys():
        series = [serie[g] for serie in g_series]
        comp[g] = wrapper_model(function, *series,
                                y=y[g], dgp_series=dgp_series[g], 
                                comp_kws=comp_kws)
    return comp

###############################################################################
### Applies a function for one simulation to all g functions for the explored parameter space
def comp_wrapper_par(function, *par_series, wrapper_model=comp_wrapper_g, 
                   y = est.dd_inf(), dgp_series=est.dd_inf(), comp_kws={}): 
    #Example: g_series as probabilities, function as predict from probabilities. 
    comp = {}
    for par in par_series[0].keys(): # Zero just picks first series in par_series
        series = [serie[par] for serie in par_series]
        comp[par] = wrapper_model(function, *series, 
                                    y=y[par], dgp_series=dgp_series[par], 
                                    comp_kws=comp_kws)
    return comp

###############################################################################
### Applies function across gs if they are saved in individual files
def comp_wrapper_gseries(function, *g_series, g_functions, #Requires g_functions, since series is strings
                         wrapper_model=comp_wrapper_model, 
                         filename = 'V2', 
                         comp_kws={}, **kwargs): 
    #NOTE: Series are now strings, which specify the file to be loaded. 
        #Hence the added antics in the beginning. 
    
    # Check for added series
    if 'dgp_series' in kwargs.keys():
        if isinstance(kwargs['dgp_series'], str): 
            dgp_series = kwargs['dgp_series'] #String can't be copied (nor do we need to)
        else: 
            dgp_series = kwargs['dgp_series'].copy() #Copy, to avoid infinite dict 
    else: 
        dgp_series = est.dd_inf()
    if 'y' in kwargs.keys(): 
        if isinstance(kwargs['y'], str): 
            y = kwargs['y']
        else: 
            y = kwargs['y'].copy()
    else: 
        y = est.dd_inf() 
        
    # Prepare to load relevant files (original names are replaced for consistency)
    output={}
    for i in range(0,len(g_series)):
        output[i] = g_series[i]   
    if dgp_series !=  est.dd_inf(): 
        output_dgp = dgp_series
    else: output_dgp = None
    if y !=  est.dd_inf(): 
        output_y = y
    else: output_y = None
    
    
    #Load and process data 
    comp = {}
    for g in g_functions.keys():
        #Load data 
        series = []
        for i in range(0,len(g_series)):
            with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_%s.txt' \
               % (filename, g, output[i]), "rb") as f:   
                series.append(pickle.loads(f.read()))
#            print(y)
        if output_dgp !=  None: 
            with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_%s.txt' \
               % (filename, g, output_dgp), "rb") as f:
                dgp_series  = pickle.loads(f.read())    
        if output_y !=  None: 
            with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_%s.txt' \
               % (filename, g, output_y), "rb") as f:
                y  = pickle.loads(f.read())  
        # Perform computation below
        comp[g] = wrapper_model(function, *series,
                                y=y, dgp_series=dgp_series, 
                                comp_kws=comp_kws)
    return comp
    

###############################################################################
### Calculate a series over the explored parameter space for each g. 
def comp_wrapper_parseries(function, *par_series, wrapper_model=comp_wrapper_g, 
                   mult_series = False, summary_function = np.mean, 
                   load_individually=False, filename='V3', parameter_space=None, 
                   comp_kws={},
                   **kwargs): 
    #NOTE: Series are now strings, which specify the file to be loaded. 
        #Hence the added antics in the beginning. 
    
    # Check for added series
    if 'dgp_series' in kwargs.keys(): 
        dgp_series = kwargs['dgp_series']
    else: 
        dgp_series = est.dd_inf()
    if 'y' in kwargs.keys(): 
        y = kwargs['y']
    else: 
        y = est.dd_inf() 
    
    # Prepare to load relevant files
    if load_individually ==False: 
        parameter_space = par_series[0].keys()
    else: #loads individually
        output={}
        for i in range(0,len(par_series)):
            output[i] = par_series[i]   
        if dgp_series !=  est.dd_inf(): 
            output_dgp = dgp_series
        else: output_dgp = None
        if y !=  est.dd_inf(): 
            output_y = y
        else: output_y = None
        
    
    comp = {}    
    #Prepare dict with a dataframe for each g function 
    if load_individually==False: #If data was compiled before
        for g in par_series[0][np.random.choice(list(par_series[0].keys()))].keys():
            comp[g] = {}
            for split in par_series[0][np.random.choice(list(par_series[0].keys()))][g]['DGP'].keys(): 
                if mult_series ==False: 
                    comp[g][split] = pd.DataFrame()
                else: 
                    comp[g][split] = {}
                    for i in range(0,mult_series): 
                        comp[g][split][i] = pd.DataFrame()
            
    else:  #If data is in individual files, we need to load one to get the subsequent structure
         with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_%s.txt' \
                   % (filename, output[0], np.random.choice(parameter_space)), "rb") as f:   
             temp = pickle.loads(f.read())
         for g in temp.keys():
            comp[g] = {}
            for split in temp[g]['DGP'].keys(): 
                if mult_series ==False: 
                    comp[g][split] = pd.DataFrame()
                else: 
                    comp[g][split] = {}
                    for i in range(0,mult_series): 
                        comp[g][split][i] = pd.DataFrame()
    
    #Append rows to dataframe for each parameter    
    for par in parameter_space:
        if load_individually==False:
            series = [serie[par] for serie in par_series]
            temp = wrapper_model(function, *series, 
                                 y=y[par], dgp_series=dgp_series[par], 
                                 comp_kws=comp_kws)            
        
        else: 
            series = []
            
            for i in range(0,len(par_series)):
                with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_%s.txt' \
                   % (filename, output[i], par), "rb") as f:   
                    series.append(pickle.loads(f.read()))
            if output_dgp !=  None: 
                with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_%s.txt' \
                   % (filename, output_dgp, par), "rb") as f:
                    dgp_series  = pickle.loads(f.read())    
            if output_y !=  None: 
                with open(os.getcwd() + '\\simulation_results\\single_iterations\\'+'%s_%s_%s.txt' \
                   % (filename, output_y, par), "rb") as f:
                    y  = pickle.loads(f.read())    
            temp = wrapper_model(function, *series, 
                                 y=y, dgp_series=dgp_series, 
                                 comp_kws=comp_kws)            

        
        row = {}
        #Calculate results for each g function
        for g in temp.keys():
            row[g] = {}        
            for split in temp[g][np.random.choice(list(temp[g].keys()))].keys():  #Random model
                row[g][split] = {}
                if mult_series ==False: 
                    row[g][split]['Parameter'] = par 
                else: 
                    for i in range(0, mult_series): 
                        row[g][split][i] = {}
                        row[g][split][i]['Parameter'] = par 
                
                #Calculate a column with results for each model
                for model in temp[g].keys():
                    if mult_series ==False:   #Check for number of columns in series 
                        row[g][split][model] = summary_function(temp[g][model][split], axis=0)
                    else: #If more than one value, summarize for each variable (e.g. for each beta)
                        temp2 = summary_function(temp[g][model][split], axis=0)
                        for i in range(0, mult_series): 
                            row[g][split][i][model] = temp2[i]                
                
                #Append the result rows back into the dataframes
                if mult_series ==False:
                     comp[g][split]=comp[g][split].append(row[g][split], ignore_index=True)
                else: 
                    for i in range(0,mult_series): 
                        comp[g][split][i]=comp[g][split][i].append(row[g][split][i], ignore_index=True)
#    del output, output_dgp, output_y, dgp_series, y
    
    return comp


###############################################################################
### Calculate a series over the explored parameter space for each g. 
    # Note: This function does the same as the above, but if there are individual files for each g. 
def comp_wrapper_parseries_g(function, *par_series, g_functions, parameter_space, 
                             wrapper_model=comp_wrapper_gseries, 
                             summary_function=np.mean, 
                             filename = 'V3', mult_series = False, 
                             comp_kws={}, **kwargs): 
    # Check for added series
    if 'dgp_series' in kwargs.keys(): 
        dgp_series = kwargs['dgp_series']
    else: 
        dgp_series = est.dd_inf()
    if 'y' in kwargs.keys(): 
        y = kwargs['y']
    else: 
        y = est.dd_inf() 
        
    #Prepare dict with a dataframe for each g function 
    comp={}
    for g in g_functions.keys(): 
        comp[g] = {}
        for split in ('Train', 'Test'): 
                if mult_series == False: 
                    comp[g][split] = pd.DataFrame()
                else: 
                    comp[g][split] = {}
                    for i in range(0,mult_series): 
                        comp[g][split][i] = pd.DataFrame()   
    
    #Append rows to dataframe for each parameter 
    for par in parameter_space: 
        #Get computation for current parameter set
        temp = wrapper_model(function, *par_series, g_functions=g_functions, 
                             filename = filename + '_' + str(par), 
                             dgp_series = dgp_series, y=y,
                             comp_kws=comp_kws)
        row = {}
        #Calculate results for each g function
        for g in temp.keys():
            row[g] = {}        
            for split in temp[g][np.random.choice(list(temp[g].keys()))].keys():  #Random model
                row[g][split] = {}
                if mult_series ==False: 
                    row[g][split]['Parameter'] = par 
                else: 
                    for i in range(0, mult_series): 
                        row[g][split][i] = {}
                        row[g][split][i]['Parameter'] = par 
                
                #Calculate a column with results for each model
                for model in temp[g].keys():
                    if mult_series ==False:   #Check for number of columns in series 
                        row[g][split][model] = summary_function(temp[g][model][split], axis=0)
                    else: #If more than one value, summarize for each variable (e.g. for each beta)
                        temp2 = summary_function(temp[g][model][split], axis=0)
                        for i in range(0, mult_series): 
                            row[g][split][i][model] = temp2[i]                
                
                #Append the result rows back into the dataframes
                if mult_series ==False:
                     comp[g][split]=comp[g][split].append(row[g][split], ignore_index=True)
                else: 
                    for i in range(0,mult_series): 
                        comp[g][split][i]=comp[g][split][i].append(row[g][split][i], ignore_index=True)
    return comp                        

###############################################################################
###############################################################################
###############################################################################
### Bottom level functions. 
### Each takes as an input a list of simulations and performs an operation individually or aggregately. 

###############################################################################
### Compute average with-in each simulation iteration
def comp_average(series, **kwargs): 
    
    comp = [np.nanmean(serie.astype(np.float64), axis=0) for serie in series]
    
    if 'coefficient' in kwargs.keys(): # Option for returning only one  coefficient
        comp = [np.asarray([serie[kwargs['coefficient']]]) for serie in comp]

#    if 'coefficient' not in kwargs.keys(): # Calculate all averages
#        comp = [np.mean(serie.astype(np.float64), axis=0) for serie in series]
#    else: # Option for returning only one  
#        comp = [np.mean(serie.astype(np.float64), axis=0)[kwargs['coefficient']] for serie in series]

    return comp

###############################################################################
### Error measures
### Cmpute the mean error of a series vs. DGP (i.e. bias). 
def comp_me(series, dgp_series, **kwargs): 
#    comp = [np.mean((serie.astype(np.float64)-dgp_serie.astype(np.float64)), axis=0)[coefficient] \
#                for serie, dgp_serie in zip(series, dgp_series)]    
    comp = [np.mean((serie.astype(np.float64)-dgp_serie.astype(np.float64)), axis=0) \
                for serie, dgp_serie in zip(series, dgp_series)]    
    
    if 'coefficient' in kwargs.keys(): # Option for returning only one  coefficient
        comp = [np.asarray([serie[kwargs['coefficient']]]) for serie in comp]    
#        comp = comp[kwargs['coefficient']]

    return comp

def comp_mme(series, dgp_series, **kwargs): 
    comps = comp_me(series, dgp_series, **kwargs)
    mean_comp =  [np.mean(comp) for comp in comps]
    return mean_comp

### Compute the mean squared error of a series vs. the DGP.
def comp_mse(series, dgp_series, **kwargs): 
    comp = [np.mean((serie.astype(np.float64)-dgp_serie.astype(np.float64))**2, axis=0) \
               for serie, dgp_serie in zip(series, dgp_series)]    
    #mse = [np.mean(np.square(serie-dgp_serie), axis=0) for serie, dgp_serie in zip(series, dgp_series)]        

    if 'coefficient' in kwargs.keys(): # Option for returning only one  coefficient
        comp = [np.asarray([serie[kwargs['coefficient']]]) for serie in comp]
    
    return comp

def comp_mmse(series, dgp_series, **kwargs): 
    comps = comp_mse(series, dgp_series, **kwargs)
    mean_comp =  [np.mean(comp) for comp in comps]
    return mean_comp

# Root mean squred error (RMSE)
def comp_rmse(series, dgp_series, **kwargs):  #Root mean squred error
    
    comp = [np.sqrt(np.nanmean((serie.astype(np.float64)-dgp_serie.astype(np.float64))**2, axis=0)) \
                for serie, dgp_serie in zip(series, dgp_series)]

    if 'coefficient' in kwargs.keys(): # Option for returning only one  coefficient
        comp = [np.asarray([serie[kwargs['coefficient']]]) for serie in comp]
    
    return comp

def comp_mrmse(series, dgp_series, **kwargs):  #Mean oot mean squred error
    rmses = comp_rmse(series, dgp_series)
    mrmse =  [np.nanmean(rmse) for rmse in rmses]
    return mrmse

# Squared error for estimator of average 
def comp_se_avg(series, dgp_series, **kwargs):  #Root mean squred error
    comp = [((np.mean(serie,axis=0)-np.mean(dgp_serie,axis=0))**2) for serie, dgp_serie in zip(series, dgp_series)]    
    
    if 'coefficient' in kwargs.keys(): # Option for returning only one  coefficient
        comp = [np.asarray([serie[kwargs['coefficient']]]) for serie in comp]
    
    return comp


def comp_mse_avg(series, dgp_series, **kwargs):  #Mean oot mean squred error
    comps = comp_se_avg(series, dgp_series)
    mean_comp =  [np.mean(comp) for comp in comps]
    return mean_comp

# Root mean squared error for estimator of average 
def comp_rse_avg(series, dgp_series, **kwargs):  #Root mean squred error
    comp = [np.sqrt((np.mean(serie,axis=0)-np.mean(dgp_serie,axis=0))**2) for serie, dgp_serie in zip(series, dgp_series)]    

    if 'coefficient' in kwargs.keys(): # Option for returning only one  coefficient
        comp = [np.asarray([serie[kwargs['coefficient']]]) for serie in comp]
    
    return comp


def comp_mrse_avg(series, dgp_series, **kwargs):  #Mean oot mean squred error
    comps = comp_rse_avg(series, dgp_series)
    mean_comp =  [np.mean(comp) for comp in comps]
    return mean_comp

###############################################################################
# Bootstrap stuff
def comp_boot_average(series, **kwargs): #
    comp = [np.nanmean(serie.astype(np.float64), axis=0) for serie in series]
    
    if 'coefficient' in kwargs.keys(): # Option for returning only one  coefficient
        comp = [np.asarray([serie[:,kwargs['coefficient']]]) for serie in comp]

    return comp  

def comp_boot_getcoeff(series, **kwargs): #
    
    if 'coefficient' in kwargs.keys(): # Option for returning only one  coefficient
        comp = [np.asarray([serie[:,kwargs['coefficient']]]) for serie in series]

    return comp      

def comp_boot_average_sdev(series, run_average = False, **kwargs): #
    if run_average == True: # Needs to be run through boot average first 
        comps = comp_boot_average(series, **kwargs)
    elif 'coefficient' in kwargs.keys():
        comps = [np.asarray([serie[:,kwargs['coefficient']]]) for serie in series]
    else: 
        comps = series 
        
    sdev_comp = [np.nanstd(comp) for comp in comps]
    
    return sdev_comp

def comp_boot_average_test(series, run_average = False, **kwargs):
    if run_average == True: # Needs to be run through boot average first 
        comps = comp_boot_average(series, **kwargs)
    elif 'coefficient' in kwargs.keys():
        comps = [np.asarray([serie[:,kwargs['coefficient']]]) for serie in series]
    else: 
        comps = series 

    comp_reject = [tblz.comp_bootstrap_test(comp) for comp in comps]
    
    return comp_reject 

def comp_boot_average_confint(series, run_average = False, 
                              conf_level=0.05, method='linear',
                              **kwargs): #
    if run_average == True: # Needs to be run through boot average first 
        comps = comp_boot_average(series, **kwargs)
    elif 'coefficient' in kwargs.keys():
        comps = [np.asarray([serie[:,kwargs['coefficient']]]) for serie in series]
    else: 
        comps = series 
        
    lower = [np.percentile(comp, 100*(0.5*conf_level), interpolation=method) for comp in comps]
    upper = [np.percentile(comp, 100*(1-0.5*conf_level), interpolation=method) for comp in comps]
#    conf = [np.concatenate((lower,upper)) for comp in comps]
#    print(conf[0])
    conf = [lower, upper]
    
    return conf


###############################################################################
# Compute attenuation factors (or something along those lines)
def comp_attenuationfactor(series, dgp_series, **kwargs):
    comp = [np.mean(np.divide(serie.astype(np.float64),
                              dgp_serie.astype(np.float64),
                     where=dgp_serie != 0),
            axis=0) for serie, dgp_serie in zip(series, dgp_series)]
    
    if 'coefficient' in kwargs.keys(): # Option for returning only one  coefficient
        comp = [np.asarray([serie[kwargs['coefficient']]]) for serie in comp]
    
    return comp

def comp_attenuationfactor_mean(series, dgp_series, **kwargs):  #Mean oot mean squred error
    comps = comp_attenuationfactor(series, dgp_series)
    mean_comp =  [np.nanmean(comp.astype(np.float64)) for comp in comps]
    return mean_comp
    ###############################################################################
# Concatenate across all simulations to get a pooled sample
def comp_pool_simulations(series, variable=0, **kwargs): 
    try: 
        iteration = [serie[:,variable] for serie in series ]
    except Exception: 
        iteration = [serie.iloc[:,variable] for serie in series ]
    pool = np.concatenate(iteration)
    return pool

###############################################################################
### Predict yhat based on probability
def comp_predict_from_probability(probability_series, **kwargs):
    yhat = [pd.DataFrame(probability>=0.5) for probability in probability_series]
    return yhat

def add_mode(y, **kwargs): 
    #mode = [pd.concat([sim_y.mode().iloc[0]]*len(sim_y), ignore_index=True) for sim_y in y] 
    mode = [pd.DataFrame(np.resize(sim_y.mode().iloc[0], (len(sim_y),1))) for sim_y in y] 
    #from scipy.stats import mode #First value is the mode
    #mod = [pd.concat([pd.DataFrame(mode(sim_y)[0])]*len(sim_y), ignore_index=True) for sim_y in y]    
    return mode
    
def add_mode1(function, series, y, **kwargs): 
    series['Mode'] = {}
    for split in series['DGP'].keys(): 
        series['Mode'][split] = [sim_y.mode() for sim_y in y['y'][split]]
    return series
    

###############################################################################
## Various accuracy measures 
def comp_acc_from_prediction(yhat, y, **kwargs): 
    acc = [float((np.mean(yhat[i]==y[i]))) for i in range(0,len(yhat))]
    return acc

def comp_accmeasures_from_prediction(yhat, y, **kwargs): 
    #Compute true/false positives/negatives
    tp = [np.sum(np.logical_and(y[i]==1, yhat[i]==1)) for i in range(0,len(yhat))]
    fp = [np.sum(np.logical_and(y[i]==0, yhat[i]==1)) for i in range(0,len(yhat))]
    fn = [np.sum(np.logical_and(y[i]==1, yhat[i]==0)) for i in range(0,len(yhat))]
    tn = [np.sum(np.logical_and(y[i]==0, yhat[i]==0)) for i in range(0,len(yhat))]
    
    accs = {} #Compute various measures 
    accs['Accuracy']  = [float((tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i])) for i in range(0,len(yhat))]
    accs['Precision'] = [float((tp[i])/(tp[i]+fp[i])) for i in range(0,len(yhat))]
    accs['Recall']    = [float((tp[i])/(tp[i]+fn[i])) for i in range(0,len(yhat))]
    accs['F1 score']  = [float((2*(tp[i])/(tp[i]+fp[i])*(tp[i])/(tp[i]+fn[i]))/((tp[i])/(tp[i]+fp[i])+(tp[i])/(tp[i]+fn[i]))) for i in range(0,len(yhat))]

    return accs

###############################################################################
## Print parameters when saved in dict 
def parameter_print(parameters):
    for key,value in list(parameters.items()):
        print('%s: %s' %(key, value) )


