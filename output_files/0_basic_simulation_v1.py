9
# IMPORT DEPENDENCIES
import numpy as np
import seaborn as sns
from datetime import datetime 
from collections import defaultdict
import importlib

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
#import auxilliary as aux 
sns.set_style('whitegrid')

time_start = datetime.now()

###############################################################################
# DEFINE PARAMETERS
parameters = {}
#Generel set-up
parameters['seed'] = 60677
#parameters['seed'] = np.random.randint(10**5)               #Seed for pseudo-random draws            
#print(parameters['seed'])
parameters['M'] = 10                    # Number of simulation repetitions
parameters['k'] = 1                     # Number of regressors
parameters['n'] = 1*10**3               # Number of observations 
parameters['V'] = 1                    # Number of irrelevant variables
parameters['C'] = 0                     # Number of confounding variables 
parameters['Z'] = 0                     # Number of instruments. 
parameters['B'] = 0                     # Bootstrap replications
np.random.seed(parameters['seed'])      # Also restarts at seed in each simulation
parameters['run_wellspecified'] = False 
parameters['add_error'] = False 

#Technical stuff
parameters['parallel'] = False           # Simulations run parallel or serial
parameters['decimals'] = 2              #Decimals in output tables 
parameters['reduce_size'] = False       #Saves simulations as 16 insted of 64
parameters['save_file'] = False       #Save results in files 
parameters['start_time']= datetime.now() #Time measure used for status emssages

#Specifications for how data is drawn
parameters['beta_distribution'] = dgp.draw_beta_normal      #Draw betas from this distribution
parameters['beta_mean'], parameters['beta_scale'] = 1, 1    #Mean and scale for beta distribution
parameters['redraw'] = False                                #Redraw beta in each simulation?
parameters['x_distribution']= dgp.draw_x_normal             #Dr  aw x from this distribution
#parameters['x_distribution']= dgp.draw_x_normal_iv
#parameters['x_distribution']= dgp.draw_x_normal_iv_v2
parameters['x_distribution_parameters'] = dgp.gen_x_normal_unitvariance_randommean #Draw mean and correlation for x
#parameters['x_distribution_parameters'] = dgp.gen_x_normal_unitvariance_randommean_iv
parameters['x_mean']= 0                                     #Average expected value of x variables. 
parameters['u_distribution'] = dgp.draw_u_normal             #Define error term for y. 
parameters['u_scale'] = 1 
parameters['y_generate'] = dgp.gen_y_reg                 # Compute y from error and g. 
parameters['y_squashing'], parameters['y_squashing_prime'] =  dgp.linear_output, dgp.linear_output_prime
#parameters['error_distribution_parameters'] = dgp.gen_error_normal # Distribution for optional measurement error 
parameters['error_distribution'] = dgp.gen_error_normal_indep
parameters['error_scale'] = 2
parameters['error_distribution'] = dgp.draw_x_normal



###############################################################################
#DEFINE DGP FUNCTIONS
g_functions = {}
g_functions['Linear']        = {'g_name': 'Linear', 'g_dgp': dgp.g_logit,     #Generator function g(x,beta)
                                   'g_parameters': dgp.g_logit_pars,  # Length of beta (may vary from k)
                                   'g_dgp_prime': dgp.g_logit_prime,            # g'(x,beta) wrt. x. 
                                   'g_dgp_prime_beta': dgp.g_logit_prime_beta, # Optional: g'(x,beta) wrt. beta. Else numeric diff. 
                                   'g_hyper_nn': {'layers': (30,)}}             # Optional: Hyperparameters passed to neural net                                
g_functions['Polynomial_2'] = {'g_name': 'Polynomial (2)',  'g_dgp': dgp.g_polynomial_2,  'g_parameters': dgp.g_polynomial_par_2,
                                   'g_dgp_prime': dgp.g_polynomial_prime_2, 'g_dgp_prime_beta':dgp.g_polynomial_prime_beta_2,
                                   'g_hyper_nn': {'layers': (30,)}}
g_functions['Polynomial_3'] = {'g_name': 'Polynomial (3)',  'g_dgp': dgp.g_polynomial_3, 'g_parameters': dgp.g_polynomial_par_3,
                                   'g_dgp_prime': dgp.g_polynomial_prime_3, 'g_dgp_prime_beta':dgp.g_polynomial_prime_beta_3,
                                   'g_hyper_nn': {'layers': (30,)}}
g_functions['Wiggly']    = {'g_name': 'Wiggly',   'g_dgp': dgp.g_wiggly, 'g_parameters': dgp.g_wiggly_pars,
                                   'g_dgp_prime': dgp.g_wiggly_prime, 'g_dgp_prime_beta':dgp.g_wiggly_prime_beta,
                                   'g_hyper_nn': {'layers': (80,)}}
g_functions['Pointy']    = {'g_name': 'Pointy',       'g_dgp': dgp.g_pointy, 'g_parameters': dgp.g_pointy_pars,
                                   'g_dgp_prime': dgp.g_pointy_prime, 'g_dgp_prime_beta':dgp.g_pointy_prime_beta,
                                   'g_hyper_nn': {'layers': (80,)}}
g_functions['Trigpol_3']    = {'g_name': 'Trig. pol (3)',   'g_dgp': dgp.g_trigpol_1, 'g_parameters': dgp.g_trigpol_par_1,
                                   'g_dgp_prime': dgp.g_trigpol_prime_1, 'g_dgp_prime_beta':dgp.g_trigpol_prime_beta_1,
                                   'g_hyper_nn': {'layers': (80,)}}
g_functions['Ackley']       = {'g_name': 'Ackley',          'g_dgp': dgp.g_ackley, 'g_parameters': dgp.g_ackley_pars,
                                   'g_dgp_prime': dgp.g_ackley_prime, 'g_dgp_prime_beta': dgp.g_ackley_prime_beta, 
                                   'g_hyper_nn': {'layers': (100,)}}
g_functions['Rastrigin']    = {'g_name': 'Rastrigin',       'g_dgp': dgp.g_rastrigin, 'g_parameters': dgp.g_rastrigin_pars,
                                   'g_dgp_prime': dgp.g_rastrigin_prime, 'g_dgp_prime_beta':dgp.g_rastrigin_prime_beta, 
                                   'g_hyper_nn': {'layers': (100,)}}
g_functions['Drop-Wave']    = {'g_name':  'Drop-Wave',      'g_dgp': dgp.g_dropwave, 'g_parameters': dgp.g_dropwave_pars,
                                   'g_dgp_prime': dgp.g_dropwave_prime, 'g_dgp_prime_beta':dgp.g_dropwave_prime_beta,
                                   'g_hyper_nn': {'layers': (100,)}}

g_function = g_functions['Linear']

################################################################################
### Define estimators
estimators = {}
estimators['MLE'] = {'name': 'MLE', 'estimator': est.estimator_mle_dgp_reg, 'est_kwargs':{'g_function': g_function}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:forest green', 'linestyle':':', 'linewidth': 2}}
estimators['OLS (I)'] = {'name': 'OLS (I)',     'estimator': est.estimator_ols, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':'--', 'linewidth': 2}}
estimators['OLS (II)'] = {'name': 'OLS (II)',     'estimator': est.estimator_ols_poly, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:royal blue', 'linestyle':'--', 'linewidth': 2}}
estimators['NN (I)']    = {'name': 'NN (I)',        'estimator': nn.estimator_nn_reg,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':'-', 'linewidth': 2},  
                      'est_kwargs': {'layers': (100,), 'activation': 'relu'}, 'bootstrapper': nn.bootstrap_estimator,
                      'mrg_kwargs': {'layers': (100,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
estimators['NN (II)']    = {'name': 'NN (II)',        'estimator': nn.estimator_nn_reg, 'fig_kwargs': {'color': 'xkcd:dark purple', 'linestyle':'-', 'linewidth': 2},
                      'est_kwargs': {'layers': (100,100), 'activation': 'relu'}, 
                      'mrg_kwargs': {'layers': (100,100), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
#estimators['NW'] = {'name': 'NW', 'estimator': est.estimator_nw, 'est_kwargs':{'reg_type': 'lc'}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:dark teal', 'linestyle':'--', 'linewidth': 2}}
#estimators['NW'] = {'name': 'NW', 'estimator': est.estimator_nw, 'est_kwargs':{'reg_t ype': 'll'}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:dark teal', 'linestyle':'--', 'linewidth': 2}}
#estimators['SP (SL)'] = {'name': 'SP (SL)', 'estimator': est.estimator_semiparametric_semilinear, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:dark teal', 'linestyle':'--', 'linewidth': 2}}
#estimators['SP (SI)'] = {'name': 'SP (SI)', 'estimator': est.estimator_semiparametric_singleindex, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:dark teal', 'linestyle':'--', 'linewidth': 2}}
##IV estimators (naive)
#estimators['2SLS'] = {'name': '2SLS',     'estimator': est.estimator_2sls_ols, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':'-', 'linewidth': 2}}
#estimators['NN-NN']    = {'name': '2SNN',        'estimator': nn.estimator_2sls_nn,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':'-', 'linewidth': 2},  
#                      'est_kwargs': {'layers': (100,), 'activation': 'relu'}, 'bootstrapper': nn.bootstrap_estimator,
#                      'mrg_kwargs': {'layers': (100,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
#estimators['NN (I)']['fig_kwargs']['linestyle'] = '--' #If IV, redefine slightly 
#estimators['NN-OLS']    = {'name': 'NN-OLS',        'estimator': nn.estimator_2sls_nn_ols,  'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':':', 'linewidth': 2},  
#                      'est_kwargs': {'layers': (100,), 'activation': 'relu'}, 'bootstrapper': nn.bootstrap_estimator,
#                      'mrg_kwargs': {'layers': (100,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
#
#estimators['OLS-NN']    = {'name': 'NN-NN',        'estimator': nn.estimator_2sls_ols_nn,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':':', 'linewidth': 2},  
#                      'est_kwargs': {'layers': (100,), 'activation': 'relu'}, 'bootstrapper': nn.bootstrap_estimator,
#                      'mrg_kwargs': {'layers': (100,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
##IV estimators (control function)
#estimators['2SLS'] = {'name': '2SLS',     'estimator': est.estimator_2sls_ols_control, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':'-', 'linewidth': 2}}
#estimators['2SNN']    = {'name': '2SNN',        'estimator': nn.estimator_2sls_nn_control,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':'-', 'linewidth': 2},  
#                      'est_kwargs': {'layers': (100,), 'activation': 'relu'}, 'bootstrapper': nn.bootstrap_estimator,
#                      'mrg_kwargs': {'layers': (100,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
#estimators['NN (I)']['fig_kwargs']['linestyle'] = '--' #If IV, redefine slightly 
#estimators['NN-OLS']    = {'name': 'NN-OLS',        'estimator': nn.estimator_2sls_nn_ols_control,  'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':':', 'linewidth': 2},  
#                      'est_kwargs': {'layers': (100,), 'activation': 'relu'}, 'bootstrapper': nn.bootstrap_estimator,
#                      'mrg_kwargs': {'layers': (100,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
#
#estimators['OLS-NN']    = {'name': 'NN-NN',        'estimator': nn.estimator_2sls_ols_nn_control,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':':', 'linewidth': 2},  
#                      'est_kwargs': {'layers': (100,), 'activation': 'relu'}, 'bootstrapper': nn.bootstrap_estimator,
#                      'mrg_kwargs': {'layers': (100,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}

#if 'g_hyper_nn' in g_function.keys(): 
#    if 'NN (I)' in estimators: 
#        estimators['NN (I)']['est_kwargs'].update(g_function['g_hyper_nn'])
#        estimators['NN (I)']['mrg_kwargs'].update(g_function['g_hyper_nn'])
#    if 'NN (II)' in estimators: 
#        estimators['NN (II)']['est_kwargs'].update(g_function['g_hyper_nn'])
#        estimators['NN (II)']['mrg_kwargs'].update(g_function['g_hyper_nn'])
#        #2layer net simply replicates first layer again, increasing representational capacity
#        estimators['NN (II)']['est_kwargs']['layers'] = 2*estimators['NN (II)']['est_kwargs']['layers'] #2 layers
#        estimators['NN (II)']['mrg_kwargs']['layers'] = 2*estimators['NN (II)']['mrg_kwargs']['layers'] #2 layers


##########################################################################
#RUN MC SIMULATION 
if __name__ == '__main__':
    __spec__ = None
    if parameters['run_wellspecified'] == False: 
        if parameters['B'] > 0:
            res_betahats, res_mrgeffs, data, res_probs, res_boot_expects, res_boot_mrgeffs = \
                mc.MC_simulate(parameters, estimators, g_function)            
        else: 
            res_betahats, res_mrgeffs, data, res_probs = \
                mc.MC_simulate(parameters, estimators, g_function)
        res_mrgeffs_obs = res_mrgeffs
    else: 
        res_betahats, res_mrgeffs, data, res_probs, res_betahats_obs, res_mrgeffs_obs, res_probs_obs = \
                mc.MC_simulate(parameters, estimators, g_function)
    print('Finished simulation.')
    
    ##import os, pickle
    #with open(os.getcwd() + '\\simulation_results\\'+'res_betahats_V1.txt', "rb") as f: 
    #    #data2 = f.read(pickle.loads(data))
    #    data2 = pickle.loads(f.read())
        
    ##Visualize the first few variables of one of the runs
    #smr.visualize_run(data=data, run=0)
#    figz.fig_wrapper(figurefunc=figz.fig_visualize_run, series=data)
    
#    #Accuracy
#    res_yhats = smr.comp_wrapper_model(smr.comp_predict_from_probability, res_probs)
#    res_yhats = smr.comp_wrapper_addmodel(function=smr.add_mode, 
#                                          model_series=res_yhats, y=data)
#    res_accs = smr.comp_wrapper_model(smr.comp_accmeasures_from_prediction, 
#                                      res_yhats, y=data)
#    tblz.tables_accuracy(accs = res_accs, decimals = parameters['decimals'], 
#                        print_string=True, save_file=False)
#    
##    #Average marginal effects
    res_mrgeffs_avg = smr.comp_wrapper_model(smr.comp_average, res_mrgeffs, comp_kws = {'coefficient':0})
    res_mrgeffs_avg_obs = smr.comp_wrapper_model(smr.comp_average, res_mrgeffs_obs, comp_kws = {'coefficient':0})
##    figz.plot_distribution(res_mrgeffs_avg) #Consistency of marginal effect figure
    figz.fig_wrapper(figurefunc=figz.fig_distribution, series=res_mrgeffs_avg, 
                     estimators=estimators, split='Test' )
    figz.fig_wrapper(figurefunc=figz.fig_distribution, series=res_mrgeffs_avg_obs, 
                     estimators=estimators, split='Test')
#    tblz.tables_avgmargeff(mrgeffs_avg = res_mrgeffs_avg, decimals = parameters['decimals'], 
#                          print_string=True, save_file=False)
#    avg, avg_obs = {}, {}
#    avg[g_function['g_name']] = res_mrgeffs_avg
#    avg_obs[g_function['g_name']] = res_mrgeffs_avg_obs
#    tblz.table_wrapper_g_double(g_series1 = avg, 
#                                g_series2 = avg_obs, cell_function =tblz.table_cell_avgstd, 
#                        g_names=g_functions, decimals=parameters['decimals'], print_string=True, 
#                        models = estimators.keys(), #split1='Train', title1 = 'Train', split2='Test', title2='Test',
#                        )  
#    tblz.table_wrapper_g_double(g_series1 = avg_obs, 
#                                g_series2 = avg_obs, cell_function =tblz.table_cell_avgstd, 
#                        g_names=g_functions, decimals=parameters['decimals'], print_string=True, 
#                        models = estimators.keys(), split1='Train', title1 = 'Train', split2='Test', title2='Test',
#                        )  
#    res_mrgeffs_rmse = smr.comp_wrapper_model(smr.comp_rmse, res_mrgeffs, dgp_series = res_mrgeffs)
#    res_mrgeffs_rmse_obs = smr.comp_wrapper_model(smr.comp_rmse, res_mrgeffs_obs, dgp_series = res_mrgeffs_obs, 
#                                                  comp_kws = {'coefficient':0})
#    res_mrgeffs_mrmse = smr.comp_wrapper_model(smr.comp_mrmse, res_mrgeffs, dgp_series = res_mrgeffs)
#    res_mrgeffs_mrmse_obs = smr.comp_wrapper_model(smr.comp_mrmse, res_mrgeffs_obs, dgp_series = res_mrgeffs_obs)
#    tblz.tables_avgmargeff(mrgeffs_avg = res_mrgeffs_mrmse, decimals = parameters['decimals'], 
#                          print_string=True, save_file=False, filename='MARGINAL EFFFECTS (RMSE)')
#    tblz.tables_avgmargeff(mrgeffs_avg = res_mrgeffs_mrmse_obs, decimals = parameters['decimals'], 
#                          print_string=True, save_file=False, filename='MARGINAL EFFFECTS (RMSE)')
#    res_mrgeffs_me = smr.comp_wrapper_model(smr.comp_me, res_mrgeffs, dgp_series = res_mrgeffs )
#    res_mrgeffs_mme = smr.comp_wrapper_model(smr.comp_mme, res_mrgeffs, dgp_series = res_mrgeffs )
#    
    #RMSE for probability 
#    res_probs_rmse = smr.comp_wrapper_model(smr.comp_rmse, res_probs, dgp_series =res_probs )
    
    # G table
    res_mrgeffs_rmse, res_mrgeffs_rmse_obs  = {}, {}
    res_mrgeffs_rmse[g_function['g_name']] = smr.comp_wrapper_model(smr.comp_rmse, res_mrgeffs, dgp_series = res_mrgeffs, 
                                                  comp_kws = {'coefficient':0})
    res_mrgeffs_rmse_obs[g_function['g_name']] = smr.comp_wrapper_model(smr.comp_rmse, res_mrgeffs_obs, dgp_series = res_mrgeffs_obs, 
                                                  comp_kws = {'coefficient':0})
#    tblz.table_wrapper_g(g_series = res_mrgeffs_rmse_obs, cell_function =tblz.table_cell_avgstd, 
#                        g_names=g_functions, decimals=parameters['decimals'], print_string=True, 
#                        models = estimators.keys(),
#                        )    
    tblz.table_wrapper_g_double(g_series1 = res_mrgeffs_rmse, 
                                g_series2 = res_mrgeffs_rmse_obs, cell_function =tblz.table_cell_avgstd, 
                        g_names=g_functions, decimals=parameters['decimals'], print_string=True, 
                        models = estimators.keys(), #split1='Train', title1 = 'Train', split2='Test', title2='Test',
                        )    
    
    
#    res_mrgeffs_attfactor, res_mrgeffs_attfactor_obs  = {}, {}
#    res_mrgeffs_attfactor[g_function['g_name']] = smr.comp_wrapper_model(smr.comp_attenuationfactor_mean, 
#                                                 res_mrgeffs, dgp_series = res_mrgeffs, 
#                                                  )#comp_kws = {'coefficient':0})    
#    res_mrgeffs_attfactor_obs[g_function['g_name']] = smr.comp_wrapper_model(smr.comp_attenuationfactor_mean, 
#                                                 res_mrgeffs_obs, dgp_series = res_mrgeffs_obs, 
#                                                  )#comp_kws = {'coefficient':0})        
#    tblz.table_wrapper_g_double(g_series1 = res_mrgeffs_attfactor, 
#                                g_series2 = res_mrgeffs_attfactor_obs, cell_function =tblz.table_cell_avgstd, 
#                        g_names=g_functions, decimals=parameters['decimals'], print_string=True, 
#                        models = estimators.keys(), #split1='Train', title1 = 'Train', split2='Test', title2='Test',
#                        )      
    # G table avg
#    res_mrgeffs_rse_obs_avg  = {}
#    res_mrgeffs_rse_obs_avg[g_function['g_name']] = smr.comp_wrapper_model(smr.comp_rse_avg, res_mrgeffs_obs, dgp_series = res_mrgeffs_obs, 
#                                                  comp_kws = {'coefficient':0})
#    tblz.table_wrapper_g(g_series = res_mrgeffs_rse_obs_avg, cell_function =tblz.table_cell_avgstd, 
#                        g_names=g_functions, decimals=parameters['decimals'], print_string=True, 
##                        models = ['DGP', 'MLE', 'OLS (I)', 'NN (I)'],
#                        models = estimators.keys(),
#                        )    
    
    # Bootstrap test
#    importlib.reload(smr)
#    res_boot_mrgeffs_avg = smr.comp_wrapper_model(smr.comp_boot_average, res_boot_mrgeffs, comp_kws = {'coefficient':2})
#    res_boot_mrgeffs_avg_std0 = {}
#    res_boot_mrgeffs_avg_std0[g_function['g_name']]= smr.comp_wrapper_model(smr.comp_boot_average_sdev, res_boot_mrgeffs, 
#                                                     comp_kws = {'coefficient':0})
#    res_boot_mrgeffs_avg_std2 = {}
#    res_boot_mrgeffs_avg_std2[g_function['g_name']]= smr.comp_wrapper_model(smr.comp_boot_average_sdev, res_boot_mrgeffs, 
#                                                     comp_kws = {'coefficient':2})
#    
#    res_mrgeffs_avg0 = {}
#    res_mrgeffs_avg0[g_function['g_name']] = smr.comp_wrapper_model(smr.comp_average, res_mrgeffs, 
#                                            comp_kws = {'coefficient':0})
#    res_mrgeffs_avg2 = {}
#    res_mrgeffs_avg2[g_function['g_name']] = smr.comp_wrapper_model(smr.comp_average, res_mrgeffs, 
#                                            comp_kws = {'coefficient':2})
#    importlib.reload(tblz)
#    tblz.table_wrapper_g_double(g_series1 = res_boot_mrgeffs_avg_std0, 
#                                g_series2 = res_boot_mrgeffs_avg_std2, 
#                                extra_series1=res_mrgeffs_avg0,
#                                extra_series2=res_mrgeffs_avg2,
#                                cell_function =tblz.table_cell_avg_extrastdev, 
#                        g_names=g_functions, decimals=parameters['decimals'], print_string=True, 
#                        models = estimators.keys(), #split1='Train', title1 = 'Train', split2='Test', title2='Test',
#                        title1 = '$x_{0}$', title2 = '$v_{0}$',
#                        )   

print('Script execution time:', str(datetime.now()-time_start).strip("0").strip(":00"))
del time_start 