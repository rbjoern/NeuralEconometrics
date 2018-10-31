# IMPORT DEPENDENCIES
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

#Import own files 
import sys 
sys.path.append(r'..\functions')
import monte_carlo_simulation as mc
import dgp_stuff as dgp
import neural_net as nn
import estimators as est
import summaries as smr
import figurez as figz
import tablez as tblz

##############################################################################
# DEFINE PARAMETERS
parameters = {}
#Generel set-up
parameters['seed'] = 4256               #Seed for pseudo-random draws            
parameters['M'] = 100                 #Number of simulation repetitions
parameters['k'] = 5                       #Number of regressors
parameters['n'] = 10**5             #Number of observations 
parameters['V'] = 0                     # Number of irrelevant variables
parameters['C'] = 0                     #Number of confounding variables 
parameters['Z'] = 0                     # Number of instruments. 
np.random.seed(parameters['seed'])      #Also restarts at seed in each simulation
parameters['run_wellspecified'] = False 
parameters['add_error'] = False

#Technical stuff
parameters['parallel'] = True           # Simulations run parallel or serial
parameters['decimals'] = 2              #Decimals in output tables 
parameters['reduce_size'] = True       #Saves simulations as 16 insted of 64
parameters['save_file'] = True       #Save results in files 
parameters['filename'] = 'sec4_simulation_single_prod_' +  str(datetime.date(datetime.now())).replace('-','_')
parameters['start_time']= datetime.now() #Time measure used for status emssages

#Specifications for how data is drawn
parameters['beta_distribution'] = dgp.draw_beta_normal      #Draw betas from this distribution
parameters['beta_mean'], parameters['beta_scale'] = 1, 1    #Mean and scale for beta distribution
parameters['redraw'] = False                                #Redraw beta in each simulation?
parameters['x_distribution']= dgp.draw_x_normal             #Draw x from this distribution
#parameters['x_distribution']= dgp.draw_x_normal_iv
#parameters['x_distribution']= dgp.draw_x_normal_iv_v2
parameters['x_distribution_parameters'] = dgp.gen_x_normal_unitvariance_randommean #Draw mean and correlation for x
#parameters['x_distribution_parameters'] = dgp.gen_x_normal_unitvariance_randommean_iv
parameters['x_mean']= 0                                     #Average expected value of x variables. 
parameters['u_distribution'] = dgp.draw_u_normal             #Define error term for y. 
parameters['u_scale'] = 1 
parameters['y_generate'] = dgp.gen_y_reg                 # Compute y from error and g. 
parameters['y_squashing'], parameters['y_squashing_prime'] =  dgp.linear_output, dgp.linear_output_prime


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
                                   'g_dgp_prime': dgp.g_dropwave_prime,'g_dgp_prime_beta':dgp.g_dropwave_prime_beta,
                                   'g_hyper_nn': {'layers': (100,)}}

################################################################################
### Define estimators
estimators = {}
estimators['MLE'] = {'name': 'MLE', 'estimator': est.estimator_mle_dgp_reg, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:forest green', 'linestyle':':', 'linewidth': 2}}
estimators['OLS (I)'] = {'name': 'OLS (I)',     'estimator': est.estimator_ols, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':'--', 'linewidth': 2}}
estimators['OLS (II)'] = {'name': 'OLS (II)',     'estimator': est.estimator_ols_poly, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:royal blue', 'linestyle':'--', 'linewidth': 2}}
estimators['NN (I)']    = {'name': 'NN (I)',        'estimator': nn.estimator_nn_reg,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':'-', 'linewidth': 2},  
                      'est_kwargs': {'layers': (100,), 'activation': 'relu'}, 
                      'mrg_kwargs': {'layers': (100,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
estimators['NN (II)']    = {'name': 'NN (II)',        'estimator': nn.estimator_nn_reg, 'fig_kwargs': {'color': 'xkcd:dark purple', 'linestyle':'-', 'linewidth': 2},
                      'est_kwargs': {'layers': (100,100), 'activation': 'relu'}, 
                      'mrg_kwargs': {'layers': (100,100), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
##estimators['NW'] = {'name': 'NW', 'estimator': est.estimator_nw, 'est_kwargs':{'reg_type': 'lc'}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:dark teal', 'linestyle':'--', 'linewidth': 2}}
##estimators['NW'] = {'name': 'NW', 'estimator': est.estimator_nw, 'est_kwargs':{'reg_t ype': 'll'}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:dark teal', 'linestyle':'--', 'linewidth': 2}}
##estimators['SP (SL)'] = {'name': 'SP (SL)', 'estimator': est.estimator_semiparametric_semilinear, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:dark teal', 'linestyle':'--', 'linewidth': 2}}
##estimators['SP (SI)'] = {'name': 'SP (SI)', 'estimator': est.estimator_semiparametric_singleindex, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:dark teal', 'linestyle':'--', 'linewidth': 2}}
##IV estimators 
#estimators['2SLS'] = {'name': '2SLS',     'estimator': est.estimator_2sls, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':'-', 'linewidth': 2}}
#estimators['NN-NN']    = {'name': 'NN-NN',        'estimator': nn.estimator_2sls_nn,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':'-', 'linewidth': 2},  
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



##########################################################################
### Run simulations and save results to files. 
if __name__ == '__main__':
    __spec__ = None
    res_betahats, res_mrgeffs, data, res_probs = \
            mc.MC_simulate_dgps_indfiles(parameters, estimators, g_functions) 
    

print('Script execution time:', str(datetime.now()-parameters['start_time']).strip("0").strip(":00"))