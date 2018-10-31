# Preliminaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from datetime import datetime
import warnings

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
sns.set_style('whitegrid')

# Parameters
parameters = {} #Define estimation parameters
parameters['seed'] = 1033
np.random.seed(parameters['seed']) #Also reseed at data split
parameters['standardize_features'] = True
parameters['standardize_label'] = False #Logarithm already numerically stable #Science
parameters['B'] = 99  # Bootstrap iterations (0 for no bootstrap)
parameters['bootstrap_parallel'] = False  
parameters['bootstrap_averages'] = True # Bootstrap only produces average of each for memory reasons. 
parameters['honest_estimation'] = True
parameters['honest_estimators'] = ['NN', 'NN (I)', 'NN (II)', '2SNN']
parameters['grid_search'] = False
parameters['grid_repetitions'] = 100 

time_start = datetime.now()
print('Runtime: ', datetime.now() - time_start, '. Began running the code. Start time: ', time_start, sep='')
from sklearn.exceptions import DataConversionWarning 
warnings.filterwarnings("ignore", category=DataConversionWarning) 


if __name__ == '__main__': #Begin program
    ################################################################################
    ################################################################################
    ################################################################################
    ### Data processing
    
    ################################################################################
    # Import and preprocessing
    #Largely copy of their statafile from https://economics.mit.edu/faculty/angrist/raw_data1/raw_data/angkru1991
    raw_data = pd.read_stata(r'data/NEW7080.dta') 
    raw_data = raw_data.rename(columns={'v1': 'AGE', 
                                'v2': 'AGEQ',
                                'v4': 'EDUC',
                                'v5': 'ENOCENT',
                                'v6': 'ESOCENT',
                                'v9': 'LWKLYWGE',
                                'v10': 'MARRIED',
                                'v11': 'MIDATL',
                                'v12': 'MT',
                                'v13': 'NEWENG',
                                'v16': 'CENSUS',
                                'v18': 'QOB',
                                'v19': 'RACE',
                                'v20': 'SMSA',
                                'v21': 'SOATL',
                                'v24': 'WNOCENT',
                                'v25': 'WSOCENT',
                                'v27': 'YOB',
                                })
    raw_data.drop('v8', inplace=True, axis=1)
    
    raw_data['COHORT'] = '20.29'
    raw_data.loc[(raw_data['YOB']<=39) & (raw_data['YOB']>=30), 'COHORT'] = '30.39'
    raw_data.loc[(raw_data['YOB']<=49) & (raw_data['YOB']>=40), 'COHORT'] = '40.49'
    
    raw_data.loc[(raw_data['CENSUS']==80), 'AGEQ'] = raw_data['AGEQ']-1900
    raw_data['AGEQSQ'] = raw_data['AGEQ']*raw_data['AGEQ']
    
    for i in range(0,10): #Generate YOB dummies
        raw_data['YR2'+str(i)] = 0
        raw_data.loc[(raw_data['YOB']==1920+i), 'YR2'+str(i)] = 1
        raw_data.loc[(raw_data['YOB']==30+i), 'YR2'+str(i)] = 1
        raw_data.loc[(raw_data['YOB']==40+i), 'YR2'+str(i)] = 1
    
    for i in range (1,5): #Generate QOB dummies
        raw_data['QTR'+str(i)] = 0
        raw_data.loc[(raw_data['QOB']== i), 'QTR'+str(i)] = 1
        
    for j in range(1,4): 
        for i in range (0,10): 
            raw_data['QTR'+str(j)+str(20+i)] = raw_data['QTR'+str(j)]*raw_data['YR2'+str(i)]
    del i,j 
    
    ################################################################################
    # Prepare actual data (and filter)
    data = {}
    
    # Data filter
    #data_filter = raw_data['COHORT']=='20.29'
    data_filter = raw_data['COHORT']=='30.39'
    #data_filter = raw_data['COHORT']=='40.49'
    #data_filter = raw_data['COHORT'] != 'Everyone'
    
    # Model specifications
    #y = ['LWKLYWGE']
    xs = {}
    xs['Basic'] = ['YR2'+str(i) for i in range(0,9)]
    xs['w/age'] = ['YR2'+str(i) for i in range(0,9)]+['AGEQ', 'AGEQSQ']
    xs['w/dummies'] = ['YR2'+str(i) for i in range(0,9)] + ['RACE', 'MARRIED', 'SMSA', 'NEWENG', 'MIDATL', 
                                             'ENOCENT', 'WNOCENT', 'SOATL', 'ESOCENT', 'WSOCENT', 'MT']
    xs['Full'] = ['YR2'+str(i) for i in range(0,9)] + ['RACE', 'MARRIED', 'SMSA', 'NEWENG', 'MIDATL', 
                                             'ENOCENT', 'WNOCENT', 'SOATL', 'ESOCENT', 'WSOCENT', 'MT', 'AGEQ', 'AGEQSQ']
    #z =  ['QTR'+str(j)+str(20+i) for j in range(1,4) for i in range(0,10)]
    
    # Prepare in desired format
    for specification in xs.keys(): 
        data[specification] = {}
        data[specification]['y'], data[specification]['x'], data[specification]['z'] = {}, {}, {}
        data[specification]['y']['Train'] = raw_data.loc[data_filter, 'LWKLYWGE']
        data[specification]['x']['Train'] = raw_data.loc[data_filter, ['EDUC']+xs[specification]]#.dropna(inplace=True)
        data[specification]['z']['Train'] = raw_data.loc[data_filter, 
                                            ['QTR'+str(j)+str(20+i) for j in range(1,4) for i in range(0,10)]]
    del specification,data_filter, raw_data
    
    ################################################################################
    ### Standardize
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    if parameters['standardize_label'] == True: 
        scaler_y = {}
    if parameters['standardize_features'] == True: 
        scaler_x = {}
    
    temp_data = {}
    for spec in xs.keys(): 
        #Optionally standardize 
        temp_data[spec] = {}
        temp_data[spec]['x'], temp_data[spec]['y'], temp_data[spec]['z'] = {},{},{},
        if parameters['standardize_label'] == True: 
    #        temp_data[spec] = {}
    #        temp_data[spec]['y'] = {}
            scaler_y[spec] = StandardScaler().fit(data[spec]['y']['Train'].values.reshape(-1,1))
            temp_data[spec]['y']['Train'] = pd.DataFrame(scaler_y[spec].transform(data[spec]['y']['Train'].values.reshape(-1,1)))
        else: 
            temp_data[spec]['y'] = data[spec]['y'].copy()
        if parameters['standardize_features'] == True: 
    #        temp_data[spec]['x'] = {}
            scaler_x[spec] = StandardScaler().fit(data[spec]['x']['Train'])
            temp_data[spec]['x']['Train'] = pd.DataFrame(scaler_x[spec].transform(data[spec]['x']['Train']))
        else: 
            temp_data[spec]['x']['Train'] = data[spec]['x']['Train'].copy()
        
        # No standardization of instruments (all dummies)
        temp_data[spec]['z']['Train'] = data[spec]['z']['Train'].copy()
    
    ##############################################################################
    ### Honest data splitting
    
    np.random.seed(parameters['seed']) #Reseed right before data split, to ensure same split. 
    for spec in xs.keys():
        # If some estimators are 'honest', split data accordingly
        if parameters['honest_estimation'] == True:
            # Split data
            data[spec]['y']['Estimation_split'], data[spec]['y']['Hyper_split'], \
            temp_data[spec]['y']['Estimation_split'], temp_data[spec]['y']['Hyper_split'], \
            data[spec]['x']['Estimation_split'], data[spec]['x']['Hyper_split'], \
            temp_data[spec]['x']['Estimation_split'], temp_data[spec]['x']['Hyper_split'], \
            data[spec]['z']['Estimation_split'], data[spec]['z']['Hyper_split'], \
            temp_data[spec]['z']['Estimation_split'], temp_data[spec]['z']['Hyper_split'], \
                = train_test_split(
                        data[spec]['y']['Train'],
                        temp_data[spec]['y']['Train'],
                        data[spec]['x']['Train'], 
                        temp_data[spec]['x']['Train'],
                        data[spec]['z']['Train'], 
                        temp_data[spec]['z']['Train'],
                        test_size = 0.3, shuffle = True)
                
    # Helper function which finds data for honest and dishonest estimators respectively. 
    def get_data(data, estimator, parameters):
        output = {}
        output['y'], output['x'], output['z'] = {},{},{}
        if parameters['honest_estimation'] == True and estimator in parameters['honest_estimators']:        
            output['y']['Train'], output['x']['Train'], output['z']['Train'] = \
                data['y']['Estimation_split'], data['x']['Estimation_split'], data['z']['Estimation_split']
        else: 
            output['y']['Train'], output['x']['Train'], output['z']['Train'] = \
                data['y']['Train'], data['x']['Train'], data['z']['Train']
        return output
     
    ################################################################################
    ################################################################################
    ################################################################################
    ### Grid search    
    
    if parameters['grid_search'] == True:     
        print('Runtime:', datetime.now() - time_start, '. Began grid search for neural network.')
        
        # Grid to be searched.
        from itertools import product
        params = {'hidden_layer_sizes': [i for i in range(60,300+1,10)]+\
                                 [i for i in product(range(100,200+1,25),range(25,150+1,25))], # Initial grid
              'alpha': [10**-i for i in range(1,5)]+[0.2,0.3, 0.4, 0.5, 1]
              }
    #    print(params)
        
        # Define a basic network to serach over
        from sklearn.neural_network import MLPRegressor
        basic_nn = MLPRegressor( 
                                activation='relu', solver='adam',  
                                learning_rate_init = 10**-3,
                                early_stopping =False, 
                                tol = 10**-4, 
                                max_iter=100) 
        
        from sklearn.model_selection import RandomizedSearchCV
        if __name__ == '__main__':
            grid_results = pd.DataFrame(RandomizedSearchCV(basic_nn, params, 
                                            scoring='neg_mean_squared_error', 
                                            cv = 3, refit = False, return_train_score=False, 
        #                                    fit={'callbacks': callbacks},
                                            n_jobs = -1, 
                                            n_iter=parameters['grid_repetitions']).\
                                                fit(temp_data['Full']['x']['Hyper_split'],
                                                    np.ravel(temp_data['Full']['y']['Hyper_split'])).\
                                            cv_results_)
        
        grid_results.to_csv('tables/grid_results.csv')
        print('Runtime:', datetime.now() - time_start, '. Finished grid search for neural network.')
    
        
        del params, basic_nn 
        
    ################################################################################
    ################################################################################
    ################################################################################
    ### Estimation
    
    ################################################################################
    ### Define estimators 
    estimators = {}
    layers = (100,75)
    alpha = 0.2
    estimators['OLS (I)'] = {'name': 'OLS (I)',     'estimator': est.estimator_ols, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':'--', 'linewidth': 3}}
    estimators['OLS (II)'] = {'name': 'OLS (II)',     'estimator': est.estimator_ols_poly, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:royal blue', 'linestyle':'--', 'linewidth': 2}}
    estimators['NN']    = {'name': 'NN',        'estimator': nn.estimator_nn_reg,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':'-', 'linewidth': 2},  
                          'est_kwargs': {'layers': layers, 'activation': 'relu', 'alpha':alpha}, 'bootstrapper': nn.bootstrap_estimator,
                          'mrg_kwargs': {'layers': layers, 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
    #estimators['NN (II)']    = {'name': 'NN (II)',        'estimator': nn.estimator_nn_reg, 'fig_kwargs': {'color': 'xkcd:dark purple', 'linestyle':'--', 'linewidth': 2},
    #                      'est_kwargs': {'layers': (200,100), 'activation': 'relu', 'alpha':10**-4}, 'bootstrapper': nn.bootstrap_estimator, 
    #                      'mrg_kwargs': {'layers': (200,100), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
    ####IV estimators (control function)
    ###estimators['2SLS_1'] = {'name': '2SLS_1',     'estimator': est.estimator_2sls_ols, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':'-', 'linewidth': 2}}
    estimators['2SLS'] = {'name': '2SLS',     'estimator': est.estimator_2sls_ols_control, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':'-', 'linewidth': 3}}
    estimators['2SNN']    = {'name': '2SNN',        'estimator': nn.estimator_2sls_nn_control,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':'-', 'linewidth': 2},  
                          'est_kwargs': {'layers': layers, 'activation': 'relu', 'alpha':alpha}, 'bootstrapper': nn.bootstrap_estimator,
                          'mrg_kwargs': {'layers': layers, 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
    estimators['NN']['fig_kwargs']['linestyle'] = '--' #If IV, redefine slightly 
    #estimators['NN-OLS']    = {'name': 'NN-OLS',        'estimator': nn.estimator_2sls_nn_ols_control,  'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':':', 'linewidth': 2},  
    #                      'est_kwargs': {'layers': (200,100), 'activation': 'relu', 'alpha':10**-4}, 'bootstrapper': nn.bootstrap_estimator,
    #                      'mrg_kwargs': {'layers': (200,100), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
    #
    #estimators['OLS-NN']    = {'name': 'NN-NN',        'estimator': nn.estimator_2sls_ols_nn_control,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':':', 'linewidth': 2},  
    #                      'est_kwargs': {'layers': (200,100), 'activation': 'relu', 'alpha':10**-4}, 'bootstrapper': nn.bootstrap_estimator,
    #                      'mrg_kwargs': {'layers': (200,100), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
    del layers, alpha
    
    
    ################################################################################
    ### Estimate models
    betahat, expect, mrgeff = {}, {}, {}
    if parameters['B'] > 0:  #If no replications, bootstrap is not desired 
        boot_expect, boot_mrgeff = {}, {}
    
    for spec in xs.keys(): 
        #Prepare stuff 
        betahat[spec], expect[spec], mrgeff[spec] = {}, {},{}
        exog = np.array([False] + [True]*(len(data[spec]['x']['Train'].T)-1)) #Only first presumed endogenous
        
        #Estimate models
        for estimator in estimators:
            betahat[spec][estimator], expect[spec][estimator], mrgeff[spec][estimator] \
                            = estimators[estimator]['estimator'](get_data(temp_data[spec], estimator, parameters),
                                                                    est_kwargs= estimators[estimator]['est_kwargs'],
                                                                    mrg_kwargs=estimators[estimator]['mrg_kwargs'], 
                                                                    splits=('Train',), exog=exog)                
            # If standardized, adjust results 
            if parameters['standardize_label'] == True: 
                for split in expect[spec][estimator].keys(): 
                    expect[spec][estimator][split] = expect[spec][estimator][split]*scaler_y[spec].scale_ + scaler_y[spec].mean_
                    mrgeff[spec][estimator][split] = mrgeff[spec][estimator][split]*(scaler_y[spec].scale_/scaler_x[spec].scale_)
            elif parameters['standardize_features'] == True: 
                for split in expect[spec][estimator].keys(): 
                    mrgeff[spec][estimator][split] = mrgeff[spec][estimator][split]*(1/scaler_x[spec].scale_)
        
            print('Runtime:', datetime.now() - time_start, 'Spec: ' + spec +  '. Estimator: ' + estimator + '. Finished estimation.')
        
        # Bootstrap models  (optional)
        if parameters['B'] > 0:  #If no replications, bootstrap is not desired 
            boot_expect[spec], boot_mrgeff[spec] = {}, {}
            for estimator in estimators:
                if __name__ == '__main__':
                    boot_expect[spec][estimator], boot_mrgeff[spec][estimator] \
                        = nn.bootstrap_estimator_v2(estimator=estimators[estimator],
                                                    data=get_data(temp_data[spec], estimator, parameters),
                                                    B=parameters['B'], seed=parameters['seed'],
                                                    parallel = parameters['bootstrap_parallel'], 
                                                    splits=('Train',),exog=exog,
                                                    get_averages=parameters['bootstrap_averages'])
                
                # If standardized, adjust results 
                if parameters['standardize_label'] == True: 
                    for split in boot_expect[spec][estimator].keys(): 
                        boot_expect[spec][estimator][split] = boot_expect[spec][estimator][split]*scaler_y[spec].scale_ + scaler_y[spec].mean_
                        boot_mrgeff[spec][estimator][split] = boot_mrgeff[spec][estimator][split]*(scaler_y[spec].scale_/scaler_x[spec].scale_)
                elif parameters['standardize_features'] == True: 
                    for split in boot_expect[spec][estimator].keys(): 
                        boot_mrgeff[spec][estimator][split] = boot_mrgeff[spec][estimator][split]*(1/scaler_x[spec].scale_)
    
                print('Runtime:', str(datetime.now() - time_start), 'Spec: ' + spec +  '. Estimator: ' + estimator + '. Finished bootstrapping.')
    
    del estimator, exog, split, temp_data 
    
    ################################################################################
    ################################################################################
    ################################################################################
    ### Summaries 
    
    ################################################################################
    ### Calculate summaries 
    def comp_summary(output, summary_function, specs=xs, estimators=estimators, **kwargs): 
        comp = {}
        for spec in specs.keys(): 
            comp[spec] = {}
            for estimator in estimators:
                comp[spec][estimator] = {}
                for split in ('Train',):
                    comp[spec][estimator][split] =  summary_function(output[spec][estimator][split], **kwargs)
        return comp 
    
    # Average marginal effects
    def smr_avg_mrgeff(mrgeffs, **kwargs): 
        return np.mean(mrgeffs, axis=0)[0]
                
    mrgeff_avg = comp_summary(output=mrgeff, summary_function = smr_avg_mrgeff)            
    
    # Bootstap stuff
    if parameters['B'] > 0:  
        def smr_bootstrap_expect_avg(boot_expects, **kwargs): 
            return np.nanmean(boot_expects, axis=0)
        def smr_bootstrap_mrgeffs_avg(boot_mrgeffs, **kwargs): 
            return np.nanmean(boot_mrgeffs, axis=0)[:,0]             
        def smr_bootstrap_mrgeffs(boot_mrgeffs, **kwargs): 
            return boot_mrgeffs[:,0]
        
        # Calculate
        if parameters['bootstrap_averages'] == False:             
            boot_expect_avg = comp_summary(output=boot_expect, summary_function = smr_bootstrap_expect_avg)            
            boot_mrgeff_avg = comp_summary(output=boot_mrgeff, summary_function = smr_bootstrap_mrgeffs_avg)
        else: # Averages done in estimation
            boot_expect_avg = boot_expect
            boot_mrgeff_avg = comp_summary(output=boot_mrgeff, summary_function = smr_bootstrap_mrgeffs)
    
    else: 
        boot_expect_avg = est.dd_inf()
        boot_mrgeff_avg = est.dd_inf()
            
    #Mrgeff data 
    mrgeff_data = {}
    
    def comp_mrgeffdata(output, data, variable_iloc=0, specs=xs, estimators=estimators, parameters=parameters, **kwargs):
        comp = {}
        for spec in specs.keys(): 
            comp[spec] = {}
            for estimator in estimators:
                comp[spec][estimator] = {}
                for split in ('Train',):
                    comp[spec][estimator][split] =  \
                        pd.concat((get_data(data[spec], estimator, parameters)['x']['Train'].reset_index(drop='index').iloc[:,variable_iloc],
                           pd.DataFrame(output[spec][estimator][split]).iloc[:,0]), axis=1, ignore_index = True)
        return comp 
        
    
    mrgeff_data['EDUC'] = comp_mrgeffdata(output=mrgeff, data = data, variable_iloc=0)
    #mrgeff_data['AGE'] = comp_mrgeffdata(output=mrgeff, data = data, variable_iloc=-2)
    
    ################################################################################
    ### Table with average marginal effects 
    importlib.reload(tblz)
    tblz.table_wrapper_g(g_series = mrgeff_avg, 
                            cell_function =tblz.table_cell_regoutput_3line_confinterval, 
                            extra_series = boot_mrgeff_avg, 
                            decimals=3, print_string=True, 
                            models = estimators.keys(),split = 'Train',
                            save_file = True, filename = 'tbl_case_mrgeff_avg',
                            cell_writer = tblz.write_cells_3line,
                            label = 'tbl_case_mrgeff_avg',
                            caption = 'Estimates of the average return to schooling',
                            note1 = '\\AngristTable{'+ '{:,}'.format(parameters['B'])+'}',
                            ) 
    
    ################################################################################
    ### Figure with marginal effects (distribution & heterogeneous returns)
    figz.fig_wrapper(figurefunc=figz.fig_distribution, series=mrgeff['Full'], 
                         series_extra = mrgeff_data['EDUC']['Full'], 
    #                     models2 = ['2SNN'],  
                         figurefunc_extra=figz.fig_plot_mrgeff_grpby,              
                         estimators=estimators, split='Train', #legend='figure'
    #                     models = ['OLS (I)', 'NN (I)', '2SLS', '2SNN'],
                         titles = ['Distribution of marginal effect','Differences in marginal effects'],
                         xlabel = 'Marginal effect of education', xlabel2 = 'Years of education.',
                         ylabel = 'Density', ylabel2= 'Marginal effect of education', 
                         legend = 'first',
                         #ymax =[25, 0.1],
                         #ymin = [0.1,0],
                        save_file = True, filename = 'case_mrgeff_comparison_1'
                         )