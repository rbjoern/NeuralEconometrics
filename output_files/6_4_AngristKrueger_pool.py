import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from datetime import datetime

#Import own files 
import sys 
sys.path.append(r'F:\Documents\TheEnd\Code\Functions')
sys.path.append(r'C:\Users\rbjoe\Dropbox\Kugejl\10.semester\TheEnd\Code\Functions')
sys.path.append(r'C:\Users\okorasb\Documents\TheEnd\Code\Functions')
import monte_carlo_simulation as mc
import dgp_stuff as dgp
import neural_net as nn
import estimators as est
import summaries as smr
import figurez as figz
import tablez as tblz
sns.set_style('whitegrid')

# Preliminaries
np.random.seed(1033)
time_start = datetime.now()

################################################################################
################################################################################
################################################################################
### Data processing


################################################################################
# Import and preprocessing
#Largely copy of their statafile from https://economics.mit.edu/faculty/angrist/raw_data1/raw_data/angkru1991
raw_data = pd.read_stata('NEW7080.dta') 
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
# Filter data        
        
################################################################################
# Prepare actual data (and filter)
data = {}

# Data filter
#data_filter = raw_data['COHORT']=='20.29'
#data_filter = raw_data['COHORT']=='30.39'
#data_filter = raw_data['COHORT']=='40.49'
data_filter = raw_data['COHORT'] != 'Everyone'

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
################################################################################
################################################################################
### Define estimators 
estimators = {}
estimators['OLS (I)'] = {'name': 'OLS (I)',     'estimator': est.estimator_ols, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':'--', 'linewidth': 3}}
estimators['OLS (II)'] = {'name': 'OLS (II)',     'estimator': est.estimator_ols_poly, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:royal blue', 'linestyle':'--', 'linewidth': 2}}
estimators['NN (I)']    = {'name': 'NN (I)',        'estimator': nn.estimator_nn_reg,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':'-', 'linewidth': 2},  
                      'est_kwargs': {'layers': (30,), 'activation': 'relu', 'alpha':10**-4}, 'bootstrapper': nn.bootstrap_estimator,
                      'mrg_kwargs': {'layers': (30,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
estimators['NN (II)']    = {'name': 'NN (II)',        'estimator': nn.estimator_nn_reg, 'fig_kwargs': {'color': 'xkcd:dark purple', 'linestyle':'--', 'linewidth': 2},
                      'est_kwargs': {'layers': (30,30), 'activation': 'relu', 'alpha':10**-4}, 'bootstrapper': nn.bootstrap_estimator, 
                      'mrg_kwargs': {'layers': (30,30), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
####IV estimators (control function)
###estimators['2SLS_1'] = {'name': '2SLS_1',     'estimator': est.estimator_2sls_ols, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':'-', 'linewidth': 2}}
estimators['2SLS'] = {'name': '2SLS',     'estimator': est.estimator_2sls_ols_control, 'est_kwargs':{}, 'mrg_kwargs':{}, 'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':'-', 'linewidth': 3}}
estimators['2SNN']    = {'name': '2SNN',        'estimator': nn.estimator_2sls_nn_control,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':'-', 'linewidth': 2},  
                      'est_kwargs': {'layers': (30,), 'activation': 'relu', 'alpha':10**-4}, 'bootstrapper': nn.bootstrap_estimator,
                      'mrg_kwargs': {'layers': (30,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
estimators['NN (I)']['fig_kwargs']['linestyle'] = '--' #If IV, redefine slightly 
#estimators['NN-OLS']    = {'name': 'NN-OLS',        'estimator': nn.estimator_2sls_nn_ols_control,  'fig_kwargs': {'color': 'xkcd:light blue', 'linestyle':':', 'linewidth': 2},  
#                      'est_kwargs': {'layers': (30,), 'activation': 'relu', 'alpha':10**-4}, 'bootstrapper': nn.bootstrap_estimator,
#                      'mrg_kwargs': {'layers': (30,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}
#
#estimators['OLS-NN']    = {'name': 'NN-NN',        'estimator': nn.estimator_2sls_ols_nn_control,  'fig_kwargs': {'color': 'xkcd:dark orange', 'linestyle':':', 'linewidth': 2},  
#                      'est_kwargs': {'layers': (30,), 'activation': 'relu', 'alpha':10**-4}, 'bootstrapper': nn.bootstrap_estimator,
#                      'mrg_kwargs': {'layers': (30,), 'activation': nn.relu, 'activation_prime': nn.relu_prime}}


################################################################################
################################################################################
################################################################################
### Estimation
from sklearn.preprocessing import StandardScaler
#scaler_y = StandardScaler().fit(y.values.reshape(-1,1))


parameters = {}
parameters['standardize_features'] = True
parameters['standardize_label'] = False #Logarithm already numerically stable #Science
parameters['B'] = 0 # Bootstrap iterations (0 for no bootstrap)
parameters['bootstrap_averages'] = True # Bootstrap only produces average of each for memory reasons. 

################################################################################
### Estimate models
betahat, expect, mrgeff, temp_data = {}, {}, {}, {}
if parameters['B'] > 0:  #If no replications, bootstrap is not desired 
    boot_expect, boot_mrgeff = {}, {}

for spec in xs.keys(): 
    #Prepare stuff 
    betahat[spec], expect[spec], mrgeff[spec] = {}, {},{}
    exog = np.array([False] + [True]*(len(data[spec]['x']['Train'].T)-1)) #Only first presumed endogenous
    
    #Optionally standardize 
    temp_data[spec] = {}
    temp_data[spec]['x'], temp_data[spec]['y'], temp_data[spec]['z'] = {},{},{},
    if parameters['standardize_label'] == True: 
#        temp_data[spec] = {}
#        temp_data[spec]['y'] = {}
        scaler_y = StandardScaler().fit(data[spec]['y']['Train'].values.reshape(-1,1))
        temp_data[spec]['y']['Train'] = pd.DataFrame(scaler_y.transform(data[spec]['y']['Train'].values.reshape(-1,1)))
    else: 
        temp_data[spec]['y'] = data[spec]['y'].copy()
    if parameters['standardize_features'] == True: 
#        temp_data[spec]['x'] = {}
        scaler_x = StandardScaler().fit(data[spec]['x']['Train'])
        temp_data[spec]['x']['Train'] = pd.DataFrame(scaler_x.transform(data[spec]['x']['Train']))
    else: 
        temp_data[spec]['x']['Train'] = data[spec]['x']['Train'].copy()
    
    # No standardization of instruments (all dummies)
    temp_data[spec]['z']['Train'] = data[spec]['z']['Train'].copy()
    
    #Estimate models
    for estimator in estimators: 
        betahat[spec][estimator], expect[spec][estimator], mrgeff[spec][estimator] \
                    = estimators[estimator]['estimator'](temp_data[spec], est_kwargs= estimators[estimator]['est_kwargs'],
                                                                   mrg_kwargs=estimators[estimator]['mrg_kwargs'], 
                                                                   splits=('Train',), exog=exog)                
        # If standardized, adjust results 
        if parameters['standardize_label'] == True: 
            for split in expect[spec][estimator].keys(): 
                expect[spec][estimator][split] = expect[spec][estimator][split]*scaler_y.scale_ + scaler_y.mean_
                mrgeff[spec][estimator][split] = mrgeff[spec][estimator][split]*(scaler_y.scale_/scaler_x.scale_)
        elif parameters['standardize_features'] == True: 
            for split in expect[spec][estimator].keys(): 
                mrgeff[spec][estimator][split] = mrgeff[spec][estimator][split]*(1/scaler_x.scale_)
    
        print('Runtime:', datetime.now() - time_start, 'Spec: ' + spec +  '. Estimator: ' + estimator + '. Finished estimation.')
    
    # Bootstrap models 
    if parameters['B'] > 0:  #If no replications, bootstrap is not desired 
        boot_expect[spec], boot_mrgeff[spec] = {}, {}
        for estimator in estimators: 
            boot_expect[spec][estimator], boot_mrgeff[spec][estimator] \
                = nn.bootstrap_estimator(estimator=estimators[estimator],
                                                        data=temp_data[spec], B=parameters['B'], 
                                                        splits=('Train',),exog=exog,
                                                        get_averages=parameters['bootstrap_averages'])
            # If standardized, adjust results 
            if parameters['standardize_label'] == True: 
                for split in boot_expect[spec][estimator].keys(): 
                    boot_expect[spec][estimator][split] = boot_expect[spec][estimator][split]*scaler_y.scale_ + scaler_y.mean_
                    boot_mrgeff[spec][estimator][split] = boot_mrgeff[spec][estimator][split]*(scaler_y.scale_/scaler_x.scale_)
            elif parameters['standardize_features'] == True: 
                for split in boot_expect[spec][estimator].keys(): 
                    boot_mrgeff[spec][estimator][split] = boot_mrgeff[spec][estimator][split]*(1/scaler_x.scale_)

                
#            if parameters['bootstrap_averages'] == True: 
#                for split in boot_expect[spec][estimator].keys(): 
#                    boot_expect[spec][estimator][split] = np.nanmean(boot_expect[spec][estimator][split], axis=0)
#                    boot_mrgeff[spec][estimator][split] =  np.nanmean(boot_mrgeff[spec][estimator][split], axis=0)[:,0]             
#                
            print('Runtime:', str(datetime.now() - time_start), 'Spec: ' + spec +  '. Estimator: ' + estimator + '. Finished bootstrapping.')

del estimator, exog, split, temp_data 

################################################################################
################################################################################
################################################################################
### Calculate summaries 
mrgeff_avg = {}
for spec in xs.keys(): 
    mrgeff_avg[spec] = {}
    for estimator in estimators:
        mrgeff_avg[spec][estimator] = {}
        for split in ('Train',):
            mrgeff_avg[spec][estimator][split] =  np.mean(mrgeff[spec][estimator][split], axis=0)[0]

#mrgeff_avg_2 = smr.comp_wrapper_g(smr.comp_average, mrgeff, comp_kws = {'coefficient':0})
if parameters['B'] > 0:  
    if parameters['bootstrap_averages'] == False:             
        boot_expect_avg = {}
        for spec in xs.keys(): 
            boot_expect_avg[spec] = {}
            for estimator in estimators:
                boot_expect_avg[spec][estimator] = {}
                for split in ('Train',):
                    boot_expect_avg[spec][estimator][split] =  np.nanmean(boot_expect[spec][estimator][split], axis=0)
        
        boot_mrgeff_avg = {}
        for spec in xs.keys(): 
            boot_mrgeff_avg[spec] = {}
            for estimator in estimators:
                boot_mrgeff_avg[spec][estimator] = {}
                for split in ('Train',):
                    boot_mrgeff_avg[spec][estimator][split] =  np.nanmean(boot_mrgeff[spec][estimator][split], axis=0)[:,0]             
    
    else: # Averages done in estimation
        boot_expect_avg = boot_expect
        boot_mrgeff_avg = {}
        for spec in xs.keys(): 
            boot_mrgeff_avg[spec] = {}
            for estimator in estimators:
                boot_mrgeff_avg[spec][estimator] = {}
                for split in ('Train',):
                    boot_mrgeff_avg[spec][estimator][split] =  boot_mrgeff[spec][estimator][split][:,0]             

mrgeff_data = {}
mrgeff_data['EDUC'] = {}
for spec in xs.keys(): 
    mrgeff_data['EDUC'][spec] = {}
    for estimator in estimators:
        mrgeff_data['EDUC'][spec][estimator] = {}
        for split in ('Train',):
            mrgeff_data['EDUC'][spec][estimator][split] =  pd.concat((data[spec]['x'][split].reset_index(drop='index').iloc[:,0], 
                       pd.DataFrame(mrgeff[spec][estimator][split]).iloc[:,0]), axis=1, ignore_index = True)

mrgeff_data['AGE'] = {}            
for spec in xs.keys():
    mrgeff_data['AGE'][spec] = {}
    for estimator in estimators:
        mrgeff_data['AGE'][spec][estimator] = {}
        for split in ('Train',):
            mrgeff_data['AGE'][spec][estimator][split] =  pd.concat((data['Full']['x'][split].reset_index(drop='index').iloc[:,-2].round(), 
                       pd.DataFrame(mrgeff[spec][estimator][split]).iloc[:,0]), axis=1, ignore_index = True)            
            
del spec, estimator, split
################################################################################
################################################################################
################################################################################
### Present results 

################################################################################
### Table with average marginal effects 
importlib.reload(tblz)
tblz.table_wrapper_g(g_series = mrgeff_avg, cell_function =tblz.table_cell_regoutput_3line, 
                     extra_series = boot_mrgeff_avg, 
                    decimals=3, print_string=True, 
                    models = estimators.keys(),split = 'Train',
                    save_file = True, filename = 'tbl_case_mrgeff_avg',
                    cell_writer = tblz.write_cells_3line,
                    label = 'tbl_case_mrgeff_avg',
                    caption = 'Estimates of the return to schooling based on table V in Angrist \\& Krueger (1991).',
                    note1 = '\\AngristTable{'+ '{:,}'.format(parameters['B'])+'}',
#                    note1 = 'The table shows the average marginal effect of an extra year of education.' \
#                            + ' Stars show a bootstrapped significance test with *** p<0.01, ** p<0.05, * p<0.1.' \
#                            + ' Parentheses show bootstrapped standard errors.'  \
#                            + ' Brackets show a 95 pct. confidence measure based on the bootstrap as described in the text.' \
#                            + ' Data consists of 329,509 males born in the United states between 1930-1939 and measured in 1980.'  \
#                            + ' It is sampled frrom a 5 pct. sample of that year\'s US census' \
#                            + ' The dependent value is log of weekly earnings.' \
#                            + ' Instruments are a full set of quarter of birth times year of birth interactions. ',
                    ) 

################################################################################
### Figure with distribution of marginal effects 
importlib.reload(figz)
figz.fig_wrapper_g(g_series = mrgeff, g_figfunc = figz.fig_distribution, 
                  titles=xs.keys(), estimators=estimators, 
                  save_file=False, filename='mrgeff_distribution',
                  legend = 'figure', split='Train', n_rows=2, n_cols=2,
                  share_y=False, share_x=False)


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
                     ymax =[25, 0.1],
                     ymin = [0.1,0],
                    save_file = True, filename = 'fig_mrgeff_comparison_1'
                     )

# Figure for two effects 
figz.fig_wrapper(figurefunc=figz.fig_plot_mrgeff_grpby, series=mrgeff_data['EDUC']['Full'], 
                     series_extra = mrgeff_data['AGE']['Full'], 
#                     models2 = ['2SNN'],  
                     figurefunc_extra=figz.fig_plot_mrgeff_grpby,              
                     estimators=estimators, split='Train', #legend='figure'
#                     models = ['OLS (I)', 'NN (I)', '2SLS', '2SNN'],
                     titles = ['By education','By age'],
                     legend = 'first',
                     xlabel2 = 'Age (quarter of year)', xlabel = 'Years of education.',
                     ylabel= 'Marginal effect of education', 
                     ymax = 0.15,
                     #xmin = [6.7, 29.5],
                     ymin = 0,
                     save_file = True, filename = 'mrgeff_comparison_2'
                     )

  

print('Script execution time:', str(datetime.now()-time_start).strip("0").strip(":00"))
del time_start               