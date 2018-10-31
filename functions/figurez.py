#Preliminaries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy as sp
from collections import defaultdict

#Import own files 
import sys
sys.path.append(r'F:\Documents\_Speciale\Code\Functions')
sys.path.append(r'C:\Users\rbjoe\Dropbox\Kugejl\10.semester\TheEnd\Code\Functions')
import monte_carlo_simulation as mc
import dgp_stuff as dgp
import neural_net as nn
import estimators as est

#This file has the following sections:     
    # General figure and ax settings
    # Wrappers 
    # Axes for use on wrappers 
    # Specail ad hoc figures


###############################################################################
###############################################################################
###############################################################################
### SETTINGS    
def ax_settings(ax, ax_counter,
                kwargs,
                legend= 'all', 
                **kws):
    fontsize = 13
    #General ax settings 
    ax.yaxis.grid(True)
    #ax.tick_params(labeltop=False, labelright=True, labelleft=True)
    
    #Legend (custom sort)
    if legend in ('all', 'first'): 
        handles, labels = ax.get_legend_handles_labels()
        if ('estimators' in kws.keys()) & (kws['estimators'] != defaultdict(dict)):     
            estimator_order = dict([(j,i) for i,j in enumerate(kws['estimators'].keys())])
            estimator_order.update({'DGP': -1})
            if 'DGP_last' in kwargs.keys(): #Move DGP to last for nicer legend
                if kwargs['DGP_last'] == True:
                    estimator_order['DGP'] = 99            
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: estimator_order[t[0]]))        
        else: #Sort alphabetically  (https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined)
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    if legend == 'all':
        ax.legend(handles, labels, 
                  frameon=True, fancybox=True, 
                  fontsize=fontsize,
                  prop={'size': np.max((13, len(labels)*2.1))})
    elif legend == 'first':
        if ax_counter == 0: 
            ax.legend(handles, labels, 
                      frameon=True, fancybox=True, 
                      fontsize=fontsize,
                      prop={'size': np.max((13, len(labels)*2.1))})
        else: 
            ax.legend().set_visible(False)
    else:   
        ax.legend().set_visible(False)
    
    #Optional ax settings
    if 'xlabel' in kwargs.keys(): 
        ax.set_xlabel(kwargs['xlabel'], fontsize=fontsize)
    
    if 'ylabel' in kwargs.keys(): 
        ax.set_ylabel(kwargs['ylabel'], fontsize=fontsize)
    
    if 'xscale' in kwargs.keys(): 
        ax.set_xscale(kwargs['xscale'])
    
    if 'ymax' in kwargs.keys():
        if type(kwargs['ymax']) in (int, float, np.float64): 
            ax.set_ylim(top=kwargs['ymax'])
        elif isinstance(kwargs['ymax'], list): 
            if type(kwargs['ymax'][ax_counter]) in (int, float, np.float64): 
               ax.set_ylim(top=kwargs['ymax'][ax_counter])
    
    if 'ymin' in kwargs.keys(): 
        if type(kwargs['ymin']) in (int, float, np.float64): 
            ax.set_ylim(bottom=kwargs['ymin'])
        elif isinstance(kwargs['ymin'], list): 
            if type(kwargs['ymin'][ax_counter]) in (int, float, np.float64): 
               ax.set_ylim(bottom=kwargs['ymin'][ax_counter])
        
    
    if 'xmax' in kwargs.keys(): 
        if type(kwargs['xmax']) in (int, float, np.float64): 
            ax.set_xlim(right=kwargs['xmax'])
        elif isinstance(kwargs['xmax'], list):
            if type(kwargs['xmax'][ax_counter]) in (int, float, np.float64): 
                ax.set_xlim(right=kwargs['xmax'][ax_counter])
    
    if 'xmin' in kwargs.keys(): 
        if type(kwargs['xmin']) in (int, float, np.float64): 
            ax.set_xlim(left=kwargs['xmin'])
        elif isinstance(kwargs['xmin'], list): 
            if type(kwargs['xmin'][ax_counter]) in (int, float, np.float64): 
               ax.set_xlim(left=kwargs['xmin'][ax_counter])
        
    # Optional parameters passed
    if 'g_function' in kws.keys(): 
        if 'g_name' in kws['g_function'].keys(): 
            ax.set_title(kws['g_function']['g_name'], size=fontsize)
    
    if 'titles' in kwargs.keys(): 
        ax.set_title(list(kwargs['titles'])[ax_counter], size=fontsize)
    
    return ax 
#
def fig_settings(fig, kwargs,
                legend= 'all', 
                **kws): 
    #General settings
    #plt.rcParams['axes.edgecolor'] = 'white'
#    plt.rcParams['axes.linewidth'] = 1
    fontsize = 13    

    if legend == 'figure': 
        # Reorder handles 
        handles, labels = fig.get_axes()[0].get_legend_handles_labels()
        if ('estimators' in kws.keys()) & (kws['estimators'] != defaultdict(dict)):     
            estimator_order = dict([(j,i) for i,j in enumerate(kws['estimators'].keys())])
            estimator_order.update({'DGP': -1})
            if 'DGP_last' in kwargs.keys(): #Move DGP to last for nicer legend
                if kwargs['DGP_last'] == True:
                    estimator_order['DGP'] = 99
            
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: estimator_order[t[0]]))        
        else: #Sort alphabetically  (https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined)
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            
        #Create legend 
        plt.figlegend(handles, labels, 
                      loc='lower center', ncol=(len(labels)+1)//2, 
                       frameon=True, fancybox=True, 
                       #bbox_to_anchor=[0.5, -0.05],
                       #borderaxespad=-1,
                       fontsize=fontsize)


###############################################################################
###############################################################################
###############################################################################
### WRAPPERS
### Creates a figure for a single simulation
def fig_wrapper(figurefunc, series, split='Test', fig_kws={}, 
                series_extra={}, figurefunc_extra={},
                g_function=defaultdict(dict), estimators=defaultdict(dict), 
                models = False, 
                legend = 'all',
                save_file = False, filename = 'fig_basic',
                **kwargs): 
    if series_extra != {}: #Different figures...
        n_cols = 2 # ... leads to double number of rows
    else: n_cols = 1
    
    fig, axes = plt.subplots(nrows = 1, ncols = n_cols, figsize=(12,6))
    
    if models == False:  #Example : g_series['Logit']['Train'].keys()
       models = series.keys()
    
    # Create (first) figure
    ax = fig.get_axes()[0]
    figurefunc(series, ax, 
               estimators=estimators, split=split,models=models, **fig_kws)
    ax = ax_settings(ax, ax_counter=0,# Applies same general options always
                 legend=legend, kwargs=kwargs, 
                 estimators=estimators) 
    
    # Optionally add a second figure
    if series_extra != {}: #Add second type of figures 
        if 'ylabel2' in kwargs.keys(): 
            kwargs['ylabel'] = kwargs['ylabel2']
        if 'xlabel2' in kwargs.keys(): 
            kwargs['xlabel'] = kwargs['xlabel2']
        if 'models2' in kwargs.keys(): 
            models = kwargs['models2']
        if figurefunc_extra != {}: 
            figurefunc = figurefunc_extra
            
        ax = fig.get_axes()[1]
        figurefunc(series_extra, ax, 
                   estimators=estimators, split=split, models=models, **fig_kws)
        ax = ax_settings(ax, ax_counter=1,# Applies same general options always
                     legend=legend, kwargs=kwargs, 
                     estimators=estimators) 
    
    # Apply overall figure settings    
    fig_settings(fig, # Applies same general options always 
                 legend=legend, kwargs=kwargs, estimators=estimators) 
    
    #Save file (optional)    
    if save_file == True:
        plt.savefig(os.getcwd() + '\\figures\\'+'fig_%s.png' % filename, 
                    format='png',bbox_inches='tight', dpi=300)
    
    
    plt.show()

###############################################################################
#### Creates a set of subplots, where each is a figure for one g. 
def fig_wrapper_g(g_figfunc, g_series, n_rows=3, n_cols=3, split='Test',fig_kws={}, 
                  g_functions=defaultdict(dict), subset = False, 
                  estimators=defaultdict(dict), models=False,
                  g_series_extra={}, g_figfunc_extra = {},
                  save_file=False, filename='fig_g',
                  legend = 'all', 
                  share_x=True, share_y=True, **kwargs): 
    
    if g_series_extra != {}: #Different figures...
        n_rows = 2*n_rows # ... leads to double number of rows

    fig, axes = plt.subplots(nrows = n_rows, ncols = n_cols, figsize=(max(3*n_cols, 12),max(3*n_rows,6)), 
                             sharex=share_x, sharey=share_y)

    
    if subset == False: #If not subset specified, print for all functions
        gs =  g_series.keys()
    else: 
        gs = subset # #Print only figures for subset of g_functions
    if models == False:  #Example : g_series['Logit']['Train'].keys()
        try: #works if different g[estimator][split] 
                models = set(g_series[np.random.choice(list(g_series.keys()))]\
                                  ['Train'].keys())-set(['Parameter'])
        except KeyError: 
                models = set(g_series[np.random.choice(list(g_series.keys()))].keys())

            
    
    ax_counter = 0            
    for g, ax in zip(gs, fig.get_axes()):
        # A few options for disregarding some labels in figures with multiple panels
        if ax_counter >= (n_rows-1)*n_cols and 'xlabel_last' in kwargs.keys(): 
            kwargs['xlabel'] = kwargs['xlabel_last']
        if ax_counter in [i for i in range(0,n_rows*n_cols, n_rows*n_cols//n_rows)]and 'ylabel_first' in kwargs.keys(): 
            kwargs['ylabel'] = kwargs['ylabel_first']
        elif ax_counter not in [i for i in range(0,n_rows*n_cols, n_rows*n_cols//n_rows)]and 'ylabel_first' in kwargs.keys(): 
            kwargs['ylabel'] = ''
        # Create figure 
        g_figfunc(g_series[g], ax, estimators=estimators, models=models, split=split, **fig_kws)
        
        #Apply ax settings
        ax = ax_settings(ax, ax_counter=ax_counter,# Applies same general options always
                         legend=legend, kwargs=kwargs, 
                         g_function=g_functions[g], estimators=estimators) 
        ax_counter += 1    
    
    if g_series_extra != {}: #Add second type of figures 
        if 'ylabel2' in kwargs.keys(): 
            kwargs['ylabel'] = kwargs['ylabel2']
        if 'xlabel2' in kwargs.keys(): 
            kwargs['xlabel'] = kwargs['xlabel2']
        if g_figfunc_extra != {}: 
            g_figfunc = g_figfunc_extra
        
    
        for g, ax in zip(gs, fig.get_axes()[ax_counter:]): 
            # Create figure
            g_figfunc(g_series_extra[g], ax, estimators=estimators, models=models, split=split, **fig_kws)
            
            #Apply ax settings
            ax = ax_settings(ax, ax_counter=ax_counter,# Applies same general options always
                             legend=legend, kwargs=kwargs, 
                             g_function=g_functions[g], estimators=estimators) 
            ax_counter += 1        
    
    # Apply overall figure settings    
    fig_settings(fig, # Applies same general options always 
                 legend=legend, kwargs=kwargs, 
                 g_function=g_functions[g], estimators=estimators) 
    
    #Save file (optional)    
    if save_file == True:
        plt.savefig(os.getcwd() + '\\figures\\'+'fig_%s.png' % filename, 
                    format='png',bbox_inches='tight', dpi=300)
            
    plt.show()
    
###############################################################################
#### Creates a set of subplots, where each is a figure for one g. 
    # However, here the figure compares two series for the same gs and models (e.g. observed/true)
    # The code assumes the series have the same gfunctions and models
def fig_wrapper_g_double(g_figfunc, g_series1, g_series2, fig_kws={}, 
                         n_rows=3, n_cols=3, 
                         split1='Test', split2='Test',
                         g_figfunc2=None, g_series1_2=None, g_series2_2=None,  #Possibility of 2 different series
                         g_functions=defaultdict(dict), subset = False, 
                         estimators=defaultdict(dict), models=False,
                         save_file=False, filename='fig_g_double',
                         legend = 'all', 
                         share_x=True, share_y=True, **kwargs): 
    if g_series1_2 != None: #Different figures...
        n_rows = 2*n_rows # ... leads to double number of rows
    fig, axes = plt.subplots(nrows = n_rows, ncols = n_cols, figsize=(max(3*n_cols, 12),max(3*n_rows,6)), 
                             sharex=share_x, sharey=share_y)
    
    if subset == False: #If not subset specified, print for all functions
        gs =  g_series1.keys()
    else: 
        gs = subset # #Print only figures for subset of g_functions
    if models == False:  #Example : g_series['Logit']['Train'].keys()
        try: #works if different g[estimator][split] 
                models = set(g_series1[np.random.choice(list(g_series1.keys()))]\
                                  ['Train'].keys())-set(['Parameter'])
        except KeyError: 
                models = set(g_series1[np.random.choice(list(g_series1.keys()))].keys())
    
    ax_counter = 0            
    for g, ax in zip(gs, fig.get_axes()): 
        # Create figure 
        g_figfunc(g_series1[g], ax, estimators=estimators, models=models, split=split1,
                      update_fig_kwargs={'linestyle': '-'}, **fig_kws)
        g_figfunc(g_series2[g], ax, estimators=estimators, models=models, split=split2,
                      update_fig_kwargs={'linestyle': ':', 'label':'_nolabel_'}, **fig_kws)
        #Apply ax settings
        ax = ax_settings(ax, ax_counter=ax_counter,# Applies same general options always
                         legend=legend, kwargs=kwargs, 
                         g_function=g_functions[g], estimators=estimators) 
        ax_counter += 1    
    
    if g_series1_2 != None: #Add second type of figures 
        if 'ylabel2' in kwargs.keys(): 
            kwargs['ylabel'] = kwargs['ylabel2']
        if 'xlabel2' in kwargs.keys(): 
            kwargs['xlabel'] = kwargs['xlabel2']
        if g_figfunc2 != None: 
            g_figfunc = g_figfunc2
        
    
        for g, ax in zip(gs, fig.get_axes()[ax_counter:]): 
            # Create figure
            g_figfunc(g_series1_2[g], ax, estimators=estimators, models=models, split=split1,
                          update_fig_kwargs={'linestyle': '-'}, **fig_kws)
            g_figfunc(g_series2_2[g], ax, estimators=estimators, models=models, split=split2,
                          update_fig_kwargs={'linestyle': ':', 'label':'_nolabel_'}, **fig_kws)
            #Apply ax settings
            ax = ax_settings(ax, ax_counter=ax_counter,# Applies same general options always
                             legend=legend, kwargs=kwargs, 
                             g_function=g_functions[g], estimators=estimators) 
            ax_counter += 1          
    
    
    # Apply overall figure settings    
    fig_settings(fig, # Applies same general options always 
                 legend=legend, kwargs=kwargs, 
                 g_function=g_functions[g], estimators=estimators) 
    
    #Save file (optional)    
    if save_file == True:
        plt.savefig(os.getcwd() + '\\figures\\'+'fig_%s.png' % filename, 
                    format='png',bbox_inches='tight', dpi=300)
            
    plt.show()
    


###############################################################################
###############################################################################
###############################################################################
# AXES FOR USE OM WRAPPERS        
def fig_visualize_run(data, ax, run=0, **kwargs): 
    ax.scatter(data['x']['Train'][run][0], data['x']['Train'][run][1], 
                        c=np.ravel(data['y']['Train'][run]), cmap='viridis', alpha=0.7)


def fig_distribution(series, ax, 
                     estimators = {},  update_fig_kwargs = {},
                     variable=0, models=False, split='Test', **kwargs):
    if models==False: 
        models = series.keys()
    for model in models: #(set(models)- set(['DGP'])):     
        if model == 'DGP': 
            fig_kwargs = {'color':'xkcd:dark red', 'linestyle':':', 'linewidth':3}
        elif 'fig_kwargs' in  estimators[model].keys(): 
            fig_kwargs = estimators[model]['fig_kwargs'].copy()
        else: 
            fig_kwargs  = {}
            
        if  update_fig_kwargs != {}: #Allows changes to global figure settinbgs
            fig_kwargs.update(update_fig_kwargs) #E.g. used to set lines for observed/actual data.
        
        if 'label' not in fig_kwargs.keys(): #Allows for no label if specified
            fig_kwargs['label'] = model #Default: Just use model name                
         
        if model != 'DGP':
            comp = np.array([serie[variable] for serie in series[model][split]])
#            print(comp)
            if len(np.unique(comp))>1: # Elements are different 
                sns.distplot(comp, 
                             hist=False, kde=True, ax=ax, kde_kws=fig_kwargs)
            else: 
                ax.axvline(x=np.unique(comp), **fig_kwargs) # Sa
        else: #Collapse distribution to single line
            comp = np.mean(series['DGP'][split], axis=0)
            try: 
                ax.axvline(x=comp[variable], **fig_kwargs)
            except Exception:  
                ax.axvline(x=comp, **fig_kwargs)
                
def fig_distribution_pool(series, ax, 
                          estimators = {}, update_fig_kwargs = {},
                          variable=0, models=False, split='Test', **kwargs):
    if models==False: 
        models = series.keys()
        
    for model in models: 
        if model == 'DGP': 
            fig_kwargs = {'color':'xkcd:dark red', 'linestyle':':', 'linewidth':3}
        elif 'fig_kwargs' in estimators[model].keys():
            fig_kwargs = estimators[model]['fig_kwargs'].copy()
        else: 
            fig_kwargs  = {}

        if  update_fig_kwargs != {}: #Allows changes to global figure settinbgs
            fig_kwargs.update(update_fig_kwargs) #E.g. used to set lines for observed/actual data.
        
        if 'label' not in fig_kwargs.keys(): #Allows for no label if specified
            fig_kwargs['label'] = model #Default: Just use model name        
        
        sns.distplot(np.array([serie for serie in series[model][split]]),  
                     hist=False, kde=True, ax=ax, kde_kws=fig_kwargs)

    
    
def fig_parseries(series, ax, 
                  estimators={}, update_fig_kwargs = {},
                  split='Test', models=False, 
                  **kwargs): 
    if models == False: 
        models = set(series[split].keys())-set(['Parameter'])
    
    for model in models: 
        #Figure settings
        if model == 'DGP': 
            fig_kwargs = {'color':'xkcd:dark red', 'linestyle':':', 'linewidth':3}
        elif 'fig_kwargs' in estimators[model].keys():
            fig_kwargs = estimators[model]['fig_kwargs'].copy()
        else: 
            fig_kwargs  = {}
        
        if  update_fig_kwargs != {}: #Allows changes to global figure settinbgs
            fig_kwargs.update(update_fig_kwargs) #E.g. used to set lines for observed/actual data.
        
        if 'label' not in fig_kwargs.keys(): #Allows for no label if specified
            fig_kwargs['label'] = model #Default: Just use model name
    
        #Create figure
        ax.plot(series[split]['Parameter'], series[split][model], 
                marker='s', markersize=4, **fig_kwargs)
  
def fig_scatter_mrgeff(series, ax, 
                  estimators={}, update_fig_kwargs = {},
                  split='Test', models=False,
                  coefficient=0,
                  **kwargs):         
    if models == False: 
        models = series.split()
#    print(series.keys())
    
    
    for model in models: 
        if model == 'DGP': 
            fig_kwargs = {'color':'xkcd:dark red', 'linestyle':':', 'linewidth':3}
        elif 'fig_kwargs' in estimators[model].keys():
            fig_kwargs = estimators[model]['fig_kwargs'].copy()
        else: 
            fig_kwargs  = {}        

        if  update_fig_kwargs != {}: #Allows changes to global figure settinbgs
            fig_kwargs.update(update_fig_kwargs) #E.g. used to set lines for observed/actual data.
        
        if 'label' not in fig_kwargs.keys(): #Allows for no label if specified
            fig_kwargs['label'] = model #Default: Just use model name        
        
        
        ax.scatter(x=np.array(series[model][split])[:,0], #Intended to be a data series
                   y=np.array(series[model][split])[:,1], #Intended to e.g. mrgeff
                              **fig_kwargs) 
 
def fig_plot_mrgeff_grpby(series, ax, 
                  estimators={}, update_fig_kwargs = {},
                  split='Test', models=False,
                  coefficient=0,
                  **kwargs):         
    if models == False: 
        models = series.split()
    
    for model in models: 
        if model == 'DGP': 
            fig_kwargs = {'color':'xkcd:dark red', 'linestyle':':', 'linewidth':3}
        elif 'fig_kwargs' in estimators[model].keys():
            fig_kwargs = estimators[model]['fig_kwargs'].copy()
        else: 
            fig_kwargs  = {}        

        if  update_fig_kwargs != {}: #Allows changes to global figure settinbgs
            fig_kwargs.update(update_fig_kwargs) #E.g. used to set lines for observed/actual data.
        
        if 'label' not in fig_kwargs.keys(): #Allows for no label if specified
            fig_kwargs['label'] = model #Default: Just use model name        
        
        temp = pd.DataFrame(series[model][split]).groupby(0, as_index=False).mean()
        
        ax.plot(np.array(temp.iloc[:,0]), #Intended to be a data series
                       np.array(temp.iloc[:,1]), #Intended to e.g. mrgeff
                       **fig_kwargs)        
    


###############################################################################
###############################################################################
###############################################################################
### AD HOC SPECIAL FIGURES (from before the main set-up was developed)
### #Plots distribution of (usually) average marginal effects 
def plot_distribution(series, split='Test', models=False, save_file = True, filename='plot_distribution'): 
    if models==False: 
        models = series.keys()

    for model in (set(models)- set(['DGP'])): 
         sns.distplot(np.array([serie[0] for serie in series[model][split]]), label = model, 
                     hist=False, kde=True)
    plt.axvline(x=np.mean(series['DGP'][split], axis=0)[0], color='xkcd:dark red', label='DGP')
    plt.legend(frameon=True, fancybox=True)
    if save_file == True:
        plt.savefig(os.getcwd() + '\\figures\\'+'%s.eps' % filename, format='eps',bbox_inches='tight')
    plt.show()


###############################################################################
### Visualizes the first two variables of one of the simulation iterations
def visualize_run(data, run=2, save_file = False, filename='visualize_run'): #Visualize the first few variables of one of the runs
    #print('The first two variables of iteration number', run+1, 'visualized:')
    plt.scatter(data['x']['Train'][run][0], data['x']['Train'][run][1], c=np.ravel(data['y']['Train'][run]), cmap='viridis', alpha=0.7)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    if save_file == True:
        plt.savefig(os.getcwd() + '\\figures\\'+'%s.eps' % filename, format='eps',bbox_inches='tight')
    plt.show()