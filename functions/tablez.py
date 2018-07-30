#Preliminaries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import io
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
    # Latex printer
    # Inputs for tables (cell functions)
    # Table wrappers 
    # Ad hoc special tables
    
    
###############################################################################
###############################################################################
###############################################################################
#Custom latex printer. 

#Cell functions 
def write_cells(df, output, numRows): 
    for i in range(numRows):
        output.write("\\vspace{6pt}\\textbf{%s} & %s\\\\\n"
                     % (df.index[i], " & ".join([str(val) for val in df.iloc[i]])))
    return output

def write_cells_2line(df, output, numRows): 
    for i in range(numRows):
    #Inserts mean and standard deviation on different lines. 
        output.write("\\vspace{6pt}\\textbf{%s} & %s\\\\\n"
                 % (df.index[i], " & ".join([str('\\cellbreak[t]{' + \
                    str(val).split(" ")[0] +'\\\\' + '\\footnotesize' + \
                    str(val).split(" ")[1]+ '}' ) \
                    for val in df.iloc[i]])))
    return output

def write_cells_3line(df, output, numRows): 
    for i in range(numRows):
    #Inserts mean and standard deviation on different lines. 
        output.write("\\vspace{6pt}\\textbf{%s} & %s\\\\\n"
                 % (df.index[i], " & ".join([str('\\cellbreak[t]{' + \
                    str(val).split(" ")[0] +'\\\\' + '\\footnotesize' + \
                    str(val).split(" ")[1]+'\\\\' + '\\footnotesize' + \
                    str(val).split(" ")[2].replace('{', ' $\{\pm').replace('}','\}$') + '}' ) \
                    for val in df.iloc[i]])))
    return output

#Printer wrapper function
def table_to_latex_custom(df, alignment="c", cell_writer = write_cells_2line,
                          double_columns = False,
                          caption='', label='', **notes):
    #Based on https://techoverflow.net/2013/12/08/converting-a-pandas-dataframe-to-a-customized-latex-tabular/
    numColumns = df.shape[1]
    numRows = df.shape[0]
    output = io.StringIO()
    #colFormat = ("%s|%s" % (alignment, alignment * numColumns))
    colFormat = ("l|%s" % (alignment * numColumns))
    #Write header
    output.write("\\begin{table}[t]\n\\centering\n")
    output.write("\\begin{threeparttable}\n\\caption{%s}\n\\label{%s}\n" %(caption,label))
    output.write("\\begin{tabular}{%s}\\toprule\n" % colFormat)
    if double_columns == False : #Default: Only one column title
        columnLabels = ["\\textbf{%s}" % label for label in df.columns]
        output.write("& %s\\\\\\midrule\n" % " & ".join(columnLabels))
    else: #Alternative: Double column names 
        order = list(dict.fromkeys(df.columns.labels[0])) #removes duplicates while preserving order 
        raw_labels = list(df.columns.levels[0])
        sorted_labels = [x for _,x in sorted(zip(order,raw_labels))]
        columnLabels1 = ["\\textbf{%s}" % label for label in sorted_labels]
        output.write("& %s\\\\\n" % " & ".join(["\\multicolumn{2}{c}{"+val + "} " for val in columnLabels1]))
        columnLabels2 = ["\\textbf{%s}" % label for label in df.columns.levels[1]]
        output.write("& %s\\\\\\midrule\n" % " & ".join([columnLabels2[0] + " & " + columnLabels2[1] 
                                                            for _ in range(0,len(columnLabels1))]))
        
    #Write data lines
    output = cell_writer(df, output, numRows)
    #Write footer
    output.write("\\bottomrule\n\\end{tabular}\n")
    if len(notes)!=0: 
        output.write("\\begin{tablenotes}\\footnotesize\n")
        count = 0
        for note in notes: 
            if count ==0:   output.write("\\item \\textit{Notes:} ") #Notes: 
            else:           output.write("\\item ")         #Other lines                
            output.write("%s\n" % notes[note])
            count+=1 
        output.write("\\end{tablenotes}\n")
    output.write("\\end{threeparttable}\n")
    output.write("\\end{table}\n")
    return output.getvalue()

###############################################################################
###############################################################################
###############################################################################
### INPUTS FOR THE TABLES
def table_cell_avgstd(series, decimals=2, **kwargs): 
#    print(series)
#    cell = '%.2f (%.3f)' % (np.mean(series), np.std(series))
    cell = '{:.{dec1}f} ({:.{dec2}f})'.format(np.nanmean(series), np.std(series), dec1=decimals, dec2=decimals+1)
#    str(np.round(np.mean(series), decimals)) + str(' (') \
#    + str(np.round(np.std(series), decimals+1)) + str(')')
    return cell

def table_cell_avg(series, decimals=2, **kwargs): 
    cell = '{:.{dec1}f}'.format(np.mean(series), dec1=decimals)
    return cell

def table_cell_avgminmax(series, decimals=2, **kwargs): 
    
    cell = '{:.{dec1}f} ({:.{dec1}f}, {:.{dec1}f})'.format(np.mean(series), np.min(series), np.max(series), dec1=decimals)

    return cell

def table_cell_avg_extrastdev(series, extra_series, decimals=2, **kwargs): 
    
    cell = '{:.{dec1}f} ({:.{dec2}f})'.format(np.nanmean(series), np.std(extra_series), 
            dec1=decimals+1, dec2=decimals+1)
    
    return cell



def table_cell_nothing(series, decimals=2, **kwargs): 

    cell = '{:.{dec}f}'.format(series, dec=decimals)

    return cell

def table_cell_regoutput(series, extra_series, decimals=2, **kwargs): 

    cell = '{:.{dec1}f} ({:.{dec2}f})'.format(series, np.std(extra_series), dec1=decimals, dec2=decimals+1)

    return cell

# Compute confidence interval based measures for bootstrap
def comp_bootstrap_confinterval(estimate, bootstrap, conf_level=0.05, method='midpoint'):
    lower = np.percentile(bootstrap, 100*(0.5*conf_level), interpolation=method)
    upper = np.percentile(bootstrap, 100*(1-0.5*conf_level), interpolation=method)
    
    return lower, upper

def table_cell_avgconf(series, decimals=2, conf_level=0.05, method='midpoint',
                       **kwargs): 
    

    lower, upper = comp_bootstrap_confinterval(series, series, conf_level, method)  
    cell = '{:.{dec1}f} [{:.{dec1}f},{:.{dec1}f}]'.format(np.mean(series), lower, upper, dec1=decimals)

    return cell

def table_cell_avgconf_boot(series, extra_series, decimals=2, conf_level=0.05, method='midpoint',
                       **kwargs): 
#    avg_conf = np.nanmean(extra_series, axis=1)
#    print(np.shape(extra_series))
#    lower = avg_conf[0]
#    upper = avg_conf[1]
    lower = np.nanmean(extra_series[0])
    upper = np.nanmean(extra_series[1])
    
    cell = '{:.{dec1}f} [{:.{dec1}f},{:.{dec1}f}]'.format(np.mean(series), lower, upper, dec1=decimals)

    return cell

def comp_bootstrap_confpm(estimate, bootstrap, conf_level=0.05, method='midpoint'):
    lower = np.percentile(bootstrap, 100*(0.5*conf_level), interpolation=method)
    upper = np.percentile(bootstrap, 100*(1-0.5*conf_level), interpolation=method)
    
#    pm = 0.5*(upper-lower)
    pm = np.max((np.abs(estimate-lower), np.abs(estimate-upper)))
#    
    return pm

def comp_bootstrap_test(bootstrap, null=0, conf_level=0.05, method='linear'):
    lower = np.percentile(bootstrap, 100*(0.5*conf_level), interpolation=method)
    upper = np.percentile(bootstrap, 100*(1-0.5*conf_level), interpolation=method)

    if (null < lower) or (null > upper): 
        reject_null = True
    else: 
        reject_null = False 
    
    return reject_null

def comp_bootstrap_teststars(bootstrap, null=0, conf_levels=[0.1, 0.05, 0.01], method='linear'):
    
    if comp_bootstrap_test(bootstrap, conf_level=conf_levels[-1], null=0, method=method) == True: 
        stars = '***'
    elif comp_bootstrap_test(bootstrap, conf_level=conf_levels[-2], null=0,method=method) == True: 
        stars = '**'
    elif comp_bootstrap_test(bootstrap, conf_level=conf_levels[-3], null=0,method=method) == True: 
        stars = '*'
    else: 
        stars = ''
    return stars 


def table_cell_regoutput_3line(series, extra_series, 
                               decimals=2, conf_level=0.05,
                               add_stars = True, 
                               **kwargs): 

    if add_stars == True: 
        stars = comp_bootstrap_teststars(bootstrap=extra_series)
    else: stars = ''
        
    
    bootstrap_pm = comp_bootstrap_confpm(estimate=series, 
                                             bootstrap=extra_series,
                                             conf_level=conf_level,
                                             )
    
    cell = '{:.{dec1}f}{:} ({:.{dec2}f}) {{{:.{dec1}f}}}'.format(series, 
                                              stars,
                                              np.std(extra_series), 
                                              bootstrap_pm,
                                              dec1=decimals, dec2=decimals+1)
    return cell



###############################################################################
###############################################################################
###############################################################################
### WRAPPER TABLES
### Table which prints a single unit for each g/model with g in rows and models in columns. 
def table_wrapper_g(g_series, cell_function, 
                    extra_series = est.dd_inf(), #Typically adds bootstrapper to cell function.
                    g_functions=defaultdict(dict), g_subset=False, 
                    models = False, 
                    split='Test', decimals=2,
                    print_string=True, 
                    save_file = False, filename='table_wrapper_g', 
                    transpose = False,
                    cell_writer = write_cells_2line,
                    **latex_kws): 
    if g_subset == False: #If not subset specified, print for all functions
        gs =  g_series.keys()
    else: 
        gs = g_subset # #Print only figures for subset of g_functions
    if models == False: 
        models = g_series[np.random.choice(list(g_series.keys()))].keys()
    
    printer=est.dd_inf()
    for g in gs:
        if 'g_name' in g_functions[g].keys(): # Allow for "pretty" version of g. 
            g_name = g_functions[g]['g_name']
        else: g_name = g 
        
        for model in models: 
            printer[model][g_name] = cell_function(series=g_series[g][model][split],
                                                   extra_series = extra_series[g][model][split],
                                                   decimals=decimals)
            #str(np.round(np.mean(g_series[g][model][split]), decimals)) + str(' (') \
            #                    + str(np.round(np.std(g_series[g][model][split]), decimals+1)) + str(')')
    table = pd.DataFrame({model: printer[model] for model in printer.keys()}, 
                                   columns = printer.keys(), 
                                   index = printer[np.random.choice(list(printer.keys()))].keys())
    
    if transpose == True: 
        table = table.transpose()
    
    if print_string==True: 
        print('---------------------------------------------------------------------')
        print('TABLE: '+filename)
        print('---------------------------------------------------------------------')
        print(table.round(decimals))
        print('---------------------------------------------------------------------')    
   
    if save_file==True: 
        with open(os.getcwd() + '\\tables\\'+'%s.tex' % filename, "w") as f: 
            f.write(table_to_latex_custom(table,
                                          cell_writer = cell_writer, 
                                          **latex_kws))
    
    return table

### Table which compares two different series for the same models/g_functi9ons
def table_wrapper_g_double(g_series1, g_series2, cell_function, 
                            extra_series1 = est.dd_inf(),extra_series2 = est.dd_inf(), #Typically adds bootstrapper to cell function.
                            g_functions=defaultdict(dict), g_subset=False, 
                            models = False, 
                            split1='Test', split2='Test', decimals=2,
                            print_string=True,
                            save_file = False, filename='table_wrapper_g', 
                            **kwargs): 
    if g_subset == False: #If not subset specified, print for all functions
        gs =  g_series1.keys()
    else: 
        gs = g_subset # #Print only figures for subset of g_functions
    if models == False: 
        models = g_series1[np.random.choice(list(g_series1.keys()))].keys()
    
    printer1, printer2 = est.dd_inf(),est.dd_inf()
    for g in gs:
        if 'g_name' in g_functions[g].keys(): # Allow for "pretty" version of g. 
            g_name = g_functions[g]['g_name']
        else: g_name = g 
        
        for model in models: 
            printer1[model][g_name] = cell_function(series=g_series1[g][model][split1], 
                                                    extra_series = extra_series1[g][model][split1],
                                                    decimals=decimals)
            printer2[model][g_name] = cell_function(series=g_series2[g][model][split2], 
                                                    extra_series = extra_series2[g][model][split2],
                                                    decimals=decimals)
            #str(np.round(np.mean(g_series[g][model][split]), decimals)) + str(' (') \
            #                    + str(np.round(np.std(g_series[g][model][split]), decimals+1)) 
    if 'title1' in kwargs.keys(): 
        title1 = kwargs['title1']
        del kwargs['title1']
    else: title1 = 'Act.' #Actual model
    if 'title2' in kwargs.keys(): 
        title2 = kwargs['title2']
        del kwargs['title2']
    else: title2 = 'Obs.' #Actual model
    
    table_stuff1 = {(model, title1): printer1[model] for model in printer1.keys()}
    table_stuff2 = {(model, title2): printer2[model] for model in printer2.keys()}
    table_stuff = {**table_stuff1, **table_stuff2}
#    table_stuff = table_stuff1.update(table_stuff2) #Updates table1 and returns None. 

    # Reorder index    
    index = pd.MultiIndex.from_tuples(table_stuff.keys())
    index.set_labels([[i for i in index.labels[0][0:len(models)] for _ in (0,1)], # duplicates first level in original order 
                    [0,1]*len(models)], inplace=True) #Iterate (Act., Obs.)

    #Generate table    
    table = pd.DataFrame(table_stuff, 
                           columns = index, 
                           index = printer1[np.random.choice(list(printer1.keys()))].keys(),
                           )
    
    if print_string==True: 
        print('---------------------------------------------------------------------')
        print('TABLE: '+filename)
        print('---------------------------------------------------------------------')
        print(table.round(decimals))
        print('---------------------------------------------------------------------')    
    
    if save_file==True: 
        with open(os.getcwd() + '\\tables\\'+'%s.tex' % filename, "w") as f: 
            f.write(table_to_latex_custom(table,double_columns=True, **kwargs))
    
    return table








###############################################################################
###############################################################################
###############################################################################
### AD HOC SPECIAL TABLES
### average marginal effect
def tables_avgmargeff(mrgeffs_avg, split='Test', models=False, print_string=True, 
                      decimals=2, save_file = True, filename='tables_avgmargeff'):
    if models==False: 
        models = mrgeffs_avg.keys()
    table = pd.DataFrame({model: np.round(np.mean(mrgeffs_avg[model][split], axis=0), 
                                       decimals) for model in models}, 
                                   columns = models)
    
    if print_string == True: 
        print('---------------------------------------------------------------------')
        print('TABLE: MARGINAL EFFECTS')
        print('---------------------------------------------------------------------')
        print(table.round(decimals))
        print('---------------------------------------------------------------------')
    
    if save_file==True: 
        with open(os.getcwd() + '\\tables\\'+'%s.tex' % filename, "w") as f: 
            f.write(table.to_latex())
    
    return table

### Accuracy and other measures for a single simulation. 
def tables_accuracy(accs, models=False, decimals=2, print_string=True, save_file = True, filename='tables_accuracy'):
    if models==False: 
        models = accs.keys()
    
    printer = {}
    for model in models: 
        printer[model] = {}
        printer[model]['In-sample']     = accs[model]['Train']['Accuracy']
        printer[model]['Out-of-sample'] = accs[model]['Test']['Accuracy']
        printer[model]['Precision']     = accs[model]['Test']['Precision']
        printer[model]['Recall']        = accs[model]['Test']['Recall']
        printer[model]['F1 score']      = accs[model]['Test']['F1 score']
        for key in printer[model].keys(): 
            printer[model][key] = np.round(np.mean(printer[model][key]), decimals)
    
    table = pd.DataFrame({model: printer[model] for model in printer.keys()}, 
                                   columns = printer.keys(), 
                                   index = printer['DGP'].keys())
    
    if print_string == True: 
        print('---------------------------------------------------------------------')
        print('TABLE: ACCURACY')
        print('---------------------------------------------------------------------')
        print(table.round(decimals).to_string())
        print('---------------------------------------------------------------------')
        
    if save_file==True: 
        with open(os.getcwd() + '\\tables\\'+'%s.tex' % filename, "w") as f: 
            f.write(table_to_latex_custom(table))
    
    return table