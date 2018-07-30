# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 09:35:22 2018

@author: rabj
"""

import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os 

#Import own files 
import sys 
sys.path.append(r'F:\Documents\_Speciale\Code\Functions')
sys.path.append(r'C:\Users\rbjoe\Dropbox\Kugejl\10.semester\TheEnd\Code\Functions')
import monte_carlo_simulation as mc
import dgp_stuff as dgp
import neural_net as nn
import estimators as est
import summaries as smr
sns.set_style('whitegrid')
from matplotlib import cm
from matplotlib.collections import LineCollection

#np.random.seed(33)


def plot_function(function, save_file = False, filename='fig_gfunc', 
                  parameter_multiplier=1, parameter_adder=0): 
    #Beta - random or just ones? 
    #beta_draw = np.ones
    beta_draw = dgp.draw_beta_normal
    
    #Generate data
    x1 = np.linspace(-10,10,300)
    x2 = np.linspace(-10,10,300)
    beta = beta_draw((parameter_multiplier*1+parameter_adder)).ravel()
    #beta = np.reshape(dgp.draw_beta_normal(1,1,parameter_multiplier*1), (1,1))
    
    fig = plt.figure(figsize=(12,4))
    #Plot 2D: 
    g1 = function(pd.DataFrame(x1), beta=beta)    
    
    ax = fig.add_subplot(1,2,1)
#    ax.plot(x1,g1)
    
    color = cm.viridis((g1-g1.min())/(g1.max()-g1.min())) 
    lines = [((x,y), (x0,y0)) for x, y, x0, y0 in zip(x1[:-1], g1[:-1], x1[1:], g1[1:])]
    colored_lines = LineCollection(lines, colors=color)
    #ax.add_collection(colored_lines)
    plt.gca().add_collection(colored_lines)
    
    plt.xlim(1.1*min(x1), 1.1*max(x1))
    plt.ylim(1.1*min(g1), 1.1*max(g1))
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    
    
    #PLot 3D 
    beta = beta_draw((parameter_multiplier*2,1)).ravel()
    #beta = np.reshape(dgp.draw_beta_normal(2,1,1), (2,1))
    X, Y = np.meshgrid(x1,x2)
    Z = np.zeros((len(X), len(Y)))
    for i in range(0, len(X)):
        #Xs = pd.DataFrame({'x1': X[i], 'x2': Y[i]})
        Z[i] = function(pd.DataFrame({'x1': X[i], 'x2': Y[i]}), beta=beta).ravel()
    
    ax = fig.add_subplot(1,2,2, projection='3d')
    
    
#    colors = cm.viridis((Z-Z.min())/(Z.max()-Z.min())) #https://stackoverflow.com/questions/15134004/colored-wireframe-plot-in-matplotlib
#    rcount, ccount, _ = colors.shape
##    treD = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, 
#                           facecolors=colors, shade=False)
#    treD.set_facecolor((0,0,0,0))
    treD = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
    
    #treD = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    #ax.view_init(25,90)
    ax.set_xlabel("x2")
    ax.set_ylabel("x1")
#    ax.set_xticklabels([])
    
    ax.set_zlabel("g(x)", rotation=90)
    #ax.set_title('Two regressors')
    ax.azim = 240 #Flip figure
    ax.elev= 20
    #ax.invert_xaxis()
    #ax.invert_yaxis()
    ax.dist=8
    fig.colorbar(treD, shrink=0.3, aspect=5)
    
    if save_file == True:
        plt.savefig(os.getcwd() + '\\figures\\'+'%s.png' % filename, format='png',bbox_inches='tight', dpi=300)
    plt.show()
    
def plot_function_3dtwice(function1, function2, 
                          save_file = False, filename='fig_gfunc', 
                          parameter_multiplier_1=1, parameter_adder_1=0, 
                          parameter_multiplier_2=1, parameter_adder_2=0): 
    #Beta - random or just ones? 
    #beta_draw = np.ones
    beta_draw = dgp.draw_beta_normal
    
    #Generate data
    x1 = np.linspace(-10,10,300)
    x2 = np.linspace(-10,10,300)   
    
    fig = plt.figure(figsize=(12,6))
    
    #PLot 3D_1
    beta_1 = beta_draw((parameter_multiplier_1*2+parameter_adder_1,1)).ravel()
    #beta = np.reshape(dgp.draw_beta_normal(2,1,1), (2,1))
    X, Y = np.meshgrid(x1,x2)
    Z = np.zeros((len(X), len(Y)))
    for i in range(0, len(X)):
        #Xs = pd.DataFrame({'x1': X[i], 'x2': Y[i]})
        Z[i] = function1(pd.DataFrame({'x1': X[i], 'x2': Y[i]}), beta=beta_1).ravel()
    
    ax = fig.add_subplot(1,2,1, projection='3d')
    
    treD = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
    
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("g(x)", rotation=90)
    ax.azim = 240 #Flip figure
    ax.elev= 25
    ax.dist=7
    fig.colorbar(treD, shrink=0.3, aspect=5)
    
        #PLot 3D_1
    beta_2 = beta_draw((parameter_multiplier_2*2+parameter_adder_1,1)).ravel()
    #beta = np.reshape(dgp.draw_beta_normal(2,1,1), (2,1))
    X, Y = np.meshgrid(x1,x2)
    Z = np.zeros((len(X), len(Y)))
    for i in range(0, len(X)):
        #Xs = pd.DataFrame({'x1': X[i], 'x2': Y[i]})
        Z[i] = function2(pd.DataFrame({'x1': X[i], 'x2': Y[i]}), beta=beta_2).ravel()
    
    ax = fig.add_subplot(1,2,2, projection='3d')
    
    treD = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
    
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("g(x)", rotation=90)
    ax.azim = 240 #Flip figure
    ax.elev= 25
    ax.dist=8
    fig.colorbar(treD, shrink=0.3, aspect=5)
    
    if save_file == True:
        plt.savefig(os.getcwd() + '\\figures\\'+'%s.png' % filename, format='png',bbox_inches='tight', dpi=300)
    plt.show()
    
    
def plot_function_2dtwice(function1, function2, 
                          save_file = False, filename='fig_gfunc', 
                          parameter_multiplier_1=1, parameter_adder_1=0, 
                          parameter_multiplier_2=1, parameter_adder_2=0): 
    #Beta - random or just ones? 
    #beta_draw = np.ones
    beta_draw = dgp.draw_beta_normal
    
    #Generate data
    x1 = np.linspace(-10,10,300)
#    x2 = np.linspace(-10,10,300)   
    
    fig = plt.figure(figsize=(12,6))
    
    #Plot 2d 1
    beta = beta_draw((parameter_multiplier_1*1+parameter_adder_1)).ravel()
    #Plot 2D: 
    g1 = function1(pd.DataFrame(x1), beta=beta)    
    
    ax = fig.add_subplot(1,2,1)
#    ax.plot(x1,g1)
    
    color = cm.viridis((g1-g1.min())/(g1.max()-g1.min())) 
    lines = [((x,y), (x0,y0)) for x, y, x0, y0 in zip(x1[:-1], g1[:-1], x1[1:], g1[1:])]
    colored_lines = LineCollection(lines, colors=color)
    #ax.add_collection(colored_lines)
    plt.gca().add_collection(colored_lines)
    
    plt.xlim(1.1*min(x1), 1.1*max(x1))
    plt.ylim(1.1*min(g1), 1.1*max(g1))
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    
    
    #PLot 2d 2 
        #Plot 2d 1
    beta = beta_draw((parameter_multiplier_2*1+parameter_adder_2)).ravel()
#    fig = plt.figure(figsize=(12,4))
    #Plot 2D: 
    g1 = function2(pd.DataFrame(x1), beta=beta)    
    
    ax = fig.add_subplot(1,2,2)
#    ax.plot(x1,g1)
    
    color = cm.viridis((g1-g1.min())/(g1.max()-g1.min())) 
    lines = [((x,y), (x0,y0)) for x, y, x0, y0 in zip(x1[:-1], g1[:-1], x1[1:], g1[1:])]
    colored_lines = LineCollection(lines, colors=color)
    #ax.add_collection(colored_lines)
    plt.gca().add_collection(colored_lines)
    
    plt.xlim(1.1*min(x1), 1.1*max(x1))
    plt.ylim(1.1*min(g1), 1.1*max(g1))
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    
    
    if save_file == True:
        plt.savefig(os.getcwd() + '\\figures\\'+'%s.png' % filename, format='png',bbox_inches='tight', dpi=300)
    plt.show()    

np.random.seed(23)
#plot_function(dgp.g_wiggly, parameter_multiplier=3, parameter_adder=0, save_file=True)

#plot_function(dgp.g_trigpol_1, parameter_multiplier=2, parameter_adder=0)



#plot_function_3dtwice(function1=dgp.g_trigpol_3, 
#                      parameter_multiplier_1=6, parameter_adder_1=0, 
#                      function2= dgp.g_dropwave, 
#                      parameter_multiplier_2=6, parameter_adder_2=6, 
#                      save_file=True, filename='fig_trig_drop')

#plot_function_2dtwice(function1=dgp.g_wiggly, 
#                      parameter_multiplier_1=4, parameter_adder_1=0, 
#                      function2= dgp.g_pointy, 
#                      parameter_multiplier_2=4, parameter_adder_2=1, 
#                      save_file=True, filename='fig_2d_example')


def plot_function_2dtwice_3dtwice(function1, function2, function3, function4 ,
                          save_file = False, filename='fig_gfunc', 
                          parameter_multiplier_1=1, parameter_adder_1=0, 
                          parameter_multiplier_2=1, parameter_adder_2=0,
                          parameter_multiplier_3=1, parameter_adder_3=0,
                          parameter_multiplier_4=1, parameter_adder_4=0,
                          ): 
    #Beta - random or just ones? 
    #beta_draw = np.ones
    beta_draw = dgp.draw_beta_normal
    
    #Generate data
    x1 = np.linspace(-10,10,300)
    x2 = np.linspace(-10,10,300)   
    
    fig = plt.figure(figsize=(12,6))
    
    #Plot 2d 1
    beta = beta_draw((parameter_multiplier_1*1+parameter_adder_1)).ravel()
    #Plot 2D: 
    g1 = function1(pd.DataFrame(x1), beta=beta)    
    
    ax = fig.add_subplot(2,2,1)
#    ax.plot(x1,g1)
    
    color = cm.viridis((g1-g1.min())/(g1.max()-g1.min())) 
    lines = [((x,y), (x0,y0)) for x, y, x0, y0 in zip(x1[:-1], g1[:-1], x1[1:], g1[1:])]
    colored_lines = LineCollection(lines, colors=color)
    #ax.add_collection(colored_lines)
    plt.gca().add_collection(colored_lines)
    
    plt.xlim(1.1*min(x1), 1.1*max(x1))
    plt.ylim(1.1*min(g1), 1.1*max(g1))
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    
    
    #PLot 2d 2 
        #Plot 2d 1
    beta = beta_draw((parameter_multiplier_2*1+parameter_adder_2)).ravel()
#    fig = plt.figure(figsize=(12,4))
    #Plot 2D: 
    g1 = function2(pd.DataFrame(x1), beta=beta)    
    
    ax = fig.add_subplot(2,2,2)
#    ax.plot(x1,g1)
    
    color = cm.viridis((g1-g1.min())/(g1.max()-g1.min())) 
    lines = [((x,y), (x0,y0)) for x, y, x0, y0 in zip(x1[:-1], g1[:-1], x1[1:], g1[1:])]
    colored_lines = LineCollection(lines, colors=color)
    #ax.add_collection(colored_lines)
    plt.gca().add_collection(colored_lines)
    
    plt.xlim(1.1*min(x1), 1.1*max(x1))
    plt.ylim(1.1*min(g1), 1.1*max(g1))
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    
    
    #PLot 3D_1
    beta_3 = beta_draw((parameter_multiplier_3*2+parameter_adder_3,1)).ravel()
    #beta = np.reshape(dgp.draw_beta_normal(2,1,1), (2,1))
    X, Y = np.meshgrid(x1,x2)
    Z = np.zeros((len(X), len(Y)))
    for i in range(0, len(X)):
        #Xs = pd.DataFrame({'x1': X[i], 'x2': Y[i]})
        Z[i] = function3(pd.DataFrame({'x1': X[i], 'x2': Y[i]}), beta=beta_3).ravel()
    
    ax = fig.add_subplot(2,2,3, projection='3d')
    
    treD = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
    
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("g(x)", rotation=90)
    ax.azim = 240 #Flip figure
    ax.elev= 25
    ax.dist=7
    fig.colorbar(treD, shrink=0.3, aspect=5)
    
    if save_file == True:
        plt.savefig(os.getcwd() + '\\figures\\'+'%s.png' % filename, format='png',bbox_inches='tight', dpi=300)
    plt.show() 
    
plot_function_2dtwice_3dtwice(function1=dgp.g_wiggly, 
                  parameter_multiplier_1=4, parameter_adder_1=0, 
                  function2= dgp.g_pointy, 
                  parameter_multiplier_2=4, parameter_adder_2=1,
                  function3=dgp.g_wiggly, 
                  parameter_multiplier_3=4, parameter_adder_3=0, 
                  function4= dgp.g_pointy, 
                  parameter_multiplier_4=4, parameter_adder_4=1, 
                  save_file=True, filename='fig_2d_example')
