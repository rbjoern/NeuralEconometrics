# Preliminaries 
import sys 
sys.path.append(r'F:\Documents\TheEnd\Code\Functions')
sys.path.append(r'C:\Users\rbjoe\Dropbox\Kugejl\10.semester\TheEnd\Code\Functions')
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import estimators as est
import neural_net as nn

np.random.seed(21321)

colors = ['xkcd:light grey','xkcd:grey blue','xkcd:salmon','xkcd:wine red']
labels = ['True negative', 'True positive', 'False negative', 'False positive']
size = (6,6)

# 0. Generate dataq
x1, x2 = np.meshgrid(np.linspace(-5,5,500),np.linspace(-5,5,500))
x = pd.DataFrame(np.vstack((x1.flatten(), x2.flatten())).T)

def circle(x1,x2,r=1):
    y = (x1**2+x2**2<r**2)
    return y

def circle2(x,r=1):
    y = (x[0]**2+x[1]**2<r**2)
    return y

#y = circle(x1,x2,r=2)
y = circle2(x,r=3)




# 1. Basic scenario
#plt.figure(figsize=size)

#for i in range (0,2): 
#    plt.scatter(x.loc[y==i,0],x.loc[y==i,1], color=colors[i], label=labels[i])
#plt.legend(frameon=True, fancybox=True)
#plt.show()

def plot_scenario(ax, x,y):
    for i in range (0,2): 
        ax.scatter(x.loc[y==i,0],x.loc[y==i,1], color=colors[i], label=labels[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_title('Target')
#    ax.axis('off')
#    return ax 

    

# 2. Logistic regression
def plot_logistic(x,y):
    betahat = est.estimate_logit(x,y)    
    prob = est.logit_prob(x,betahat)
    yhat = (prob>0.5) 
    
    plt.figure(figsize=size)
    count = 0
    for i in range (0,2): 
        for j in range (0,2):
            untrue = (y!=yhat)
            select = (untrue==i) & (yhat==j)
            plt.scatter(x.loc[select,0],x.loc[select,1], color=colors[count], label=labels[count])
            count += 1
            
    plt.legend(frameon=True, fancybox=True)
    plt.show()
    
#    return betahat, prob, yhat

#betahats, probs, yhats = {}, {}, {}
#betahats['Logit'], probs['Logit'], yhats['Logit'] = plot_logistic(x,y)
#plot_logistic(x,y)

## Neural net 

def plot_nn(ax, x,y, 
            layers=(4,),alpha=10**-1,tol=10**-4
            ):
    betahat, prob = nn.estimate_mlp(x,y, 
                                                layers = layers,
                                                activation='logistic', 
                                                solver = 'adam',
                                                #tol = tol,
                                                alpha=alpha)
    betahat = nn.unpack_mlp(betahat, k=2, layers=layers)
    yhat = (prob>0.5) 
    
#    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=size)
    
#     Plot colors 
    count = 0
    for i in range (0,2): 
        for j in range (0,2):
            untrue = (y!=yhat)
            select = (untrue==i) & (yhat==j)
            ax.scatter(x.loc[select,0],x.loc[select,1], color=colors[count], label=labels[count], s=0.1)
            count += 1
#    select = (y==0) & (yhat==0)
#    ax.scatter(x.loc[select,0],x.loc[select,1], color=colors[0], label=labels[0])
#    select = (y==1) & (yhat==1)
#    ax.scatter(x.loc[select,0],x.loc[select,1], color=colors[1], label=labels[1])
#    select = (y==1) & (yhat==0)
#    ax.scatter(x.loc[select,0],x.loc[select,1], color=colors[2], label=labels[2])
#    select = (y==0) & (yhat==1)
#    ax.scatter(x.loc[select,0],x.loc[select,1], color=colors[3], label=labels[3])
#    ax.legend(frameon=True, fancybox=True, loc = 'upper right')
#    select = (yhat==0)
#    ax.scatter(x.loc[select,0],x.loc[select,1], color=colors[0], label=labels[0])
#    select = (yhat==1)
#    ax.scatter(x.loc[select,0],x.loc[select,1], color=colors[1], label=labels[1])
    # Plot decision lines
    beta = betahat[0]
    for i in range(0,layers[0]): 
        line = -1/beta[2,i]*(beta[0,i]+beta[1,i]*x.loc[:,0])
        ax.plot(x.loc[:,0], line, label=None, color='xkcd:black', linewidth=0.2, linestyle='-')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(str(layers[0])+' nodes' + ' ('\
                 + str(np.round(100*(1-(y==yhat).sum()/len(y)),1))+' pct.)')
#    ax.axis('off')
    
    
#    plt.show()
#    return betahat, prob, yhat

#plot_nn(x,y)
#betahats['NN'], probs['NN'], yhats['NN'] = plot_nn(x,y, layers=(3,))
#print(1-np.round((y==yhats['NN']).sum()/len(y),3))

### Plot everything
fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize=(12,2.5))
plot_scenario(fig.get_axes()[0], x,y)
plot_nn(fig.get_axes()[1], x,y, layers=(3,), alpha=10**-1, tol=10**-5)
plot_nn(fig.get_axes()[2], x,y, layers=(4,), alpha=10**-1, tol=10**-5)
plot_nn(fig.get_axes()[3], x,y, layers=(32,), alpha=10**-1, tol=10**-5)

# add legend
handles, labels = fig.get_axes()[1].get_legend_handles_labels()
lgnd = plt.figlegend(handles[3:], labels[3:], loc='lower center', ncol=4)

for handle in lgnd.legendHandles:
    handle.set_sizes([40.0])


plt.savefig(os.getcwd() + '\\figures\\'+'fig_approxexample.png',
                    format='png',bbox_inches='tight', dpi=300)

plt.show()


### Plot failure example
np.random.seed(78)
fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize=(12,2,5))
plot_scenario(fig.get_axes()[0], x,y)
plot_nn(fig.get_axes()[1], x,y, layers=(4,), alpha=10**-1, tol=10**-5)
plot_nn(fig.get_axes()[2], x,y, layers=(4,), alpha=10**-1, tol=10**-5)
plot_nn(fig.get_axes()[3], x,y, layers=(4,), alpha=10**-1, tol=10**-5)

# add legend
#handles, labels = fig.get_axes()[1].get_legend_handles_labels()
lgnd = plt.figlegend(handles[3:], labels[3:], loc='lower center', ncol=4)

for handle in lgnd.legendHandles:
    handle.set_sizes([40.0])


plt.savefig(os.getcwd() + '\\figures\\'+'fig_localexample.png',
                    format='png',bbox_inches='tight', dpi=300)

plt.show()

