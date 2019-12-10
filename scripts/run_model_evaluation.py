#!/usr/bin/env python
# coding: utf-8
'''
Usage: python run_model_evaluation.py infile.csv
infile.csv is an input comma-sperated file with first column as index and last column is the target column. Other columns are the features. The script will firstly standardize each column and then test the performance of six different regression models via a nested cross validation approach.

The output file has a name of infile.out
# ##### Gang Li, 2018-09-21

'''



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression as LR
from sklearn import svm
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
import sys

infile = sys.argv[-1]
outfile = infile.split('/')[-1].replace('.csv','.out')


def normalize(X):
    X_n = np.zeros_like(X)
    for i in range(X.shape[1]):
        x = X[:,i]
        X_n[:,i] = (x-np.mean(x))/np.var(x)**0.5
    return X_n




def do_cross_validation(X,y,model):
    scores = cross_val_score(model,X,y,scoring='r2',cv=5,n_jobs=1)
    return str(np.mean(scores))+','+str(np.std(scores))+'\n'




def lr():
    return LR()




def elastic_net():
    return ElasticNetCV(n_jobs=-1)



def bayesridge():
    model = BayesianRidge()
    return model




def svr():
    parameters={
                'C':np.logspace(-5,10,num=16,base=2.0),
                'epsilon':[0,0.01,0.1,0.5,1.0,2.0,4.0]
                }
    svr = svm.SVR(kernel='rbf')
    model = GridSearchCV(svr,parameters,n_jobs=-1)
    return model



def tree():
    parameters={
                'min_samples_leaf':np.linspace(0.01,0.5,10)
                }
    dtr=DecisionTreeRegressor()
    model=GridSearchCV(dtr,parameters,n_jobs=-1)
    return model



def random_forest():
    parameters = {
                    'max_features':np.arange(0.1,1.1,0.1)
    }
    rf = RandomForestRegressor(n_estimators=1000)
    model=GridSearchCV(rf,parameters,n_jobs=-1)
    return model


def test_model_performace(infile,outfile):
    df = pd.read_csv(infile,index_col=0)
    print df.shape
    X,y = df.values[:,:-1],df.values[:,-1]
    
    # normalization for non-binary features
    X_n = np.zeros_like(X)
    for i in range(X.shape[1]):
        col = df.columns[i]
        if col.startswith('EC='): X_n[:,i] = X[:,i]
        else:
            x = X[:,i]
            X_n[:,i] = (x-np.mean(x))/np.var(x)**0.5
            
    X = X_n

    fhand = open(outfile,'w')
    fhand.write('model,mean,std\n')
    
    fhand.write('Linear model,')
    fhand.write(do_cross_validation(X,y,lr()))
    
    fhand.write('Elastic Net,')
    fhand.write(do_cross_validation(X,y,elastic_net()))
    
    fhand.write('BayesRige,')
    fhand.write(do_cross_validation(X,y,bayesridge()))
    
    fhand.write('SVR model,')
    fhand.write(do_cross_validation(X,y,svr()))
    
    fhand.write('Tree model,')
    fhand.write(do_cross_validation(X,y,tree()))

    fhand.write('Random forest,')
    fhand.write(do_cross_validation(X,y,random_forest()))
    
    fhand.close()


test_model_performace(infile,outfile)

