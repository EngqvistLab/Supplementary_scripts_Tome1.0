'''
Gang Li
2019-01-23
'''

import os
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from scipy.stats import spearmanr,pearsonr
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold


def Normalize(X):
    mean,var=list(),list()
    for i in range(X.shape[1]):
        mean.append(np.mean(X[:,i]))
        var.append(float(np.var(X[:,i]))**0.5)
    return mean,var

def standardize(X):
    Xs=np.zeros_like(X)
    n_sample,n_features=X.shape[0],X.shape[1]
    for i in range(n_features):
        Xs[:,i]=(X[:,i]-np.mean(X[:,i]))/float(np.var(X[:,i]))**0.5
    return Xs

def random_forest():
    parameters = {
                    'max_features':np.arange(0.1,1.1,0.1)
    }
    rf = RandomForestRegressor(n_estimators=1000)
    model=GridSearchCV(rf,parameters,n_jobs=-1)
    
    return model

def CrossValidation(X,Y,f,n_folds):
    kf=KFold(len(Y),n_folds=n_folds)
    p=np.zeros_like(Y)

    for train,test in kf:
        f.fit(X[train],Y[train])
        try:print f.best_params_
        except: None
        p[test]=f.predict(X[test])
    rmse_cv=np.sqrt(MSE(Y,p))
    r2_cv=r2_score(Y,p)
    r_spearman=spearmanr(Y,p)
    r_pearson=pearsonr(Y,p)
    return p,rmse_cv,r2_cv,r_spearman[0],r_pearson[0]

infile ='../model_v3/data/AA_OGT.csv'
outdir = '../model_v3/'

report=open(os.path.join(outdir,'report.txt'),'w')

# data
df = pd.read_csv(infile,index_col=0)
X = df.values[:,:-1]
Y = df.values[:,-1].ravel() 

Xs = standardize(X)
features = df.columns[:-1]

# optimize model parameter
model = random_forest()
model.fit(Xs,Y)
print model.best_params_
max_features = model.best_params_['max_features']
report.write('Best parameters:\n')
report.write(str(model.best_params_)+'\n')

# create model with optimized params
model = RandomForestRegressor(n_estimators=1000,max_features=max_features)
model.fit(Xs,Y)

# Model stats
p = model.predict(Xs)
rmse_cv = np.sqrt(MSE(p,Y))
r2_cv = r2_score(Y,p)
r_spearman = spearmanr(p,Y)
r_pearson = pearsonr(p,Y)
res = 'rmse:{:.4}\nr2:{:.4}\nspearmanr:{:.4}\np_value:{:.4}\npearonr:{:.4}\np_pearsonr:{:.4}'.format(rmse_cv,r2_cv,r_spearman[0],r_spearman[1],r_pearson[0],r_pearson[1])
print res
report.write(res+'\n')

# feature importance
fhand = open(os.path.join(outdir,'feature_importance.csv'),'w')
fhand.write('feature,importance,std\n')

std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
for i in range(X.shape[1]):
    fhand.write('{0},{1},{2}\n'.format(features[i],model.feature_importances_[i],std[i]))
fhand.close()

# Save the predicted results in a flat file
outf=open(os.path.join(outdir,'training_results.csv'),'w')
outf.write('index,exprimental,predicted\n')
for i in range(Xs.shape[0]):
    outf.write('{},{},{}\n'.format(df.index[i],Y[i],p[i]))
outf.close()
# Save model with joblib

model_name = os.path.join(outdir,'predictor/Topt_RF')
joblib.dump(model,model_name+'.pkl')
fea = open(model_name+'.f','w')
mean,var = Normalize(X)

print 'length of means:',len(mean)
print 'length of vars:',len(var)
fea.write('#Feature_name\tmean\tsigma\n')
for i in range(len(mean)):fea.write('{}\t{}\t{}\n'.format(features[i],mean[i],var[i]))
fea.close()


# cross-validate the model, get predicted Topt
model = RandomForestRegressor(n_estimators=1000,max_features=max_features)
p,rmse_cv,r2_cv,sr,pr = CrossValidation(Xs,Y,model,5)
report.write('\n\n5-fold Cross-validation results:\n')
report.write('rmse = {0}\nr2 = {1}\nsr = {2}\npr = {3}\n'.format(rmse_cv,r2_cv,sr,pr))
report.close()

# write out the predicted values by cross validation
fhand = open(os.path.join(outdir,'cross_validated_Topt.csv'),'w')
fhand.write('index,exprimental,predicted\n')
for i in range(Xs.shape[0]):
    fhand.write('{},{},{}\n'.format(df.index[i],Y[i],p[i]))
fhand.close()
             
        