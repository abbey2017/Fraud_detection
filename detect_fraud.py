# -*- coding: utf-8 -*-
"""
Created on Thu March 13, 2019

@author: abbey2017
"""


# Project to build a classifier for credit card fraud detection (Python Algorithm)
# By abiodun olaoye using Jupyter notebook

# Import relevant libraries
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.utils.fixes import signature
from sklearn import metrics

# Import data
df= pd.read_csv("creditcard.csv")


                                  ## Data Pre-processing
## Examine data size
print(df.shape)

## Explore first few lines of data
print(df.head(10).iloc[:,0:16])                   # first few columns
print(df.head(10).iloc[:,16:])                    # remaining columns

## Check data types
#print(df.dtypes)

# Perform summary statistics of data based on data types
#print(df.describe(include = [np.number]).iloc[:,0:11])
#print(df.describe(include = [np.number]).iloc[:,11:21])
#print(df.describe(include = [np.number]).iloc[:,21:]) 
  # Verdict of summary statistics: all features are well behaved but there seems to be an imbalance in the class (mean << max)
    
# Plot histogram of class to investigate imbalance
df['Count'] = df.groupby(['Class'])['Class'].transform('count')
Class_plot_data = df.sort_values(['Class']).drop_duplicates(['Class'],keep='last').iloc[:,30:32]
plt.bar(Class_plot_data.values[:,0],Class_plot_data.values[:,1],0.2,facecolor='orange')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Count of Transactions by Class')
plt.xticks(np.arange(2),('Class 0','Class 1'))
plt.show()

## Split data into training and test set while enabling stratified sampling due to class imbalance
X = df.iloc[:,1:30] 
Y = df.iloc[:,30]
test_size= 0.30
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size,random_state=40,stratify=Y)

## Re-scale all features in training set, and test set (Amount has larger values than other features from PCA)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train))     # Prefer to work with dataframes
X_test_scaled  = pd.DataFrame(scaler.transform(X_test))

## Perform summary statistics of the scaled features 
#print(X_train_scaled.describe(include = [np.number]).iloc[:,0:10])
#print(X_train_scaled.describe(include = [np.number]).iloc[:,10:20])
#print(X_train_scaled.describe(include = [np.number]).iloc[:,20:])     



### Perform Classification tasks using machine learning algorithms

                             ## Basic Classifiers
# logistic Regression
lm_model = LogisticRegression(random_state= 10,solver='lbfgs', max_iter=1000)
lm_model.fit(X_train_scaled,Y_train)
y_lm_pred = lm_model.predict(X_test_scaled)
y_lm_score = lm_model.predict_proba(X_test_scaled)[:,-1]
lm_F1 =   metrics.f1_score(Y_test,y_lm_pred)
#lm_pr,lm_re,lm_f1,_ = pd.DataFrame(list(metrics.precision_recall_fscore_support(Y_test,y_lm_pred))).iloc[:,1]
lm_AP = metrics.average_precision_score(Y_test,y_lm_score)
lm_AUROC = metrics.roc_auc_score(Y_test,y_lm_score) 
lm_PR,lm_RE,threshold = metrics.precision_recall_curve(Y_test,y_lm_score)

                           ## Ensemble Classifiers
# Randomforest
                 # experiment to select number of element based on a given number of trees in the forest 
#rf_F1_n10 = np.empty(8) * np.nan;  # To save var: import pickle; f = open('a','wb'); pickle.dump(rf_F1_n10,f); f.close()
#for i in range(3,11):    
#    rf_model = RandomForestClassifier(random_state= 10,max_depth=i, n_estimators=10)
#    rf_model.fit(X_train_scaled,Y_train)
#    y_rf_pred = rf_model.predict(X_test_scaled)
#    rf_F1_n10[i-3] =   metrics.f1_score(Y_test,y_rf_pred)

# Use optimal values of max depth and n_estimate from above to build randomforest model
rf_model = RandomForestClassifier(random_state = 10,max_depth=12, n_estimators=50)
rf_model.fit(X_train_scaled,Y_train)
y_rf_pred = rf_model.predict(X_test_scaled)
y_rf_score = rf_model.predict_proba(X_test_scaled)[:,-1]
rf_F1 =    metrics.f1_score(Y_test,y_rf_pred)
rf_AP = metrics.average_precision_score(Y_test,y_rf_score)
rf_AUROC = metrics.roc_auc_score(Y_test,y_rf_score)
rf_PR,rf_RE,threshold = metrics.precision_recall_curve(Y_test,y_rf_score)


                          ## Cascaded Classifiers
					
# Further split training set from above into 2 equal parts with stratification enabled
X_train_scaled1,X_train_scaled2,Y_train1,Y_train2 =                                   train_test_split(X_train_scaled,Y_train,test_size=0.5,random_state=40)


#logisticRegression_OneHotEncoder_GradientBoosting
grd = GradientBoostingClassifier(random_state= 10, n_estimators=50)
grd_enc = OneHotEncoder(categories='auto')
grd_lm = LogisticRegression(random_state=10,solver='lbfgs', max_iter=1000)
grd.fit(X_train_scaled1,Y_train1)
grd_enc.fit(grd.apply(X_train_scaled1)[:, :, 0])
lm_enc_grd_model = grd_lm.fit(grd_enc.transform(grd.apply(X_train_scaled2)[:, :, 0]), Y_train2)
y_grdc_pred = lm_enc_grd_model.predict(grd_enc.transform(grd.apply(X_test_scaled)[:, :, 0])) 
y_grdc_score = lm_enc_grd_model.predict_proba(grd_enc.transform(grd.apply(X_test_scaled)[:, :, 0]))[:,-1]
grdc_F1 =    metrics.f1_score(Y_test,y_grdc_pred)
grdc_AP = metrics.average_precision_score(Y_test,y_grdc_score)
grdc_AUROC = metrics.roc_auc_score(Y_test,y_grdc_score)
grdc_PR,grdc_RE,threshold = metrics.precision_recall_curve(Y_test,y_grdc_score)



                               ## Plots for results analysis 
       # Precision-Recall Plots
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
# logisticRegression
P1 = plt.subplot(131)
P1.step(lm_RE, lm_PR, color='b', alpha=0.2,
         where='post')
P1.fill_between(lm_RE, lm_PR, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('GLM Precision-Recall curve: AP={0:0.2f}, F1={1:0.2f} '.format(
          lm_AP,lm_F1),fontsize=10)
plt.rcParams["figure.figsize"] = (12,4)

# LogisticRegression_GradientBoosting
P2 = plt.subplot(132)
P2.step(grdc_RE, grdc_PR, color='lime', alpha=0.2,
         where='post')
P2.fill_between(grdc_RE, grdc_PR, alpha=0.2, color='lime', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Enhanced-GLM Precision-Recall curve: AP={0:0.2f}, F1={1:0.2f} '.format(
          grdc_AP,grdc_F1),fontsize=10)
plt.rcParams["figure.figsize"] = (12,4)
#plt.show()


# RandomForest
P3 = plt.subplot(133)
P3.step(rf_RE, rf_PR, color='red', alpha=0.2,
         where='post')
P3.fill_between(rf_RE, rf_PR, alpha=0.2, color='red', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Random Forest Precision-Recall curve: AP={0:0.2f}, F1={1:0.2f} '.format(
          rf_AP,rf_F1),fontsize=10)
plt.rcParams["figure.figsize"] = (12,4)
plt.show()
