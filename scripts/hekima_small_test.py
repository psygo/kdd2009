# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 01:29:51 2019

@author: Philippe
"""

import pandas as pd
import json

import seaborn as sns
sns.set()

from sklearn.ensemble import GradientBoostingClassifier

############################################
########## 1.1 OPENING THE DATA ############
############################################

df_test_small = pd.read_csv('orange_small_test.data', sep = '\t')

df_num_test = df_test_small.iloc[:, :-40]
df_cat_test = df_test_small.iloc[:, -40:]
    
df_num_test = df_num_test.astype('float')
df_cat_test = df_cat_test.astype('category')

#############################################
############ 1.2 FEATURE SCALING ############
#############################################

# Doesn't matter much because it's a Decision Tree Algorithm.

## Has to use the training mean and std because that's how 
## the best algorithm was chosen and optimized.
#mean_train = pd.read_csv('train_mean.csv', 
#                         names = ['Variable', 'Mean'],
#                         skiprows = 1).set_index('Variable')
#std_train = pd.read_csv('train_std.csv', 
#                        names = ['Variable', 'Std'],
#                        skiprows = 1).set_index('Variable')
#
#mean_train = pd.Series(mean_train['Mean'])
#std_train = pd.Series(std_train['Std'])
#
#df_num_test = (df_num_test - mean_train) / std_train

############################################
############## 1.3 FILL NaNs ###############
############################################

mean_train = pd.read_csv('train_mean.csv', 
                         names = ['Variable', 'Mean'],
                         skiprows = 1).set_index('Variable')
mean_train = pd.Series(mean_train['Mean'])
    
# Filling NaNs
df_num_test = df_num_test.fillna(mean_train)

# Running it twice will give an error due to 
# adding the same category twice.
for col in df_cat_test.columns:
    df_cat_test[col] = df_cat_test[col].cat.add_categories('missing')
    df_cat_test[col] = df_cat_test[col].fillna('missing')
    
df_all = pd.concat([df_num_test, df_cat_test], axis = 1)

############################################
########### 1.4 DUMMY VARIABLES ############
############################################

# Dummy Variables Encoding
# Probably can delete one of the columns for each variable.
# **Be Careful** this line might crash your computer's RAM.
df_dummies = pd.get_dummies(df_all)

# Opening the training columns.
with open('variables.json', 'r') as file:
    columns_thresh = json.load(file)

# The columns on the test set are different, so blindly using the ones
# of the train set is not a thing, I have to test each one.
columns_dummies_test = list(df_dummies.columns)
columns_test = {}
for k, v in columns_thresh.items():
    columns_test[k] = []
    for var in v:
        if var in columns_dummies_test:
            columns_test[k].append(var)
    
# Now the dict_thresh for the test set.
dict_thresh = {k : df_dummies[v]
               for k, v in columns_test.items()}

# Saving the Datasets
for k, v in dict_thresh.items():
    v.to_csv(f'test_thresh_{k}.csv')
        
# Reopening them
dict_thresh_ext = {}
for k, v in columns_thresh.items():
    dict_thresh_ext[k] = pd.read_csv(f'test_thresh_{k}.csv',
                                     index_col = 0)
    
############################################
########## 1.5 TRAINING & TESTING ##########
############################################
    
# Training Set 
dict_thresh_train = {}
for k, v in dict_thresh_ext.items():
    dict_thresh_train[k] = pd.read_csv(f'train_thresh_{k}.csv',
                                       index_col = 0)
    
# Training Targets
train_targets = {}
train_targets['Churn'] = pd.read_csv('orange_small_train_churn.labels', 
                                     header = None, names = ['churn'])
train_targets['Upselling'] = pd.read_csv('orange_small_train_upselling.labels', 
                                         header = None, names = ['upselling'])
train_targets['Appetency'] = pd.read_csv('orange_small_train_appetency.labels', 
                                         header = None, names = ['appetency'])

for k, v in train_targets.items():
    train_targets[k] = v.astype('category')

# Training's Best Models
clf_gb_ult = {
              'Churn' : GradientBoostingClassifier(n_estimators = 500, 
                                                   learning_rate = 0.05,
                                                   min_samples_split = 150,
                                                   min_samples_leaf = 20,
                                                   max_features = 35,
                                                   subsample = 0.75,
                                                   random_state = 42),
              'Upselling' : GradientBoostingClassifier(n_estimators = 300, 
                                                       learning_rate = 0.1,
                                                       min_samples_split = 200,
                                                       min_samples_leaf = 45,
                                                       max_features = 38,
                                                       subsample = 1,
                                                       random_state = 42),
              'Appetency' : GradientBoostingClassifier(n_estimators = 300, 
                                                       learning_rate = 0.05,
                                                       min_samples_split = 100,
                                                       min_samples_leaf = 55,
                                                       max_features = 38,
                                                       subsample = 1,
                                                       random_state = 42),
             }

# Matching columns on both the training and test sets.
for k, v in dict_thresh_train.items():
    vars_test = dict_thresh_ext[k].columns.tolist()
    for col in dict_thresh_train[k].columns.tolist():
        if col not in vars_test:
            del dict_thresh_train[k][col]

# Fitting and Predicting
predictions = {}
for k, v in clf_gb_ult.items():
    
    clf_gb_ult[k].fit(dict_thresh_train[k], train_targets[k].values.ravel())
    
    predictions[k] = clf_gb_ult[k].predict(dict_thresh_ext[k])
    
   
# Saving      
for k, v in predictions.items():
    df_pred_final = pd.DataFrame(v)
    df_pred_final.to_csv(f'predictions_test_final_{k}.csv', 
                         header = None,
                         index = False)
    
    

    