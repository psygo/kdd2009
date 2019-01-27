# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 17:01:34 2019

@author: Philippe
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 13:10:43 2019

@author: Philippe
"""

################################################################################
############################# 1. PREPROCESSING #################################
################################################################################

############################################
############## 1.1 IMPORTS #################
############################################

import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.ensemble import RandomForestClassifier

############################################
###### 1.2 OPENING THE TRAIN DATA ##########
############################################

df = pd.read_csv('orange_small_train.data', sep = '\t')
df_num = df.iloc[:, :-40]
df_cat = df.iloc[:, -40:]

# Targets
df_churn = pd.read_csv('orange_small_train_churn.labels', 
                       header = None, names = ['churn'])
df_upselling = pd.read_csv('orange_small_train_upselling.labels', 
                           header = None, names = ['upselling'])
df_appetency = pd.read_csv('orange_small_train_appetency.labels', 
                           header = None, names = ['appetency'])
df_targets = pd.concat([df_churn, df_upselling, df_appetency],
                       axis = 1)

############################################
####### 1.3 VERIFYING CONSISTENCY ##########
############################################

# Are the Number Variables really numbers?
sum_check = list(df_num.sum().values)
sum_check = [float(i) for i in sum_check]
num_check = sum([type(sum_check[i]) == float 
                 for i in range(0, len(sum_check))])

# The answer is yes apparently, 
# since there are 190 float numbers in the check variable
print('The number of valid numerical variables is:')
print(f'{num_check} out of {len(sum_check)}')

# Are the categorical variables really categorical?
def check_column_cat(df_col):
    """
    Checks if all the elements in a categorical column 
    are indeed categorical.
    """
    
    str_count = 0
    for i in range(0, df_col.shape[0]):
        if type(df_col.index[i]) == str:
            str_count += 1
            
    return str_count == df_col.shape[0]

# Looping on all the columns
col_checks = []
for col in df_cat.columns:
    catty = pd.DataFrame(df_cat[col])
    if int(catty.isnull().sum()) == catty.shape[0]:
        continue
    cat = catty.groupby(col).size()
    col_checks.append(check_column_cat(cat))
    
cat_check = sum(col_checks)
print('The number of valid categorical variables is:')
print(f'{cat_check} out of {len(col_checks)}')

# Correcting the data types on the dataframe
df_num = df_num.astype('float')
df_cat = df_cat.astype('category')
df_all = pd.concat([df_num, df_cat], axis = 1)

# Correcting the data types on the targets
df_churn = df_churn.astype('category')
df_upselling = df_upselling.astype('category')
df_appetency = df_appetency.astype('category')
df_targets = df_targets.astype('category')

############################################
########### 1.4 FEATURE SCALING ############
############################################

df_num = (df_num - df_num.mean()) / df_num.std()

mean, std = pd.DataFrame(df_num.mean()), pd.DataFrame(df_num.std())
mean.to_csv('train_mean.csv')
std.to_csv('train_std.csv')

############################################
### 1.5 DELETING VARS WITH TOO MANY NaNs ###
############################################

nans_per_col = df_all.isnull().sum()
num_40k = nans_per_col[nans_per_col >= 40000].shape[0]

nans_per_col.hist(bins = 20)
plt.xlabel('# of NaNs in the Column')
plt.ylabel('# of columns')
plt.title(f'Histogram of # of NaNs in the Column x # of Columns \
          \n # of Columns with more than 40,000 NaNs: {num_40k}')
plt.show()
plt.close()

def delete_useless(df, max_na):
    """
    Deletes variables with more than 'max_na' NaNs.
    """
    
    cols = df.columns
    for col in cols:    
        if df[col].isnull().sum() >= max_na:
            del df[col]

# Deleting Variables with more than 40,000 NaNs
delete_useless(df_num, 40000)
delete_useless(df_cat, 40000)

############################################
############# 1.6 FILLING NaNs #############
############################################

df_num = df_num.fillna(df_num.mean())

# Running it twice will give an error due to 
# adding the same category twice.
for col in df_cat.columns:
    df_cat[col] = df_cat[col].cat.add_categories('missing')
    df_cat[col] = df_cat[col].fillna('missing')

############################################
### 1.7 DELETING VARS WITH TOO MANY CATS ###
############################################

# Grouping the Categories of each column.
num_cats = []
for col in df_cat.columns:
    
    df_col = pd.DataFrame(df_cat[col])
    df_col = df_col.groupby(col).size()
    num_cats.append([col, df_col.index.shape[0]])

num_cats_num = [item[1] for item in num_cats]

plt.hist(num_cats_num, bins = 20)
plt.xlabel('# of Categories')
plt.ylabel('# of Variables')
plt.title('Histogram of # of Categories x # of Variables')
plt.show()
plt.close()

# Given that most of the categorical variables
# have at most 2,000 categories, let's get rid
# of the excessive ones.
cat_del = [] # categories to delete
for item in num_cats:
    if item[1] > 2000:
        cat_del.append(item[0])
        
for cat in cat_del:
    del df_cat[cat]
    
df_all = pd.concat([df_num, df_cat], axis = 1)

############################################
### 1.8 FEATURE SELECTION WITH DEC TREES ###
############################################

def get_importances(features, targets):
    """
    Returns the feature importances and orders them.
    """
    
    # Running the Tree
    clf_tree = RandomForestClassifier(n_estimators = 500,
                                      criterion = 'entropy',
                                      n_jobs = 5, 
                                      random_state = 42)
    clf_tree.fit(features, targets.values.ravel())
    importances = clf_tree.feature_importances_
    
    # Ordering the Importance of the Variables
    variables = np.array(features.columns)
    indices = np.argsort(importances)[: : -1]
    importances = importances[indices]
    variables = variables[indices]
    
    return variables, importances

# Dummy Variables Encoding
# Probably can delete one of the columns for each variable.
# But this will be done naturally when we get 
# the most important features from the Random Forest.
df_dummies = pd.get_dummies(df_all)

# Getting Importances...
dict_targets = {'Churn' : df_churn,
                'Upselling' : df_upselling,
                'Appetency' : df_appetency}

dict_importances = {k : get_importances(df_dummies, v) 
                    for k, v in dict_targets.items()}

def sum_importances(variables, importances):
    """
    Sums the importances up to the k-th variable.
    Cumulative Importance.
    """
    
    sum_importances = []
    for i in range(importances.shape[0]):
        sum_importance = importances[:(i + 1)].sum()
        sum_importances.append([variables[i], sum_importance])

    return sum_importances

# Cumulative Sum of the Importances
dict_sum_importances = {k : sum_importances(v[0], v[1])
                        for k, v in dict_importances.items()}    

def thresh_vars(features, sum_importance_list, threshold):
    """
    Thresholds the most important features.
    """
    
    thresh_vars = []
    for item in sum_importance_list:
        if item[1] <= threshold:
            thresh_vars.append(item[0])
            
    df = features.loc[:, thresh_vars]
    
    return df

# 99% Threshold
dict_thresh = {k : thresh_vars(df_dummies, v, 0.99) 
              for k, v in dict_sum_importances.items()}

# Saving the training dataset for later use during testing.
for k, v in dict_thresh.items():
    v.to_csv(f'train_thresh_{k}.csv')

# Saving the Columns of the Filtered Variables (for the testing later).
columns_thresh = {k : list(v.columns)
                  for k, v in dict_thresh.items()}
with open('variables.json', 'w') as file:
    json.dump(columns_thresh, file)
    
# Reopening the Columns of the Filtered Variables.
with open('variables.json', 'r') as file:
    columns_thresh = json.load(file)

# Importance Sum x Amount of Variables Graph

dict_sum_imp = {k : [item[1] for item in v]
                for k, v in dict_sum_importances.items()}

for k, v in dict_sum_imp.items():
    plt.plot(np.linspace(1, len(v), len(v)), v,
             label = f'Sum Importance {k}: {dict_thresh[k].shape[1]} Variables')
    
plt.axhline(0.9, color = 'green', label = '90% Threshold', ls = '--')
plt.axhline(0.99, color = 'red', label = '99% Threshold', ls = '--')
plt.xlabel('Amount of Variables')
plt.ylabel('Importance Sum')
plt.title('Importance Sum x Amount of Variables')
plt.legend()
plt.show()
plt.close()

################################################################################
############################### 2. MODELLING ###################################
################################################################################

############################################
############## 2.1 IMPORTS #################
############################################

import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# The Random Forest Import is in the Preprocessing part.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

############################################
######### 2.2 TRAIN-TEST SPLIT #############
############################################

test_size = 0.1
dict_split = {}
for k, v in dict_thresh.items():
    X_train, X_test, y_train, y_test = train_test_split(dict_thresh[k], 
                                                        dict_targets[k],
                                                        test_size = test_size,
                                                        random_state = 42)
    dict_split[k] = [X_train, X_test, 
                     y_train.values.ravel(), y_test.values.ravel()]
    
############################################
#### 2.3 EVALUATING MODELS' PERFORMANCES ###
############################################

def k_fold_val(clf, X, y, n_cv, score_type):
    """
    k-Fold Crossvalidation.
    
    Returns the Mean.
    """
    kv_score = cross_val_score(clf, 
                               X, y,
                               cv = n_cv,
                               scoring = score_type)
    
    return kv_score.mean()

def cross_val_all(clf, dict_Xy, n_cv, score_type):
    """
    Runs k-Fold Cross Validation on all 3 targets.
    """
    
    kv_all = {}
    for k, v in dict_Xy.items():
        kv_all[k] = k_fold_val(clf, 
                               v[0], v[2],
                               n_cv, score_type)
        
    return kv_all

# Dict with all the classifiers
# For the **NN**, I took a guess for the hidden layers: the number of 
# neurons in each hidden layer is about = number of variables / 2 ~ 300. 
# Avoid using too many estimators for the **Tree Classifiers**, they can take
# quite long to get optimized.
clfs = {'RF' : RandomForestClassifier(n_estimators = 200,
                                      random_state = 42,
                                      n_jobs = 5),
        'LR' : LogisticRegression(solver = 'lbfgs'),
        'GB' : GradientBoostingClassifier(n_estimators = 200, 
                                          learning_rate = 0.1, 
                                          random_state = 42),
        'DT' : DecisionTreeClassifier(),
        'GNB' : GaussianNB(),
        'SVM' : SVC(gamma = 'auto'),
        'ADA' : AdaBoostClassifier(DecisionTreeClassifier(max_depth = 3),
                                   algorithm="SAMME",
                                   n_estimators = 200),
        'NN' : MLPClassifier(solver = 'adam', max_iter = 5000, 
                             hidden_layer_sizes = (300, 300, 300, 300))}

# Each Cross Validation is effectively making n_cv * 3 fits and validations.
kv_scores_auc = {}
n_cv, score_func = 5, 'roc_auc'

# Only calling them separetely to avoid having to wait too much...
# The **SVM**, for example, takes about 1.5h to run one iteration of CV.
# Yes, I waited approximately 3 x 5 = 15h for the SVM... and all this to get
# mediocre 0.5 AUCs...
kv_scores_auc['RF'] = cross_val_all(clfs['RF'], dict_split, n_cv, score_func)
kv_scores_auc['LR'] = cross_val_all(clfs['LR'], dict_split, n_cv, score_func)
kv_scores_auc['GB'] = cross_val_all(clfs['GB'], dict_split, n_cv, score_func)
kv_scores_auc['GNB'] = cross_val_all(clfs['GNB'], dict_split, n_cv, score_func)
kv_scores_auc['SVM'] = cross_val_all(clfs['SVM'], dict_split, n_cv, score_func)
kv_scores_auc['ADA'] = cross_val_all(clfs['ADA'], dict_split, n_cv, score_func)
kv_scores_auc['NN'] = cross_val_all(clfs['NN'], dict_split, n_cv, score_func)

# Algorithm Ranking for Each Target
def sort_scores(scores):
    """
    Sorts kv_scores in descending order.
    """

    c_full, u_full, a_full = [], [], []
    for clf_k, clf_v in kv_scores_auc.items():
        
        for k, v in clf_v.items():
            if k == 'Churn':
                c = [clf_k, v]
            elif k == 'Upselling':
                u = [clf_k, v]
            elif k == 'Appetency':
                a = [clf_k, v]
                
        c_full.append(c)
        u_full.append(u)
        a_full.append(a)
        
    c_df = pd.DataFrame(c_full, columns = ['Algorithm', 'kv_score_auc'])
    u_df = pd.DataFrame(u_full, columns = ['Algorithm', 'kv_score_auc'])
    a_df = pd.DataFrame(a_full, columns = ['Algorithm', 'kv_score_auc'])
    
    ranking = {'Churn' : c_df.sort_values(by = 'kv_score_auc', 
                                          ascending = False),
               'Upselling' : u_df.sort_values(by = 'kv_score_auc', 
                                              ascending = False),
               'Appetency' : a_df.sort_values(by = 'kv_score_auc', 
                                              ascending = False)}
               
    return ranking

ranking = sort_scores(kv_scores_auc)

# Saving the `kv_scores_auc` to avoid having to run it again.
ranking_dict = {'Churn' : ranking['Churn'].values.tolist(), 
                'Upselling' : ranking['Upselling'].values.tolist(),
                'Appetency' : ranking['Appetency'].values.tolist()}
with open('kv_scores_auc_ranked.json', 'w') as file:
    json.dump(ranking_dict, file)
    
# Reopening it:
with open('kv_scores_auc_ranked.json', 'r') as file:
    ranking_json = json.load(file)
    
ranking_json = {k: pd.DataFrame(v, columns = ['Algorithm', 'kv_score_auc']) 
                for k, v in ranking_json.items()}

# Ranking Plot
for k, v in ranking_json.items():
    winner_algo = v["Algorithm"].iloc[0]
    winner_score = round(v["kv_score_auc"].iloc[0], 3)
    plt.scatter(v['Algorithm'], v['kv_score_auc'],
                label = f'{k}: {winner_algo} at {winner_score}')  
plt.xlabel('Algorithm')
plt.ylabel(f'{n_cv}-Fold Cross Validation ROC-AUC')
plt.title(f'{n_cv}-Fold Cross Validation ROC-AUC x Algorithm')
plt.legend()
plt.show()
plt.close()
    
################################################################################
######################### 3. BEST ALGORITHM OPTIMIZATION #######################
################################################################################
    
############################################
####### 3.1 SEPARATE OPTIMIZATION ##########
############################################
    
# The best algorithm for all cases seems to be **Gradient Boosting**.
    
# This is mostly going to be greedy optimization, i.e., we will fix certain
# parameters and then look for an optimal dependent one. The computational cost
# of doing a full-scale optimization is too high.
    
# Later, once we have somewhat optimized parameters, we can run a complete
# Grid Search with more restricted variables.
    
clf_gb = GradientBoostingClassifier(n_estimators = 100, 
                                    learning_rate = 0.1, 
                                    random_state = 42)    
    
# This just for reference, the grid search will be taken separetely.
# If we wanted to do search them all, it would be approximately:
# n_targets x n_cv x 4 x 10 x 6 x 8 x 4 x 4 ~= 280,000 possible combinations.
# If we do it separetely:
# n_targets x n_cv x (4 + 10 + 6 + 8 + 4 + 4) ~= 330 possible combinations.

grid_separate = {
                 'n_estimators' : {'n_estimators' : 
                                   [100, 200, 300, 400, 500]},
                 'max_features' : {'max_features' : 
                                   range(20, 41, 2)},
                 'min_samples_split' : {'min_samples_split' : 
                                        range(100, 401, 50)},
                 'min_samples_leaf' : {'min_samples_leaf' : 
                                       range(20, 60, 5)},
                 'subsample' : {'subsample' : 
                                [0.7, 0.8, 0.9, 1]},
                 'learning_rate' : {'learning_rate' : 
                                    [0.01, 0.05, 0.1, 0.5]},
                }

# Separate `GridSearchCV` for each target, for each parameter.
CV_gb = {}
for k_target, _ in dict_split.items():
    CV_gb[k_target] = {}
    for k, v in grid_separate.items():
        CV_gb[k_target][k] = GridSearchCV(estimator = clf_gb,
                                param_grid = v,
                                cv = 3,
                                scoring = 'roc_auc')
    
def get_best_param(cv_clf, dict_Xy, param):
    """
    Gets the best values for a certain parameter, for each target.
    """
    
    best = {}
    for k, v in dict_Xy.items():
        
        tic = time.time()
        cv_clf[k][param].fit(v[0], v[2])
        tac = time.time()

        best[k] = cv_clf[k][param].best_params_[param]
        
        timey = round((tac - tic) / 60, 2)
        print(f'Grid Search on {k} for the {param}: {timey} minutes.') 
        
    return best

# Should take a long while... 3h...
best_params_sep = {k : get_best_param(CV_gb, dict_split, k)
                   for k, _ in grid_separate.items()}

# Reshaping and saving
best_sep = {}
for k, _ in dict_split.items():
    best_p = {}
    for k_params, _ in grid_separate.items():
        best_p[k_params] = best_params_sep[k_params][k]
        
    best_sep[k] = best_p
        
with open('best_params_sep_new.json', 'w') as file:
    json.dump(best_sep, file)    
    
# Reopening it
with open('best_params_sep_new.json', 'r') as file:
    best_sep = json.load(file)
    
############################################
######## 3.2 GLOBAL OPTIMIZATION ###########
############################################

clf_gb_global = {}
for k, _ in dict_split.items():
    clf_gb_global[k] = GradientBoostingClassifier(n_estimators = 100, 
                                                  learning_rate = 0.1, 
                                                  random_state = 42) 

# This just for reference, the grid search will be taken separetely.
# The min_samples_split, min_samples_leaf and subsample parameters will be
# fixed for simplification.
# Now we have managed to reduce the search space to approximately:
# n_targets x n_cv x 3 x 3 x 5 = 3^4 * 5 = 405 possible combinations.
param_grid_global = {
                     'Churn' : {
                                'n_estimators' : [80, 100, 120, 300, 500],
                                'max_features' : range(35, 37, 1),
                                'min_samples_split' : [150],
                                'min_samples_leaf' : [20],
                                'subsample' : [0.75, 1],
                                'learning_rate' : [0.05, 0.1, 0.15]
                               },
                     'Upselling' : {
                                    'n_estimators' : [80, 100, 120, 300, 500],
                                    'max_features' : [38],
                                    'min_samples_split' : [200],
                                    'min_samples_leaf' : [45],
                                    'subsample' : [1],
                                    'learning_rate' : [0.05, 0.1, 0.15]
                                   }, 
                     'Appetency' : {
                                    'n_estimators' : [80, 100, 120, 300, 500],
                                    'max_features' : [38],
                                    'min_samples_split' : [100],
                                    'min_samples_leaf' : [55],
                                    'subsample' : [1],
                                    'learning_rate' : [0.05, 0.1, 0.15]
                                   }, 
                    }

CV_gb_global = {}
for k, v in dict_split.items():
    CV_gb_global[k] = GridSearchCV(estimator = clf_gb_global[k],
                                   param_grid = param_grid_global[k],
                                   cv = 3,
                                   scoring = 'roc_auc')
    
best_params_global, time_to_opt_global = {}, {}
for k, v in dict_split.items():
    
    tic = time.time()
    CV_gb_global[k].fit(v[0], v[2])
    tac = time.time()
    
    time_to_opt_global[k] = round(((tac - tic) / 60), 2)
    best_params_global[k] = CV_gb_global[k].best_params_
    
    print(f'Time of the Grid Search for {k}: {time_to_opt_global[k]} minutes.')
    print(f'{k}: {best_params_global[k]}')
    
# Saving    
with open('best_params_global.json', 'w') as file:
    json.dump(best_params_global, file)
    
# Opening
with open('best_params_global.json', 'r') as file:
    best_params_global = json.load(file)    
    
############################################
####### 3.3 FINAL MODEL'S ROC-AUC ##########
############################################
    
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

# Final Score
n_cv, score_type = 5, 'roc_auc'
kv_score_final = {}
for k, v in clf_gb_ult.items():
    
    kv_score_final[k] = cross_val_score(v, 
                                        dict_thresh[k], 
                                        dict_targets[k].values.ravel(),
                                        cv = n_cv,
                                        scoring = score_type)

for k, v in kv_score_final.items():
    kv_score_final[k] = round(kv_score_final[k].mean(), 5)
    print(f'Final ROC-AUC Score for the {k} Target: {kv_score_final[k]}')
    
with open('performance_train.json', 'w') as file:
    json.dump(kv_score_final, file)