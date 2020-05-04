#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### USER INPUT SECTION ###

# Global Variables #
global_source_name = "C:/Users/debas/Downloads/Python Code Library/SkLearn Model Codes/Datasets/Classification_Model_Dataset.csv" 
global_id_var = 'Phone_Number'
global_dep_var = 'Churn'
global_postive_class = 'Yes'
global_test_split = 0.2
global_prob_cutoff = 0.85
global_k_fold_cv = 10
global_seed = 1234

# Model Configurations (Logistic Regression, Random Forest, Support Vector, Gradient Boosting)
logistic_reg_c = [0.001, 0.01, 0.1, 1, 10]
random_forest_n_tree = [50,100,200,300,500]
random_forest_max_depth = [3,4,5]
random_forest_min_sample_split = [8,10,12]
random_forest_min_sample_leaf = [4,5,6]
support_vector_c = [0.001, 0.01, 0.1, 1, 10]
support_vector_gamma = [0.1, 0.01, 0.001]
gbm_max_depth = [3,4,5]
gbm_min_sample_leaf = [4,5,6]
gbm_n_tree = [50,100,200,300,500]
gbm_learning_rate = [0.05,0.1,0.2,0.3,0.4,0.5]
xgb_min_child_weight = [1,3,6,9]
xgb_gamma = [0.5, 1, 1.5, 2, 5]
xgb_subsample = [0.6, 0.8, 1.0]
xgb_max_depth = [3,4,5]
xgb_learning_rate = [0.05,0.1,0.2,0.3,0.4,0.5]
xgb_n_estimators = [50,100,200,300,500]


# In[ ]:


### IMPORT ALL NECCESSARY PACKAGES ###

import numpy as np
import pandas as pd
from time import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt


# In[ ]:


### USER DEFINED FUNCTION: RAW DATA IMPORT ###

def data_import(source_name, id_var, dep_var):
    
    import_start_time = time()
    
    print("\nKindly Follow The Log For Tracing The Modelling Process\n")
    
    df = pd.read_csv(global_source_name)
    
    df_x = df[df.columns[~df.columns.isin([id_var,dep_var])]]
    df_y = df.loc[:,dep_var].astype('category')
    
    numeric_cols = df_x.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_x.select_dtypes(include=['object']).columns.tolist()
    
    import_end_time = time()
    import_elapsed_time = (import_end_time - import_start_time)
    print("\nTime To Perform Data Import: %.3f Seconds\n" % import_elapsed_time)
    
    final_data_import = [df_x,df_y,numeric_cols,categorical_cols]
        
    return(final_data_import)


# In[ ]:


### USER DEFINED FUNCTION: TRAIN & TEST SAMPLE CREATION USING RANDOM SAMPLING ###

def random_sampling(x, y, split):
    
    sampling_start_time = time()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split, random_state=1000, stratify=y)
    
    sampling_end_time = time()
    sampling_elapsed_time = (sampling_end_time - sampling_start_time)
    print("\nTime To Perform Random Sampling For Train & Test Set: %.3f Seconds\n" % sampling_elapsed_time)
    
    final_sampling = [x_train, x_test, y_train, y_test]
        
    return(final_sampling)


# In[ ]:


### USER DEFINED FUNCTION: LOGISTIC REGRESSION MODEL ###

def model_logistic_regression(train_x,
                              train_y,
                              test_x,
                              test_y,
                              num_col_list,
                              cat_col_list,
                              class_pos,
                              n_cv,
                              prob_cutoff,
                              param_c):
    
    print("\nStarting Logistic Regression Model Devlopment\n")
    
    lr_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('logreg', LogisticRegression(fit_intercept=True,
                                                                   max_iter=100,
                                                                   class_weight='balanced',
                                                                   n_jobs=-1))])
    # Hyper Parameter Tuning #
    hyper_parameters = {'logreg__C':param_c}
    
    # Model Development #
    lr_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='roc_auc')
    lr_cv_model.fit(train_x, train_y)
    
    # Model Validation #
    y_pred_prob = lr_cv_model.predict_proba(test_x)[:,1]
    y_pred_class = np.where(y_pred_prob>=prob_cutoff, 1, 0)
    y_actual_class = [1 if x == class_pos else 0 for x in test_y]
    
    # ROC Curve #
    fpr, tpr, thresholds = roc_curve(y_actual_class, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve On Test Sample (Logistic Regression)')
    plt.show()
    
    bal_acc = round(balanced_accuracy_score(y_actual_class, y_pred_class, adjusted=True),2)*100   
    auc = round(roc_auc_score(y_actual_class, y_pred_class),2)*100
    prec_recall_score = round(average_precision_score(y_actual_class, y_pred_class, average = 'weighted'),2)*100
    
    
    lr_end_time = time()
    lr_elapsed_time = round(lr_end_time - lr_start_time,2)
    print("\nTime To Develop Logistic Regression Model: %.3f Seconds\n" % lr_elapsed_time)
    
    lr_model_stat = pd.DataFrame({"Model Name" : ["Logistic Regression"],
                                  "Balanced Accuracy(%)": bal_acc,
                                  "AUC(%)": auc, 
                                  "Precision-Recall Score(%)": prec_recall_score,
                                  "Time (Sec.)": lr_elapsed_time})
    final_result = (lr_cv_model,lr_model_stat)
    
    return(final_result)


# In[ ]:


### USER DEFINED FUNCTION: RANDOM FOREST MODEL ###

def model_random_forest(train_x, 
                        train_y, 
                        test_x, 
                        test_y, 
                        num_col_list, 
                        cat_col_list,
                        class_pos, 
                        n_cv,
                        prob_cutoff,
                        param_tree,
                        param_min_leaf_sample,
                        param_min_split_sample,
                        param_max_depth):
    
    print("\nStarting Random Forest Model Devlopment\n")
    
    rf_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('rf', RandomForestClassifier())])
                                     
    # Hyper Parameter Tuning #
    hyper_parameters = {'rf__n_estimators':param_tree,
                        'rf__max_features':['auto','sqrt','log2'],
                        'rf__min_samples_leaf':param_min_leaf_sample,
                        'rf__min_samples_split':param_min_split_sample,
                        'rf__max_depth':param_max_depth}
    
    # Model Development #
    rf_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='roc_auc')
    rf_cv_model.fit(train_x, train_y)
    
    # Model Validation #
    y_pred_prob = rf_cv_model.predict_proba(test_x)[:,1]
    y_pred_class = np.where(y_pred_prob>=prob_cutoff, 1, 0)
    y_actual_class = [1 if x == class_pos else 0 for x in test_y]
    
    # ROC Curve #
    fpr, tpr, thresholds = roc_curve(y_actual_class, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve On Test Sample (Random Forest)')
    plt.show()
    
    bal_acc = round(balanced_accuracy_score(y_actual_class, y_pred_class, adjusted=True),2)*100   
    auc = round(roc_auc_score(y_actual_class, y_pred_class),2)*100
    prec_recall_score = round(average_precision_score(y_actual_class, y_pred_class, average = 'weighted'),2)*100
    
    
    rf_end_time = time()
    rf_elapsed_time = round(rf_end_time - rf_start_time,2)
    print("\nTime To Develop Random Foreest Model: %.3f Seconds\n" % rf_elapsed_time)
    
    rf_model_stat = pd.DataFrame({"Model Name" : ["Random Forest"],
                                  "Balanced Accuracy(%)": bal_acc,
                                  "AUC(%)": auc, 
                                  "Precision-Recall Score(%)": prec_recall_score,
                                  "Time (Sec.)": rf_elapsed_time})
    final_result = (rf_cv_model,rf_model_stat)
    
    return(final_result)


# In[ ]:


### USER DEFINED FUNCTION: SUPPORT VECTOR MODEL ###

def model_support_vector(train_x, 
                        train_y, 
                        test_x, 
                        test_y, 
                        num_col_list, 
                        cat_col_list,
                        class_pos, 
                        n_cv,
                        prob_cutoff,
                        param_c,
                        param_gamma):
    
    print("\nStarting Support Vector Model Devlopment\n")
    
    svm_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('svm', SVC(probability = True))])
                                     
    # Hyper Parameter Tuning #
    hyper_parameters = {'svm__C':param_c,
                        'svm__gamma':param_gamma,
                        'svm__kernel':['linear','rbf']}
    
    # Model Development #
    svm_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='roc_auc')
    svm_cv_model.fit(train_x, train_y)
    
    # Model Validation #
    y_pred_prob = svm_cv_model.predict_proba(test_x)[:,1]
    y_pred_class = np.where(y_pred_prob>=prob_cutoff, 1, 0)
    y_actual_class = [1 if x == class_pos else 0 for x in test_y]
    
    # ROC Curve #
    fpr, tpr, thresholds = roc_curve(y_actual_class, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve On Test Sample (Support Vector)')
    plt.show()
    
    bal_acc = round(balanced_accuracy_score(y_actual_class, y_pred_class, adjusted=True),2)*100   
    auc = round(roc_auc_score(y_actual_class, y_pred_class),2)*100
    prec_recall_score = round(average_precision_score(y_actual_class, y_pred_class, average = 'weighted'),2)*100
    
    
    svm_end_time = time()
    svm_elapsed_time = round(svm_end_time - svm_start_time,2)
    print("\nTime To Develop Support Vector Model: %.3f Seconds\n" % svm_elapsed_time)
    
    svm_model_stat = pd.DataFrame({"Model Name" : ["Support Vector"],
                                  "Balanced Accuracy(%)": bal_acc,
                                  "AUC(%)": auc, 
                                  "Precision-Recall Score(%)": prec_recall_score,
                                  "Time (Sec.)": svm_elapsed_time})
    final_result = (svm_cv_model,svm_model_stat)
    
    return(final_result)


# In[ ]:


### USER DEFINED FUNCTION: GRADIENT BOOSTING MODEL ###

def model_gradient_boosting(train_x,
                            train_y, 
                            test_x, 
                            test_y, 
                            num_col_list, 
                            cat_col_list,
                            class_pos, 
                            n_cv,
                            prob_cutoff,
                            param_max_depth,
                            param_min_Sample_leaf,
                            param_n_tree,
                            param_lr):
    
    print("\nStarting Gradient Boosting Model Devlopment\n")
    
    gbm_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('gbm', GradientBoostingClassifier(random_state=0))])
                                     
    # Hyper Parameter Tuning #
    hyper_parameters = {'gbm__loss':['deviance','exponential'],
                        'gbm__learning_rate': param_lr,
                        'gbm__n_estimators':param_n_tree,
                        'gbm__criterion':['friedman_mse','mae'],
                        'gbm__min_samples_leaf':param_min_Sample_leaf,
                        'gbm__max_depth':param_max_depth,
                        'gbm__max_features':['auto','sqrt','log2']}
    
    # Model Development #
    gbm_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='roc_auc')
    gbm_cv_model.fit(train_x, train_y)
    
    # Model Validation #
    y_pred_prob = gbm_cv_model.predict_proba(test_x)[:,1]
    y_pred_class = np.where(y_pred_prob>=prob_cutoff, 1, 0)
    y_actual_class = [1 if x == class_pos else 0 for x in test_y]
    
    # ROC Curve #
    fpr, tpr, thresholds = roc_curve(y_actual_class, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve On Test Sample (Gradient Boosting)')
    plt.show()
    
    bal_acc = round(balanced_accuracy_score(y_actual_class, y_pred_class, adjusted=True),2)*100   
    auc = round(roc_auc_score(y_actual_class, y_pred_class),2)*100
    prec_recall_score = round(average_precision_score(y_actual_class, y_pred_class, average = 'weighted'),2)*100
    
    
    gbm_end_time = time()
    gbm_elapsed_time = round(gbm_end_time - gbm_start_time,2)
    print("\nTime To Develop Gradient Boosting Model: %.3f Seconds\n" % gbm_elapsed_time)
    
    gbm_model_stat = pd.DataFrame({"Model Name" : ["Gradient Boosting"],
                                   "Balanced Accuracy(%)": bal_acc,
                                   "AUC(%)": auc, 
                                   "Precision-Recall Score(%)": prec_recall_score,
                                   "Time (Sec.)": gbm_elapsed_time})
    final_result = (gbm_cv_model,gbm_model_stat)
    
    return(final_result)
 

# In[ ]:


### USER DEFINED FUNCTION: XGBOOST MODEL ###

def model_xgboost(train_x,
                  train_y,
                  test_x,
                  test_y,
                  num_col_list,
                  cat_col_list,
                  class_pos, 
                  n_cv,
                  prob_cutoff,
                  param_min_child_weight,
                  param_gamma,
                  param_subsample,
                  param_max_depth,
                  param_learning_rate,
                  param_n_estimators):
    
    print("\nStarting XGBoost Model Devlopment\n")
    
    xgb_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('xgb', XGBClassifier(random_state=0, objective='binary:logistic'))])
                                     
    # Hyper Parameter Tuning #
    hyper_parameters = {'xgb__min_child_weight': param_min_child_weight,
                        'xgb__gamma': param_gamma,
                        'xgb__subsample': param_subsample,
                        'xgb__max_depth': param_max_depth,
                        'xgb__learning_rate': param_learning_rate,
                        'xgb__n_estimators': param_n_estimators}
    
    # Model Development #
    xgb_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='roc_auc')
    xgb_cv_model.fit(train_x, train_y)
    
    # Model Validation #
    y_pred_prob = xgb_cv_model.predict_proba(test_x)[:,1]
    y_pred_class = np.where(y_pred_prob>=prob_cutoff, 1, 0)
    y_actual_class = [1 if x == class_pos else 0 for x in test_y]
    
    # ROC Curve #
    fpr, tpr, thresholds = roc_curve(y_actual_class, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve On Test Sample (XGBoost)')
    plt.show()
    
    bal_acc = round(balanced_accuracy_score(y_actual_class, y_pred_class, adjusted=True),2)*100   
    auc = round(roc_auc_score(y_actual_class, y_pred_class),2)*100
    prec_recall_score = round(average_precision_score(y_actual_class, y_pred_class, average = 'weighted'),2)*100
    
    
    xgb_end_time = time()
    xgb_elapsed_time = round(xgb_end_time - xgb_start_time,2)
    print("\nTime To Develop XGBosst Model: %.3f Seconds\n" % xgb_elapsed_time)
    
    xgb_model_stat = pd.DataFrame({"Model Name" : ["XGBoost"],
                                   "Balanced Accuracy(%)": bal_acc,
                                   "AUC(%)": auc, 
                                   "Precision-Recall Score(%)": prec_recall_score,
                                   "Time (Sec.)": xgb_elapsed_time})
    final_result = (xgb_cv_model,xgb_model_stat)
    
    return(final_result)
    
# In[ ]:


### USER DEFINED FUNCTION: NAIVE BAYES MODEL ###

def model_naivebayes(train_x,
                    train_y,
                    test_x,
                    test_y,
                    num_col_list,
                    cat_col_list,
                    class_pos, 
                    n_cv,
                    prob_cutoff):
    
    print("\nStarting Gaussian Naive Bayes Model Devlopment\n")
    
    gnb_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('to_dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)), 
                                     ('gnb', GaussianNB())])
                                     
   
    # Model Development #
    gnb_cv_model = final_pipeline.fit(train_x, train_y)
    
    # Model Validation #
    y_pred_prob = gnb_cv_model.predict_proba(test_x)[:,1]
    y_pred_class = np.where(y_pred_prob>=prob_cutoff, 1, 0)
    y_actual_class = [1 if x == class_pos else 0 for x in test_y]
    
    # ROC Curve #
    fpr, tpr, thresholds = roc_curve(y_actual_class, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve On Test Sample (Gaussian Naive Bayes)')
    plt.show()
    
    bal_acc = round(balanced_accuracy_score(y_actual_class, y_pred_class, adjusted=True),2)*100   
    auc = round(roc_auc_score(y_actual_class, y_pred_class),2)*100
    prec_recall_score = round(average_precision_score(y_actual_class, y_pred_class, average = 'weighted'),2)*100
    
    
    gnb_end_time = time()
    gnb_elapsed_time = round(gnb_end_time - gnb_start_time,2)
    print("\nTime To Develop Naive Bayes Model: %.3f Seconds\n" % gnb_elapsed_time)
    
    gnb_model_stat = pd.DataFrame({"Model Name" : ["Naive Bayes"],
                                   "Balanced Accuracy(%)": bal_acc,
                                   "AUC(%)": auc, 
                                   "Precision-Recall Score(%)": prec_recall_score,
                                   "Time (Sec.)": gnb_elapsed_time})
    final_result = (gnb_cv_model,gnb_model_stat)
    
    return(final_result)


# In[ ]:


### SCRIPT EXECUTION ###

# Data Import #
result_import = data_import(global_source_name, global_id_var, global_dep_var)

# Random Sampling of Test & Train Data #
result_sampling = random_sampling(result_import[0], result_import[1], global_test_split)

# Logistic Regression Model #
result_lr_model = model_logistic_regression(result_sampling[0],
                                           result_sampling[2],
                                           result_sampling[1],
                                           result_sampling[3],
                                           result_import[2],
                                           result_import[3],
                                           global_postive_class,
                                           global_k_fold_cv,
                                           global_prob_cutoff,
                                           logistic_reg_c)

# Random Forest Model #
result_rf_model = model_random_forest(result_sampling[0],
                                      result_sampling[2],
                                      result_sampling[1],
                                      result_sampling[3],
                                      result_import[2],
                                      result_import[3],
                                      global_postive_class,
                                      global_k_fold_cv,
                                      global_prob_cutoff,
                                      random_forest_n_tree,
                                      random_forest_min_sample_leaf,
                                      random_forest_min_sample_split,
                                      random_forest_max_depth)

# Support Vector Machine Model #
result_svm_model = model_support_vector(result_sampling[0],
                                        result_sampling[2],
                                        result_sampling[1],
                                        result_sampling[3],
                                        result_import[2],
                                        result_import[3],
                                        global_postive_class,
                                        global_k_fold_cv,
                                        global_prob_cutoff,
                                        support_vector_c,
                                        support_vector_gamma)

# Gradient Boosting Machine Model #
result_gbm_model = model_gradient_boosting(result_sampling[0],
                                        result_sampling[2],
                                        result_sampling[1],
                                        result_sampling[3],
                                        result_import[2],
                                        result_import[3],
                                        global_postive_class,
                                        global_k_fold_cv,
                                        global_prob_cutoff,
                                        gbm_max_depth,
                                        gbm_min_sample_leaf,
                                        gbm_n_tree,
                                        gbm_learning_rate)

# XGBoost Model #
result_xgb_model = model_xgboost(result_sampling[0],
                                 result_sampling[2],
                                 result_sampling[1],
                                 result_sampling[3],
                                 result_import[2],
                                 result_import[3],
                                 global_postive_class,
                                 global_k_fold_cv,
                                 global_prob_cutoff,
                                 xgb_min_child_weight,
                                 xgb_gamma,
                                 xgb_subsample,
                                 xgb_max_depth,
                                 xgb_learning_rate,
                                 xgb_n_estimators)

# XGBoost Model #
result_gnb_model = model_naivebayes(result_sampling[0],
                                    result_sampling[2],
                                    result_sampling[1],
                                    result_sampling[3],
                                    result_import[2],
                                    result_import[3],
                                    global_postive_class,
                                    global_k_fold_cv,
                                    global_prob_cutoff)
                                 
# Collecting All Model Output #
print("\n++++++ Overall Model Summary ++++++\n")
all_model_summary = pd.DataFrame()
all_model_summary = all_model_summary.append(result_lr_model[1],ignore_index=True).append(result_rf_model[1],ignore_index=True).append(result_svm_model[1],ignore_index=True).append(result_gbm_model[1],ignore_index=True).append(result_xgb_model[1],ignore_index=True).append(result_gnb_model[1],ignore_index=True)
display(all_model_summary)

print("\n++++++ Process Completed ++++++\n")



