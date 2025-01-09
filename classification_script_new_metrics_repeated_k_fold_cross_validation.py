# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:03:59 2024

@author: Ronald
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:59:56 2024

@author: Ronald
"""

###loading all requiered Python packages
from datetime import datetime
from functools import reduce
import functools
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
#from sklearn.feature_selection import chi2, SelectKBest, f_classif
#from sklearn.model_selection import LearningCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import seaborn as sns
import os
import pickle
# load data file
#d = pd.read_csv("https://reneshbedre.github.io/assets/posts/anova/twowayanova.txt", sep="\t")
import numpy as np
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, matthews_corrcoef


from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_validate
#from sklearn.model_selection import learning_curve
#from sklearn.metrics import scorer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score, average_precision_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import classification_report
from sklearn import svm
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import root_mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold, RepeatedKFold
import time
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import scipy 
##RandomizedGridSearch




# df_OBIA_comb_OSAVI_FINAL_texture=pd.read_csv("C:/Users/ronal/OneDrive/Dokumente/Textur_extract/OBIA_OSAVI_MASK_extract_text_NEW.csv")
#################################################
print('Pandas version')
print(pd.__version__)
print('Sklearn version')
print(sk.__version__)
print('numpy version')
print(np.__version__)
print('scipy version:')
print(scipy.__version__)
print('seaborn version')
print(sns.__version__)
print('time version')
print(time.__version__)
print('datetime version')
print(datetime.__version__)
print('pickle version')
print(pickle.__version__)
print('functools version')
print(functools.__version__)
print('os version')
print(os.__version__)
#print()


###########1 loading of the datasets and subsetting the dataset in the seven feature groups
##selection of the features based on literature research and correlation strength (after Pearson) with the Growth Classes after Porten [43]
###input file is the csv which was generated through the extraction of the generated features with the vine- row mask and the spatial aggregation
#file_name: "OBIA_OSAVI_mit_Texture_NEW.csv"- together in the directory sripts with the script itself- this is the file path after geoprocessing/ spectral, structural and texture feature extraction
#exchange this path with your path
df_OBIA_comb_OSAVI_FINAL_texture = pd.read_csv("C:/Users/Ronald/Documents/Model_metrics_new_Pub_1/OBIA_OSAVI_mit_Texture_NEW.csv")



df_OBIA_comb_OSAVI_FINAL_texture.columns.tolist()

####select_final_features
df_OBIA_comb_OSAVI_FINAL_texture_sf=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean', '_CHM_max', '_CHM_stdev', '_CHM_range', '_CHM_count','V_OM', '_Contrastmean', '_Correlationmean', '_Entropymean', '_ASMmean']] 

# load dataframes with spectral-, structural and texture feature combination
# spectral
###########standard scaler preprocessing
scaler=StandardScaler()

df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean', '_CHM_max', '_CHM_stdev', '_CHM_range', '_CHM_count', '_Contrastmean', '_Correlationmean', '_Entropymean', '_ASMmean']] = scaler.fit_transform(df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean', '_CHM_max', '_CHM_stdev', '_CHM_range', '_CHM_count', '_Contrastmean', '_Correlationmean', '_Entropymean', '_ASMmean']])

###############

#df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean']]=scaler.fit_transform(df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean', '_CHM_mean', '_CHM_max', '_CHM_stdev', '_CHM_range', '_CHM_count', 'V_OM']])


df_OBIA_comb_OSAVI_FINAL_texture_sub_1 = df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean', 'W_03_08_i']].dropna()
# df_OBIA_comb_OSAVI_FINAL_texture_sub_1.shape[0]



#df_OBIA_comb_OSAVI_FINAL_texture_sub_1 = df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean', 'W_03_08_i']]

#df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean']]=scaler.fit_transform(df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean']])


Y_1 = df_OBIA_comb_OSAVI_FINAL_texture_sub_1[['W_03_08_i']].values
Y_1.shape[0]
#spectral+ structural
X_1 = df_OBIA_comb_OSAVI_FINAL_texture_sub_1.drop('W_03_08_i', axis=1).values
#feats_1 = df_OBIA_comb_OSAVI_FINAL_texture_sub_1.drop('W_03_08_i', axis=1).columns.tolist()



df_OBIA_comb_OSAVI_FINAL_texture_sub_2 = df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean', '_CHM_mean', '_CHM_max', '_CHM_stdev', '_CHM_range', '_CHM_count', 'V_OM', 'W_03_08_i']].dropna()


#df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean']]=scaler.fit_transform(df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean', '_CHM_mean', '_CHM_max', '_CHM_stdev', '_CHM_range', '_CHM_count', 'V_OM']])


#feats_2 = df_OBIA_comb_OSAVI_FINAL_texture_sub_2.drop('W_03_08_i', axis=1).columns.tolist()
#print(feats_2)
Y_2 = df_OBIA_comb_OSAVI_FINAL_texture_sub_2[['W_03_08_i']].values
#spectral+ structural
X_2 = df_OBIA_comb_OSAVI_FINAL_texture_sub_2.drop('W_03_08_i', axis=1)

######spectral+ structural + texture

df_OBIA_comb_OSAVI_FINAL_texture_sub_3 = df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean', '_CHM_max', '_CHM_stdev', '_CHM_range', '_CHM_count','V_OM', '_Contrastmean', '_Correlationmean', '_Entropymean', '_ASMmean', 'W_03_08_i']].dropna()

Y_3 = df_OBIA_comb_OSAVI_FINAL_texture_sub_3[['W_03_08_i']].values

#spectral+ structural
X_3 = df_OBIA_comb_OSAVI_FINAL_texture_sub_3.drop('W_03_08_i', axis=1)


#####################################################################################
# only structural
df_OBIA_comb_OSAVI_FINAL_texture_sub_4 = df_OBIA_comb_OSAVI_FINAL_texture[['_CHM_mean', '_CHM_max', '_CHM_stdev', '_CHM_range', '_CHM_count', 'V_OM', 'W_03_08_i']].dropna()
#feats_4 = df_OBIA_comb_OSAVI_FINAL_texture_sub_2.drop('W_03_08_i', axis=1).columns.tolist()
#print(feats_4)
Y_4 = df_OBIA_comb_OSAVI_FINAL_texture_sub_4[['W_03_08_i']].values
#spectral+ structural
X_4 = df_OBIA_comb_OSAVI_FINAL_texture_sub_4.drop('W_03_08_i', axis=1)



#####################################################################################
# only textural


df_OBIA_comb_OSAVI_FINAL_texture_sub_5 = df_OBIA_comb_OSAVI_FINAL_texture[[ '_Contrastmean', '_Correlationmean', '_Entropymean', '_ASMmean', 'W_03_08_i']].dropna()

Y_5 = df_OBIA_comb_OSAVI_FINAL_texture_sub_5[['W_03_08_i']].values

#spectral+ structural
X_5 = df_OBIA_comb_OSAVI_FINAL_texture_sub_5.drop('W_03_08_i', axis=1)
#feats_5 = X_1 = df_OBIA_comb_OSAVI_FINAL_texture_sub_5.drop('W_03_08_i', axis=1).columns.tolist()
#rint(feats_5)


##########################################
# spectral+texture

df_OBIA_comb_OSAVI_FINAL_texture_sub_6 = df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean', '_Contrastmean', '_Correlationmean', '_Entropymean', '_ASMmean', 'W_03_08_i']].dropna()

Y_6 = df_OBIA_comb_OSAVI_FINAL_texture_sub_6[['W_03_08_i']].values

#spectral+ structural
X_6 = df_OBIA_comb_OSAVI_FINAL_texture_sub_6.drop('W_03_08_i', axis=1)
#feats_6 = X_1 = df_OBIA_comb_OSAVI_FINAL_texture_sub_6.drop('W_03_08_i', axis=1).columns.tolist()
#print(feats_5)
# structural+ texture features

#df_OBIA_comb_OSAVI_FINAL_texture_sub_7 = df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean', '_NDVI_5_3mean', '_OSAVImean', '_GNDVImean', '_NDWImean', '_TSAVImean', '_CHM_mean', '_CHM_max', '_CHM_stdev', '_CHM_range', '_CHM_count', 'V_OM', 'W_03_08_i']].dropna()

Y_6 = df_OBIA_comb_OSAVI_FINAL_texture_sub_6[['W_03_08_i']].values


#spectral+ structural
X_6 = df_OBIA_comb_OSAVI_FINAL_texture_sub_6.drop('W_03_08_i', axis=1)

############################

########################structural + texture


df_OBIA_comb_OSAVI_FINAL_texture_sub_7 = df_OBIA_comb_OSAVI_FINAL_texture[['_CHM_mean', '_CHM_max', '_CHM_stdev', '_CHM_range','V_OM', '_CHM_count', '_Contrastmean', '_Correlationmean', '_Entropymean', '_ASMmean', 'W_03_08_i']].dropna()

Y_7 = df_OBIA_comb_OSAVI_FINAL_texture_sub_7[['W_03_08_i']].values

#spectral+ structural
X_7 = df_OBIA_comb_OSAVI_FINAL_texture_sub_7.drop('W_03_08_i', axis=1)

#feats_7 = X_1 = df_OBIA_comb_OSAVI_FINAL_texture_sub_3.drop('W_03_08_i', axis=1).columns.tolist()


###########################################################################################


Y = df_OBIA_comb_OSAVI_FINAL_texture[['W_03_08_i']].values


#################################################################
#3. Defining the classification pipeline and parameters for both classifiers clf and random grid search
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("svm", SVC())])

# param_grid_SVC_random=[{
# "C":[0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
# "gamma": [0.001, 0.01, 0.1, 1, 10],}]
# "kernel":['linear', 'rbf']
# param_grid = {
#     "svm__C": [0.001, 0.01, 0.1, 1, 10],
#     "svm__gamma": [0.001, 0.01, 0.1, 1, 10]}

# # })

# param_grid_SVC=[
# {"C":[0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
# "gamma": [0.001, 0.01, 0.1, 1, 10],
# "kernel":['linear']},
# {"C":[0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
# "gamma": [0.001, 0.01, 0.1, 1, 10],
# "kernel":['rbf']},
# ]

# param_grid_SVC = {'C': [0.1, 1, 10, 100, 100],
#                   'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#                   'kernel': ['linear', 'rbf']}

param_grid_SVC = {
    "C": [0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
    "gamma": [0.001, 0.01, 0.1, 1, 10],
    "kernel": ['linear', 'rbf']

}

# # ramdomized grid search
# param_grid_SVC_random = [{
#     "C": [0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
#     "gamma": [0.001, 0.01, 0.1, 1, 10],
#     # "kernel":['linear', 'rbf']

# }]


# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("svm", SVC()),

# ])


# clf = GridSearchCV(pipeline, param_grid={
#     "svm__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     "svm__gamma": [0.001, 0.01, 0.1, 1, 10],


# })

# clf = GridSearchCV(pipeline, param_grid={
#     "svm__C": [0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
#     "svm__gamma": [0.001, 0.01, 0.1, 1, 10],
#     "kernel": ['linear', 'rbf']

# })


# # clf=GridSearchCV(pipeline, param_grid={
# # "svm__C":[0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
# # "svm__gamma": [0.001, 0.01, 0.1, 1, 10],
# # "kernel":['linear', 'poly', 'rbf', 'sigmoid']

# # })

# param_grid_SVC = {
#     "C": [0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
#     "gamma": [0.001, 0.01, 0.1, 1, 10],
#     "kernel": ['linear', 'poly', 'rbf', 'sigmoid']

# }



clf = GridSearchCV(pipeline, param_grid_SVC)
clf = GridSearchCV(SVC(), param_grid_SVC, refit=True, verbose=3)
#     )

#m1k = clf.fit(X_train_K1, Y_train_K1.ravel())


param_grid2 = [{'n_estimators': [25, 50, 100, 150, 300, 500],
                'max_features': ['sqrt', 'log2', None],
               'max_depth': [3, 6, 9, 15, 20, 30],
                'max_leaf_nodes': [3, 6, 9],
                'max_samples':[2, 4, 6],
                'min_samples_leaf':[1, 2, 4],
                'criterion':['entropy', 'gini']
                }]

# param_grid2={[
#                 'max_features': ['sqrt', 'log2', None],
#                 'max_depth': [3,6, 9, 15, 20, 30],
#                 'max_leaf_nodes': [3,6,9],
#                 'min_samples_split':[2,4,6],
#                 'max_samples_leaf':[1,2,4],
#                 'criterion':['entropy', 'gini']]}

###random


# best_params_ grid search with accuracy and best parameter search for this parameter
param_grid_RF_random_search = RandomizedSearchCV(RandomForestClassifier(),
                                                 param_grid2,
                                                 cv=5,
                                                 scoring='accuracy',
                                                 n_jobs=-1
                                                 )


####################################################################






param_grid_RF_random_search = RandomizedSearchCV(RandomForestClassifier(),
                                                     param_grid2,
                                                     cv=5,
                                                     scoring='accuracy',
                                                     n_jobs=-1, return_train_score=True
                                                     )


###for loop 
###4 achieve the machine- learning classification in for loop
#using cross validate and cross val score from sklearn packages 
##4. metris were stored in lists for each ru
    
##define the SVM classifiers with hyperparameter tuning grid search
classifier_model_list_SVM=[clf,clf,clf, clf ,clf, clf,clf]
print(len(classifier_model_list_SVM))
#define the Random forest classifier with hyperparameter tuning wth hyperparameter Grid search 
classifier_model_list_RF=[param_grid_RF_random_search,param_grid_RF_random_search, param_grid_RF_random_search, param_grid_RF_random_search, param_grid_RF_random_search, param_grid_RF_random_search, param_grid_RF_random_search]

print(len(classifier_model_list_SVM))
param_grid_RF_random_search
##create the lists for the categorical columns of the dataframe for later grouping of the Model output metrics
model_name_L_SVM=['SVM_1','SVM_2', 'SVM_3', 'SVM_4', 'SVM_5', 'SVM_6', 'SVM_7'] 
print(len(model_name_L_SVM))
model_name_L_RF=['RF_1', 'RF_2','RF_3', 'RF_4', 'RF_5', 'RF_6','RF_7']
print(len(model_name_L_RF))
################################################
data_L_X=[X_1,X_2, X_3, X_4, X_5, X_6, X_7]
print(len(data_L_X))
data_L_Y=[Y_1,Y_2, Y_3, Y_4, Y_5, Y_6, Y_7]
print(len(data_L_Y))
zipped_data=list(zip(data_L_X,data_L_Y ))

#[]

###create the lists for the different matrics
metric_list_test_SVM_Accuracy=[]
metric_list_train_SVM_Accuracy=[]
metric_list_test_SVM_RMSE=[]
metric_list_train_SVM_RMSE=[]
metric_list_test_SVM_F1_weighted=[]
metric_list_train_SVM_F1_weighted=[]


metric_list_test_RF_Accuracy=[]
metric_list_train_RF_Accuracy=[]
metric_list_test_RF_RMSE=[]
metric_list_train_RF_RMSE=[]
metric_list_test_RF_F1_weighted=[]
metric_list_train_RF_F1_weighted=[]

#######################################
#######################################
metric_name_list=[]
model_name_list=[]
len(model_name_L_SVM)
print(len(classifier_model_list_SVM))
print(len(classifier_model_list_RF))
print(len(data_L_Y))
print(len(data_L_X))
print(len(metric_list_test_SVM_Accuracy))
#print(len(metric_list_test_SVM_Accuracy[0]))
################################################################################
print(len(data_X))
print(len(data_Y))

classifier_M_NEW=['RF 1', 'RF 2', 'RF 3', 'RF 4', 'RF 5', 'RF 6', 'RF 7',
                  'SVM 1', 'SVM 2', 'SVM 3', 'SVM 4', 'SVM 5', 'SVM 6', 'SVM 7']
print(len(classifier_M_NEW))

classifier_LL=[param_grid_RF_random_search, param_grid_RF_random_search,
                param_grid_RF_random_search, param_grid_RF_random_search, param_grid_RF_random_search,
                param_grid_RF_random_search, param_grid_RF_random_search, clf, clf, clf, clf, clf,
                clf, clf]

print(len(classifier_LL))

###define the test- and training data sets for each feature group

data_X=[X_1, X_2, X_3 , X_4, X_5 ,X_6 , X_7 ,X_1, X_2, X_3 , X_4, X_5 ,X_6 , X_7]
data_Y=[Y_1, Y_2, Y_3 , Y_4, Y_5 ,Y_6 , Y_7 ,Y_1, Y_2, Y_3 , Y_4, Y_5 ,Y_6 ,Y_7]
classifier_model_list_RF



##run the for loop for model fitting and classification output with repeated k- fold cross validation

Accuracy_L_mean=[]
Accuracy_L_L=[]
Accuracy_cros_val_test=[]
Accuracy_cros_val_train=[]
Accuracy_cross_val_score=[]
dict_scorescross_val_predict={}
y_pred_L=[]
for i in range(len(data_X)):
    Y_pred = cross_val_predict(classifier_LL[i], data_X[i], data_Y[i].ravel(), cv=5,method='predict')
    accuracy_cval=cross_validate(classifier_LL[i], data_X[i], data_Y[i].ravel(), cv=RepeatedKFold(n_repeats=5), scoring=('accuracy'), return_train_score=True)
    accuracy_cval_test=accuracy_cval['test_score']
    accuracy_cval_train=accuracy_cval['train_score']
    #['train_score']
    
    
    accuracy_cvs=cross_val_score(classifier_LL[i], data_X[i], data_Y[i].ravel(), cv=RepeatedKFold(n_repeats=5), scoring='accuracy')
    #accuracy_score_proba = cross_val_predict(classifier_LL[i], data_X[i], data_Y[i].ravel(), cv=5,method='predict_proba')
    #accuracy_score_log_proba = cross_val_predict(classifier_LL[i], data_X[i], data_Y[i].ravel(), cv=5,method='predict_log_proba')
    #accuracy_score_decision_function = cross_val_predict(classifier_LL[i], data_X[i], data_Y[i].ravel(), cv=5,method='decision_function')
    accuracy_cval_test_mean=np.mean(accuracy_cval_test)
    accuracy_cval_train_mean=np.mean(accuracy_cval_train)
    
    accuracy_score_mean_cvs=np.mean(accuracy_cvs)
    
    Accuracy_cross_val_score.append(accuracy_score_mean_cvs)
    
    dict_scorescross_val_predict.update({classifier_M_NEW[i]:[classifier_M_NEW[i],Y_pred,accuracy_cval_test_mean,accuracy_cval_train_mean, accuracy_score_mean_cvs]})
    Accuracy_L_mean.append(accuracy_score_mean_cvs)
    Accuracy_L_L.append(accuracy_score)
    Accuracy_cros_val_test.append(accuracy_cval_test_mean)
    Accuracy_cros_val_train.append(accuracy_cval_train_mean)
    y_pred_L.append(Y_pred)

    
Accuracy_cross_val_score

#Accuracy_L_L

###################################
###################################

##sort the classifier based on the accuracy output score
    
pred_metrics_cross_val_predic=list(zip(classifier_M_NEW, Accuracy_L_mean, Accuracy_L_L, data_X, data_Y, Accuracy_cros_val_test, Accuracy_cros_val_train,Accuracy_cross_val_score ))


pred_metrics_cross_val_predic=list(zip(classifier_M_NEW, Accuracy_L_mean, Accuracy_L_L, data_X, data_Y, Accuracy_cros_val_test, Accuracy_cros_val_train,Accuracy_cross_val_score, y_pred_L ))

#print(pred_metrics_cross_val_predic[8])



##################
################

################5. Selecting the model with the best overall accurracy for computation of the 
##confusion matrix/ class specific accuracies
for i in pred_metrics_cross_val_predic:
          print(i[0],i[1],i[6], i[5])

pred_metrics_cross_val_predic_sorted=sorted(pred_metrics_cross_val_predic, key= lambda x: x[6])




#################select the model with the best overall accuracy of the testdata set
print('The model with the best overall Accuracy from cross val predict is:')
X_train_best_model=[]
Y_train_best_model=[]
Y_predicted_best_model=[]
cvs_best_model_acc=[]
cvalidate_best_model_test_acc=[]
validate_best_model_train_acc=[]
for i in pred_metrics_cross_val_predic_sorted[-1:]:
    print(i[0], 'Acccuracy Test Score cross validate : ', i[5],'Acccuracy Train Score cross validate: ', i[6] )
    X_train_best_model.append(i[3])
    Y_train_best_model.append(i[4])
    Y_predicted_best_model.append(i[8])
    cvs_best_model_acc.append(i[2])
    cvalidate_best_model_test_acc.append(i[5]*100)
    validate_best_model_train_acc.append(i[6]*100)
    
print(cvalidate_best_model_test_acc)
print(validate_best_model_train_acc)
for i in pred_metrics_cross_val_predic_sorted[-1:]:
    print(i[3])
    
print(Y_train_best_model)  
print(Y_predicted_best_model)

    
###build confusion matrix of the best model (with the highest accuracy) 
    
#Y_SVM_5_cross_val_predict = cross_val_predict(clf, X_5, Y_5.ravel(), cv=5)

##Beispielhaft SVM 3 für das bauen der confusionamatrix verwenden


###6. Calculation of the confusion matrix/ class specific accuracy
#calculate the User and Produser accuracy


conf_matrix_5_best_Model = confusion_matrix(Y_train_best_model[0], Y_predicted_best_model[0])

print(conf_matrix_5_best_Model)

cm_df_best_Model_1 = conf_matrix_5_best_Model.astype('float')/conf_matrix_5_best_Model.sum(axis=1)[:, np.newaxis]
cm_df_best_Model_1_perc = (100*conf_matrix_5_best_Model.astype('float')/conf_matrix_5_best_Model.sum(axis=1)[:, np.newaxis])

################################################################
print(cm_df_best_Model_1)
print(cm_df_best_Model_1_perc)
#conf_matrix_L.append(cm_df_SVM_5)
###################################

cm_df_best_Model_2_perc = (100*conf_matrix_5_best_Model.astype('float')/conf_matrix_5_best_Model.sum(axis=0)[:, np.newaxis])
print(cm_df_best_Model_2_perc)
#conf_matrix_L.append(cm_df_SVM_5)


#############################
cm_df_SVM_5_df = pd.DataFrame.from_records(cm_df_best_Model_1_perc)
print(cm_df_SVM_5_df )
cm_df_SVM_5_df.rename(columns={0: '0_W_P', 1: '1_W_P', 2: '3_W_P',
                      3: '5_W_P', 4: '7_W_P', 5: '9_W_P'}, inplace=True)
print(cm_df_SVM_5_df)
###rename index column

print(cm_df_SVM_5_df)

# cm_df_SVM_1_df_=(cm_df_SVM_1_df.columns=(['0_W', '1_W', '3_W', '5W', '7_W', '9_W'])
sumpa = cm_df_SVM_5_df.sum(axis=0)
print(sumpa)
sumna = cm_df_SVM_5_df.sum(axis=1)
print(sumna)
type(sumna.tolist())
print(sumna)
sumna_L=sumna.tolist()
print(sumna_L)
sumna_L.append('-')
print(sumna_L)


new_row={'0_W_P': sumpa[0], '1_W_P':sumpa[1] ,'3_W_P': sumpa[2],'5_W_P': sumpa[3], 
         '7_W_P': sumpa[4],'9_W_P': sumpa[0]}

cm_df_SVM_5_df=cm_df_SVM_5_df.append(new_row, ignore_index=True)
print(cm_df_SVM_5_df)

#cm_df_SVM_5_df.loc[(cm_df_SVM_5_df.index)] = sumpa
cm_df_SVM_5_df['SUM_U']=sumna_L
print(cm_df_SVM_5_df)

#cm_df_SVM_5_df.loc[len(cm_df_SVM_5_df)] = sumpa
# cm_df_SVM_1_df_ = pd.DataFrame(cm_df_SVM_1_df, columns=[
#                                '0_W_P''1_W', '3_W_P', '5_W_P,', '7_W_P', '9_W_P'])

cm_df_SVM_5_df.index=['0_W_L', '1_W_L','3_W_L', '5_W_L', '7_W_L', '9_W_L','SUM_P' ]

print(cm_df_SVM_5_df)
##calculate user and Produser Accuracy
###Useraccuracy
Useracc = []

Useeracc1 = (cm_df_SVM_5_df.iat[0,0]/cm_df_SVM_5_df.iat[0,6])*100
print(cm_df_SVM_5_df.iat[0,0])
print(cm_df_SVM_5_df.iat[0,6])
Useracc.append(Useeracc1)
Useeracc2 = (cm_df_SVM_5_df.iat[1,1]/cm_df_SVM_5_df.iat[1,6])*100
Useracc.append(Useeracc2)
Useeracc3 = (cm_df_SVM_5_df.iat[2,2]/cm_df_SVM_5_df.iat[2,6])*100
Useracc.append(Useeracc3)
Useeracc4 = (cm_df_SVM_5_df.iat[3,3]/cm_df_SVM_5_df.iat[3,6])*100
Useracc.append(Useeracc4)
Useeracc5 = (cm_df_SVM_5_df.iat[4,4]/cm_df_SVM_5_df.iat[4,6])*100
Useracc.append(Useeracc5)
Useeracc6 = (cm_df_SVM_5_df.iat[5,5]/cm_df_SVM_5_df.iat[5,6])*100
Useracc.append(Useeracc6)
print(Useracc)
Useracc.append('-')
print(Useracc)


cm_df_SVM_5_df['UA']=Useracc

print(cm_df_SVM_5_df)

#Produceraccuracy

Produseracc = []

Pro_Useeracc1 = (cm_df_SVM_5_df.iat[0,0]/cm_df_SVM_5_df.iat[6,0])*100
Produseracc.append(Pro_Useeracc1)
Pro_Useeracc2 = (cm_df_SVM_5_df.iat[1,1]/cm_df_SVM_5_df.iat[6,1])*100
Produseracc.append(Pro_Useeracc2)
Pro_Useeracc3= (cm_df_SVM_5_df.iat[2,2]/cm_df_SVM_5_df.iat[6,2])*100
Produseracc.append(Pro_Useeracc3)
Pro_Useeracc4 = (cm_df_SVM_5_df.iat[3,3]/cm_df_SVM_5_df.iat[6,3])*100
Produseracc.append(Pro_Useeracc4)
Pro_Useeracc5 = (cm_df_SVM_5_df.iat[4,4]/cm_df_SVM_5_df.iat[6,4])*100
Produseracc.append(Pro_Useeracc5)
Pro_Useeracc6 = (cm_df_SVM_5_df.iat[5,5]/cm_df_SVM_5_df.iat[6,5])*100
Produseracc.append(Pro_Useeracc6)
print(len(Produseracc))
Produseracc.append('-')
Produseracc.append('-')
print(Produseracc)
cm_df_SVM_5_df.columns.tolist()
###########################################################
##############################################################

new_row_2={'0_W_P': Produseracc[0], '1_W_P':Produseracc[1] ,'3_W_P': Produseracc[2],'5_W_P': Produseracc[3], 
         '7_W_P': Produseracc[4],'9_W_P': Produseracc[5], 'SUM_U':
            Produseracc[6], 'UA':Produseracc[7] }

cm_df_SVM_5_df=cm_df_SVM_5_df.append(new_row_2, ignore_index=True)

print(cm_df_SVM_5_df)
#
print(len(Produseracc ))
print(cm_df_SVM_5_df.shape)

Produseracc.append(Useeracc6)
cm_df_SVM_5_df['UA']=Useracc   

####add different Accuracy from cross val score/ cross validate (Train/Test)
   

print(cvalidate_best_model_test_acc)
print(validate_best_model_train_acc)

cm_df_SVM_5_df.iat[6,7]='Accuracies'
cm_df_SVM_5_df.iat[7,7]='CValidate- Test: '+str(cvalidate_best_model_test_acc[0])+' %'+' '+'CValidate- Train: '+str(validate_best_model_train_acc[0])+' %'

cm_df_SVM_5_df.index=['0_W_L', '1_W_L','3_W_L', '5_W_L', '7_W_L', '9_W_L','SUM_P', 'PA' ]

cm_df_SVM_5_df.to_csv('C:/Users/Ronald/Documents/confusion_matrizen_new/cf_axis_0.new_perc_train_pred_complete.csv')
cm_df_SVM_5_df.to_excel('C:/Users/Ronald/Documents/confusion_matrizen_new/cf_axis_0.new_perc_train_pred_complete.xlsx')


#################
########axis =1

cm_df_SVM_5_df = pd.DataFrame.from_records(cm_df_best_Model_2_perc)
print(cm_df_SVM_5_df )
cm_df_SVM_5_df.rename(columns={0: '0_W_P', 1: '1_W_P', 2: '3_W_P',
                      3: '5_W_P', 4: '7_W_P', 5: '9_W_P'}, inplace=True)
print(cm_df_SVM_5_df)
###rename index column



print(cm_df_SVM_5_df)

# cm_df_SVM_1_df_=(cm_df_SVM_1_df.columns=(['0_W', '1_W', '3_W', '5W', '7_W', '9_W'])
sumpa = cm_df_SVM_5_df.sum(axis=0)
print(sumpa)
sumna = cm_df_SVM_5_df.sum(axis=1)
print(sumna)
type(sumna.tolist())
print(sumna)
sumna_L=sumna.tolist()
print(sumna_L)
sumna_L.append('-')
print(sumna_L)

new_row={'0_W_P': sumpa[0], '1_W_P':sumpa[1] ,'3_W_P': sumpa[2],'5_W_P': sumpa[3], 
         '7_W_P': sumpa[4],'9_W_P': sumpa[0]}

cm_df_SVM_5_df=cm_df_SVM_5_df.append(new_row, ignore_index=True)
print(cm_df_SVM_5_df)

#cm_df_SVM_5_df.loc[(cm_df_SVM_5_df.index)] = sumpa
cm_df_SVM_5_df['SUM_U']=sumna_L
print(cm_df_SVM_5_df)

#cm_df_SVM_5_df.loc[len(cm_df_SVM_5_df)] = sumpa
# cm_df_SVM_1_df_ = pd.DataFrame(cm_df_SVM_1_df, columns=[
#                                '0_W_P''1_W', '3_W_P', '5_W_P,', '7_W_P', '9_W_P'])

cm_df_SVM_5_df.index=['0_W_L', '1_W_L','3_W_L', '5_W_L', '7_W_L', '9_W_L','SUM_P' ]

print(cm_df_SVM_5_df)
##calculate user and Produser Accuracy
###Useraccuracy
Useracc = []

Useeracc1 = (cm_df_SVM_5_df.iat[0,0]/cm_df_SVM_5_df.iat[0,6])*100
print(cm_df_SVM_5_df.iat[0,0])
print(cm_df_SVM_5_df.iat[0,6])
Useracc.append(Useeracc1)
Useeracc2 = (cm_df_SVM_5_df.iat[1,1]/cm_df_SVM_5_df.iat[1,6])*100
Useracc.append(Useeracc2)
Useeracc3 = (cm_df_SVM_5_df.iat[2,2]/cm_df_SVM_5_df.iat[2,6])*100
Useracc.append(Useeracc3)
Useeracc4 = (cm_df_SVM_5_df.iat[3,3]/cm_df_SVM_5_df.iat[3,6])*100
Useracc.append(Useeracc4)
Useeracc5 = (cm_df_SVM_5_df.iat[4,4]/cm_df_SVM_5_df.iat[4,6])*100
Useracc.append(Useeracc5)
Useeracc6 = (cm_df_SVM_5_df.iat[5,5]/cm_df_SVM_5_df.iat[5,6])*100
Useracc.append(Useeracc6)
print(Useracc)
Useracc.append('-')
print(Useracc)


cm_df_SVM_5_df['UA']=Useracc
print(Useracc)
Useracc.append('-')

print(cm_df_SVM_5_df)

#Produceraccuracy

Produseracc = []

Pro_Useeracc1 = (cm_df_SVM_5_df.iat[0,0]/cm_df_SVM_5_df.iat[6,0])*100
Produseracc.append(Pro_Useeracc1)
Pro_Useeracc2 = (cm_df_SVM_5_df.iat[1,1]/cm_df_SVM_5_df.iat[6,1])*100
Produseracc.append(Pro_Useeracc2)
Pro_Useeracc3= (cm_df_SVM_5_df.iat[2,2]/cm_df_SVM_5_df.iat[6,2])*100
Produseracc.append(Pro_Useeracc3)
Pro_Useeracc4 = (cm_df_SVM_5_df.iat[3,3]/cm_df_SVM_5_df.iat[6,3])*100
Produseracc.append(Pro_Useeracc4)
Pro_Useeracc5 = (cm_df_SVM_5_df.iat[4,4]/cm_df_SVM_5_df.iat[6,4])*100
Produseracc.append(Pro_Useeracc5)
Pro_Useeracc6 = (cm_df_SVM_5_df.iat[5,5]/cm_df_SVM_5_df.iat[6,5])*100
Produseracc.append(Pro_Useeracc6)
print(len(Produseracc))
Produseracc.append('-')
Produseracc.append('-')
print(Produseracc)
cm_df_SVM_5_df.columns.tolist()
###########################################################
##############################################################

new_row_2={'0_W_P': Produseracc[0], '1_W_P':Produseracc[1] ,'3_W_P': Produseracc[2],'5_W_P': Produseracc[3], 
         '7_W_P': Produseracc[4],'9_W_P': Produseracc[5], 'SUM_U':
            Produseracc[6], 'UA':Produseracc[7] }

cm_df_SVM_5_df=cm_df_SVM_5_df.append(new_row_2, ignore_index=True)

print(cm_df_SVM_5_df)
#
print(len(Produseracc ))
print(cm_df_SVM_5_df.shape)

#Produseracc.append(Useeracc6)
cm_df_SVM_5_df['UA']=Useracc   

####add different Accuracy from cross val score/ cross validate (Train/Test)
   

print(cvalidate_best_model_test_acc)
print(validate_best_model_train_acc)

cm_df_SVM_5_df.iat[6,7]='Accuracies'
cm_df_SVM_5_df.iat[7,7]='CValidate- Test: '+str(cvalidate_best_model_test_acc[0])+' %'+' '+'CValidate- Train: '+str(validate_best_model_train_acc[0])+' %'

cm_df_SVM_5_df.index=['0_W_L', '1_W_L','3_W_L', '5_W_L', '7_W_L', '9_W_L','SUM_P', 'PA' ]


###export of the best overall model for accuracy and export (here the 5 model is equivalent to the 
##seven model decribed in the paper, with all input feature groups)
cm_df_SVM_5_df.to_csv('C:/Users/Ronald/Documents/confusion_matrizen_new/cf_axis_1.new_perc_train_pred_complete.csv')
cm_df_SVM_5_df.to_excel('C:/Users/Ronald/Documents/confusion_matrizen_new/cf_axis_1.new_perc_train_pred_complete.xlsx')

#########################################
#######################################




conf_matrix_5_best_Model = confusion_matrix(Y_train_best_model[0], Y_predicted_best_model[0])

print(conf_matrix_5_best_Model)



print(cm_df_best_Model_1)
print(cm_df_best_Model_1_perc)
#conf_matrix_L.append(cm_df_SVM_5)
###################################


cm_df_best_Model_2_perc=conf_matrix_5_best_Model

print(cm_df_best_Model_2_perc)
#conf_matrix_L.append(cm_df_SVM_5)


#############################
cm_df_SVM_5_df = pd.DataFrame.from_records(cm_df_best_Model_1_perc)
print(cm_df_SVM_5_df )
cm_df_SVM_5_df.rename(columns={0: '0_W_P', 1: '1_W_P', 2: '3_W_P',
                      3: '5_W_P', 4: '7_W_P', 5: '9_W_P'}, inplace=True)
print(cm_df_SVM_5_df)
###rename index column



print(cm_df_SVM_5_df)

# cm_df_SVM_1_df_=(cm_df_SVM_1_df.columns=(['0_W', '1_W', '3_W', '5W', '7_W', '9_W'])
sumpa = cm_df_SVM_5_df.sum(axis=0)
print(sumpa)
sumna = cm_df_SVM_5_df.sum(axis=1)
print(sumna)
type(sumna.tolist())
print(sumna)
sumna_L=sumna.tolist()
print(sumna_L)
sumna_L.append('-')
print(sumna_L)

new_row={'0_W_P': sumpa[0], '1_W_P':sumpa[1] ,'3_W_P': sumpa[2],'5_W_P': sumpa[3], 
         '7_W_P': sumpa[4],'9_W_P': sumpa[0]}

cm_df_SVM_5_df=cm_df_SVM_5_df.append(new_row, ignore_index=True)
print(cm_df_SVM_5_df)

#cm_df_SVM_5_df.loc[(cm_df_SVM_5_df.index)] = sumpa
cm_df_SVM_5_df['SUM_U']=sumna_L
print(cm_df_SVM_5_df)

#cm_df_SVM_5_df.loc[len(cm_df_SVM_5_df)] = sumpa
# cm_df_SVM_1_df_ = pd.DataFrame(cm_df_SVM_1_df, columns=[
#                                '0_W_P''1_W', '3_W_P', '5_W_P,', '7_W_P', '9_W_P'])

cm_df_SVM_5_df.index=['0_W_L', '1_W_L','3_W_L', '5_W_L', '7_W_L', '9_W_L','SUM_P' ]

print(cm_df_SVM_5_df)
##calculate user and Produser Accuracy
###Useraccuracy
Useracc = []

Useeracc1 = (cm_df_SVM_5_df.iat[0,0]/cm_df_SVM_5_df.iat[0,6])*100
print(cm_df_SVM_5_df.iat[0,0])
print(cm_df_SVM_5_df.iat[0,6])
Useracc.append(Useeracc1)
Useeracc2 = (cm_df_SVM_5_df.iat[1,1]/cm_df_SVM_5_df.iat[1,6])*100
Useracc.append(Useeracc2)
Useeracc3 = (cm_df_SVM_5_df.iat[2,2]/cm_df_SVM_5_df.iat[2,6])*100
Useracc.append(Useeracc3)
Useeracc4 = (cm_df_SVM_5_df.iat[3,3]/cm_df_SVM_5_df.iat[3,6])*100
Useracc.append(Useeracc4)
Useeracc5 = (cm_df_SVM_5_df.iat[4,4]/cm_df_SVM_5_df.iat[4,6])*100
Useracc.append(Useeracc5)
Useeracc6 = (cm_df_SVM_5_df.iat[5,5]/cm_df_SVM_5_df.iat[5,6])*100
Useracc.append(Useeracc6)
print(Useracc)
Useracc.append('-')
print(Useracc)




cm_df_SVM_5_df['UA']=Useracc

print(cm_df_SVM_5_df)

#Produceraccuracy

Produseracc = []

Pro_Useeracc1 = (cm_df_SVM_5_df.iat[0,0]/cm_df_SVM_5_df.iat[6,0])*100
Produseracc.append(Pro_Useeracc1)
Pro_Useeracc2 = (cm_df_SVM_5_df.iat[1,1]/cm_df_SVM_5_df.iat[6,1])*100
Produseracc.append(Pro_Useeracc2)
Pro_Useeracc3= (cm_df_SVM_5_df.iat[2,2]/cm_df_SVM_5_df.iat[6,2])*100
Produseracc.append(Pro_Useeracc3)
Pro_Useeracc4 = (cm_df_SVM_5_df.iat[3,3]/cm_df_SVM_5_df.iat[6,3])*100
Produseracc.append(Pro_Useeracc4)
Pro_Useeracc5 = (cm_df_SVM_5_df.iat[4,4]/cm_df_SVM_5_df.iat[6,4])*100
Produseracc.append(Pro_Useeracc5)
Pro_Useeracc6 = (cm_df_SVM_5_df.iat[5,5]/cm_df_SVM_5_df.iat[6,5])*100
Produseracc.append(Pro_Useeracc6)
print(len(Produseracc))
Produseracc.append('-')
Produseracc.append('-')
print(Produseracc)
cm_df_SVM_5_df.columns.tolist()
###########################################################
##############################################################

new_row_2={'0_W_P': Produseracc[0], '1_W_P':Produseracc[1] ,'3_W_P': Produseracc[2],'5_W_P': Produseracc[3], 
         '7_W_P': Produseracc[4],'9_W_P': Produseracc[5], 'SUM_U':
            Produseracc[6], 'UA':Produseracc[7] }

cm_df_SVM_5_df=cm_df_SVM_5_df.append(new_row_2, ignore_index=True)

print(cm_df_SVM_5_df)
#
print(len(Produseracc ))
print(cm_df_SVM_5_df.shape)

Produseracc.append(Useeracc6)
#cm_df_SVM_5_df['UA']=Useracc   

####add different Accuracy from cross val score/ cross validate (Train/Test)
   

print(cvalidate_best_model_test_acc)
print(validate_best_model_train_acc)

cm_df_SVM_5_df.iat[6,7]='Accuracies'
cm_df_SVM_5_df.iat[7,7]='CValidate- Test: '+str(cvalidate_best_model_test_acc[0])+' %'+' '+'CValidate- Train: '+str(validate_best_model_train_acc[0])+' %'

cm_df_SVM_5_df.index=['0_W_L', '1_W_L','3_W_L', '5_W_L', '7_W_L', '9_W_L','SUM_P', 'PA' ]

cm_df_SVM_5_df.to_csv('C:/Users/Ronald/Documents/confusion_matrizen_new/cf_axis_0.new_counts_train_pred_complete.csv')
cm_df_SVM_5_df.to_excel('C:/Users/Ronald/Documents/confusion_matrizen_new/cf_axis_0.new_counts_train_pred_complete.xlsx')



####################################################
####################################################


classifier_model_list_SVM=[clf,clf,clf, clf ,clf, clf,clf]
print(len(classifier_model_list_SVM))
classifier_model_list_RF=[param_grid_RF_random_search,param_grid_RF_random_search, param_grid_RF_random_search, param_grid_RF_random_search, param_grid_RF_random_search, param_grid_RF_random_search, param_grid_RF_random_search]
print(len(classifier_model_list_SVM))
param_grid_RF_random_search

model_name_L_SVM=['SVM_1','SVM_2', 'SVM_3', 'SVM_4', 'SVM_5', 'SVM_6', 'SVM_7'] 
print(len(model_name_L_SVM))
model_name_L_RF=['RF_1', 'RF_2','RF_3', 'RF_4', 'RF_5', 'RF_6','RF_7']
print(len(model_name_L_RF))
################################################
data_L_X=[X_1,X_2, X_3, X_4, X_5, X_6, X_7]
print(len(data_L_X))
data_L_Y=[Y_1,Y_2, Y_3, Y_4, Y_5, Y_6, Y_7]
print(len(data_L_Y))
zipped_data=list(zip(data_L_X,data_L_Y ))

#[]
metric_list_test_SVM_Accuracy=[]
metric_list_train_SVM_Accuracy=[]
metric_list_test_SVM_RMSE=[]
metric_list_train_SVM_RMSE=[]
metric_list_test_SVM_F1_weighted=[]
metric_list_train_SVM_F1_weighted=[]


metric_list_test_RF_Accuracy=[]
metric_list_train_RF_Accuracy=[]
metric_list_test_RF_RMSE=[]
metric_list_train_RF_RMSE=[]
metric_list_test_RF_F1_weighted=[]
metric_list_train_RF_F1_weighted=[]

#######################################
#######################################
metric_name_list=[]
model_name_list=[]
len(model_name_L_SVM)
print(len(classifier_model_list_SVM))
print(len(classifier_model_list_RF))
print(len(data_L_Y))
print(len(data_L_X))
print(len(metric_list_test_SVM_Accuracy))
#print(len(metric_list_test_SVM_Accuracy[0]))
################################################################################
print(len(data_X))
print(len(data_Y))

classifier_M_NEW=['RF 1', 'RF 2', 'RF 3', 'RF 4', 'RF 5', 'RF 6', 'RF 7',
                  'SVM 1', 'SVM 2', 'SVM 3', 'SVM 4', 'SVM 5', 'SVM 6', 'SVM 7']
print(len(classifier_M_NEW))

classifier_LL=[param_grid_RF_random_search, param_grid_RF_random_search,
                param_grid_RF_random_search, param_grid_RF_random_search, param_grid_RF_random_search,
                param_grid_RF_random_search, param_grid_RF_random_search, clf, clf, clf, clf, clf,
                clf, clf]

print(len(classifier_LL))

data_X=[X_1, X_2, X_3 , X_4, X_5 ,X_6 , X_7 ,X_1, X_2, X_3 , X_4, X_5 ,X_6 , X_7]
data_Y=[Y_1, Y_2, Y_3 , Y_4, Y_5 ,Y_6 , Y_7 ,Y_1, Y_2, Y_3 , Y_4, Y_5 ,Y_6 ,Y_7]
classifier_model_list_RF


Accuracy_L_mean=[]
Accuracy_L_L=[]
Accuracy_cros_val_test=[]
Accuracy_cros_val_train=[]
Accuracy_cross_val_score=[]
dict_scorescross_val_predict={}




##BBeispiel Confusion Matrix automatisierte Generierung
#####
###

####Hier!
######################################################################
##doing the same like before but here with random grid search
###for loop for iterative production of the model metrics and get the test data fit output
#as well as the train data fit output
for i in range(0, len(model_name_L_SVM)):
    Accuracy_SVM = cross_validate(classifier_model_list_SVM[i], data_L_X[i], data_L_Y[i].ravel(), cv=RepeatedKFold(n_repeats=5), scoring=('accuracy'), return_train_score=True)
    RMSE_SVM = cross_validate(classifier_model_list_SVM[i], data_L_X[i], data_L_Y[i].ravel(), cv=RepeatedKFold(n_repeats=5), scoring=('neg_mean_squared_error'), return_train_score=True)
    F1_weighted_SVM = cross_validate(classifier_model_list_SVM[i], data_L_X[i], data_L_Y[i].ravel(), cv=RepeatedKFold(n_repeats=5), scoring=('f1_weighted'), return_train_score=True)
    Accuracy_RF = cross_validate(classifier_model_list_RF[i], data_L_X[i], data_L_Y[i].ravel(), cv=RepeatedKFold(n_repeats=5), scoring=('accuracy'), return_train_score=True)
    RMSE_RF = cross_validate(classifier_model_list_RF[i], data_L_X[i], data_L_Y[i].ravel(), cv=RepeatedKFold(n_repeats=5), scoring=('neg_mean_squared_error'), return_train_score=True)
    F1_weighted_RF = cross_validate(classifier_model_list_RF[i], data_L_X[i], data_L_Y[i].ravel(), cv=RepeatedKFold(n_repeats=5), scoring=('f1_weighted'), return_train_score=True)
    ##RMSE_score = cross_val_score(grid_search2, X_2, Y_2.ravel(), cv=RepeatedKFold(n_repeats=5), scoring='accuracy')
    
    Accuracy_test_Accuracy_SVM =  Accuracy_SVM['test_score']
    
    Accuracy_train_Accuracy_SVM =  Accuracy_SVM['train_score']
    
    Accuracy_test_Accuracy_RF =  Accuracy_RF['test_score']
    
    Accuracy_train_Accuracy_RF =  Accuracy_RF['train_score']
    
    ############calculate mean accuracies
    
    
    metric_list_test_SVM_Accuracy.append(Accuracy_test_Accuracy_SVM)
    metric_list_train_SVM_Accuracy.append(Accuracy_train_Accuracy_SVM)
    
    metric_list_test_RF_Accuracy.append(Accuracy_test_Accuracy_RF)
    metric_list_train_RF_Accuracy.append(Accuracy_train_Accuracy_RF)
    
    RMSE_test_SVM =  RMSE_SVM['test_score']
    
    # for r in RMSE_test_SVM:
    #     RMSE_perc=r/np.mean(data_L_Y[i])
    
    RMSE_train_SVM =  RMSE_SVM['train_score']
    
    RMSE_test_RF  =  RMSE_RF['test_score']
    
    RMSE_train_RF  =  RMSE_RF['train_score']
    
    
    metric_list_test_SVM_RMSE.append(RMSE_test_SVM)
    metric_list_train_SVM_RMSE.append(RMSE_train_SVM)
    
    metric_list_test_RF_RMSE.append(RMSE_test_RF)
    metric_list_train_RF_RMSE.append(RMSE_train_RF)
    
    
    F1_test_SVM =  F1_weighted_SVM['test_score']
    
    F1_train_SVM =  F1_weighted_SVM['train_score']
    
    F1_test_RF  =  F1_weighted_RF ['test_score']
    
    F1_train_RF  =  F1_weighted_RF ['train_score']
    
    
    metric_list_test_SVM_F1_weighted.append(F1_test_SVM )
    metric_list_train_SVM_F1_weighted.append(F1_train_SVM)

    metric_list_test_RF_F1_weighted.append(F1_test_RF)
    metric_list_train_RF_F1_weighted.append(F1_train_RF )

    
    
    #####################################################
    # RMSE
    ###############################
    # RMSE = cross_validate(grid_search2, X_7, Y_7.ravel(
    # ), cv=RepeatedKFold(n_repeats=5), scoring=('neg_mean_squared_error'), return_train_score=True)
    
    # RMSE_test = RF_R7_score_val_rep_no_shuffle_acc_multi_RMSE[
    #     'test_score']
    # RMSE_train = RF_R7_score_val_rep_no_shuffle_acc_multi_RMSE['train_score']
    
    # ######
    # # F1-weighted
    
    # F_1_weighted = cross_validate(grid_search2, X_7, Y_7.ravel(), cv=RepeatedKFold(n_repeats=5), scoring=('f1_weighted'), return_train_score=True)
    
    # F_1_weighted_test = RF_R7_score_val_rep_no_shuffle_acc_multi_f1_weighted['test_score']
    # F_1_weighted_train = RF_R7_score_val_rep_no_shuffle_acc_multi_f1_weighted['train_score']

print(len(metric_list_test_SVM_F1_weighted))
#print(len(metric_list_test_SVM_F1_weighted[0]))
#[]



###define the lists for metric input/ storage 
metrics_list_F1_Accuracy=[]

Modelnames_RF=['RF_1', 'RF_2', 'RF_3', 'RF_4', 'RF_5', 'RF_6', 'RF_7']
modelnames_RF_n=np.repeat(['RF_1', 'RF_2', 'RF_3', 'RF_4', 'RF_5', 'RF_6', 'RF_7'],25)
modelnames_RF_n=modelnames_RF_n.tolist()

#type(modelnames_RF_n.tolist())
print(modelnames_RF_n)

modelnames_RF_nn=modelnames_RF_n*4
print(len(modelnames_RF_nn))


Modelnames_SVM=['SVM_1', 'SVM_2', 'SVM_3', 'SVM_4', 'SVM_5', 'SVM_6', 'SVM_7']

####modelnames n
Modelnames_SVM_n=np.repeat(['SVM_1', 'SVM_2', 'SVM_3', 'SVM_4', 'SVM_5', 'SVM_6', 'SVM_7'], 25).tolist()
Modelnames_SVM_nn=Modelnames_SVM_n*4
print(len(Modelnames_SVM_nn))


Model_NN=[]

Model_NN.extend(modelnames_RF_nn)
Model_NN.extend(Modelnames_SVM_nn)
print(len(Model_NN))


n_times=7*25
n_times


##metricsnames n
metric_names_n=np.repeat(['Accuracy_Test','Accuracy_Train', 'f1-weighted_Test', 'f1-weighted_Train'],n_times).tolist()
metric_names_nn=metric_names_n*2
print(len(metric_names_nn))



RF_ACC_Test=[]
RF_ACC_Test.extend(metric_list_test_RF_Accuracy[0])
print(len(metric_list_test_RF_Accuracy))
RF_ACC_Test.extend(metric_list_test_RF_Accuracy[1])
RF_ACC_Test.extend(metric_list_test_RF_Accuracy[2])
RF_ACC_Test.extend(metric_list_test_RF_Accuracy[3])
RF_ACC_Test.extend(metric_list_test_RF_Accuracy[4])
RF_ACC_Test.extend(metric_list_test_RF_Accuracy[5])
RF_ACC_Test.extend(metric_list_test_RF_Accuracy[6])


# RF_1_ACC_mean=[]
# RF_2_ACC_mean=[]
# RF_3_ACC_mean=[]
# RF_4_ACC_mean=[]
# RF_5_ACC_mean=[]
# RF_6_ACC_mean=[]
# RF_7_ACC_mean=[]
#############mean Accuracy values RF_test from cross validate

RF_1_ACC_mean_test=np.mean(metric_list_test_RF_Accuracy[0])
RF_2_ACC_mean_test=np.mean(metric_list_test_RF_Accuracy[1])
RF_3_ACC_mean_test=np.mean(metric_list_test_RF_Accuracy[2])
RF_4_ACC_mean_test=np.mean(metric_list_test_RF_Accuracy[3])
RF_5_ACC_mean_test=np.mean(metric_list_test_RF_Accuracy[4])
RF_6_ACC_mean_test=np.mean(metric_list_test_RF_Accuracy[5])
RF_7_ACC_mean_test=np.mean(metric_list_test_RF_Accuracy[6])


print(len(RF_ACC_Test))

RF_ACC_Train=[]

RF_ACC_Train.extend(metric_list_train_RF_Accuracy[0])
RF_ACC_Train.extend(metric_list_train_RF_Accuracy[1])
RF_ACC_Train.extend(metric_list_train_RF_Accuracy[2])
RF_ACC_Train.extend(metric_list_train_RF_Accuracy[3])
RF_ACC_Train.extend(metric_list_train_RF_Accuracy[4])
RF_ACC_Train.extend(metric_list_train_RF_Accuracy[5])
RF_ACC_Train.extend(metric_list_train_RF_Accuracy[6])

#############mean Accuracy values RF_train fro, cross validate

RF_1_ACC_mean_train=np.mean(metric_list_train_RF_Accuracy[0])
RF_2_ACC_mean_train=np.mean(metric_list_train_RF_Accuracy[1])
RF_3_ACC_mean_train=np.mean(metric_list_train_RF_Accuracy[2])
RF_4_ACC_mean_train=np.mean(metric_list_train_RF_Accuracy[3])
RF_5_ACC_mean_train=np.mean(metric_list_train_RF_Accuracy[4])
RF_6_ACC_mean_train=np.mean(metric_list_train_RF_Accuracy[5])
RF_7_ACC_mean_train=np.mean(metric_list_train_RF_Accuracy[6])



print(len(RF_ACC_Train))


RF_F1_Test=[]
RF_F1_Test.extend(metric_list_test_RF_F1_weighted[0])
RF_F1_Test.extend(metric_list_test_RF_F1_weighted[1])
RF_F1_Test.extend(metric_list_test_RF_F1_weighted[2])
RF_F1_Test.extend(metric_list_test_RF_F1_weighted[3])
RF_F1_Test.extend(metric_list_test_RF_F1_weighted[4])
RF_F1_Test.extend(metric_list_test_RF_F1_weighted[5])
RF_F1_Test.extend(metric_list_test_RF_F1_weighted[6])


###mean F1 weighted test  RF models from cross validate

RF_1_F_1_weigted_mean_test=np.mean(metric_list_test_RF_F1_weighted[0])
RF_2_F_1_weigted_mean_test=np.mean(metric_list_test_RF_F1_weighted[1])
RF_3_F_1_weigted_mean_test=np.mean(metric_list_test_RF_F1_weighted[2])
RF_4_F_1_weigted_mean_test=np.mean(metric_list_test_RF_F1_weighted[3])
RF_5_F_1_weigted_mean_test=np.mean(metric_list_test_RF_F1_weighted[4])
RF_6_F_1_weigted_mean_test=np.mean(metric_list_test_RF_F1_weighted[5])
RF_7_F_1_weigted_mean_test=np.mean(metric_list_test_RF_F1_weighted[6])




RF_F1_Train=[]

RF_F1_Train.extend(metric_list_train_RF_F1_weighted[0])
RF_F1_Train.extend(metric_list_train_RF_F1_weighted[1])
RF_F1_Train.extend(metric_list_train_RF_F1_weighted[2])
RF_F1_Train.extend(metric_list_train_RF_F1_weighted[3])
RF_F1_Train.extend(metric_list_train_RF_F1_weighted[4])
RF_F1_Train.extend(metric_list_train_RF_F1_weighted[5])
RF_F1_Train.extend(metric_list_train_RF_F1_weighted[6])


###mean F1 weighted train RF models from cross validate

RF_1_F_1_weigted_mean_train=np.mean(metric_list_train_RF_F1_weighted[0])
RF_2_F_1_weigted_mean_train=np.mean(metric_list_train_RF_F1_weighted[1])
RF_3_F_1_weigted_mean_train=np.mean(metric_list_train_RF_F1_weighted[2])
RF_4_F_1_weigted_mean_train=np.mean(metric_list_train_RF_F1_weighted[3])
RF_5_F_1_weigted_mean_train=np.mean(metric_list_train_RF_F1_weighted[4])
RF_6_F_1_weigted_mean_train=np.mean(metric_list_train_RF_F1_weighted[5])
RF_7_F_1_weigted_mean_train=np.mean(metric_list_train_RF_F1_weighted[6])




SVM_ACC_Test=[]
SVM_ACC_Test.extend(metric_list_test_SVM_Accuracy[0])
SVM_ACC_Test.extend(metric_list_test_SVM_Accuracy[1])
SVM_ACC_Test.extend(metric_list_test_SVM_Accuracy[2])
SVM_ACC_Test.extend(metric_list_test_SVM_Accuracy[3])
SVM_ACC_Test.extend(metric_list_test_SVM_Accuracy[4])
SVM_ACC_Test.extend(metric_list_test_SVM_Accuracy[5])
SVM_ACC_Test.extend(metric_list_test_SVM_Accuracy[6])
print(len(SVM_ACC_Test))


#SVM_F1


######mean Accuracy test of SVM model 

SVM_1_Accuuracy_mean_test=np.mean(metric_list_test_SVM_Accuracy[0])
SVM_2_Accuuracy_mean_test=np.mean(metric_list_test_SVM_Accuracy[1])
SVM_3_Accuuracy_mean_test=np.mean(metric_list_test_SVM_Accuracy[2])
SVM_4_Accuuracy_mean_test=np.mean(metric_list_test_SVM_Accuracy[3])
SVM_5_Accuuracy_mean_test=np.mean(metric_list_test_SVM_Accuracy[4])
SVM_6_Accuuracy_mean_test=np.mean(metric_list_test_SVM_Accuracy[5])
SVM_7_Accuuracy_mean_test=np.mean(metric_list_test_SVM_Accuracy[6])






SVM_ACC_Train=[]

SVM_ACC_Train.extend(metric_list_train_SVM_Accuracy[0])
SVM_ACC_Train.extend(metric_list_train_SVM_Accuracy[1])

SVM_ACC_Train.extend(metric_list_train_SVM_Accuracy[2])
SVM_ACC_Train.extend(metric_list_train_SVM_Accuracy[3])
SVM_ACC_Train.extend(metric_list_train_SVM_Accuracy[4])
SVM_ACC_Train.extend(metric_list_train_SVM_Accuracy[5])
SVM_ACC_Train.extend(metric_list_train_SVM_Accuracy[6])

print(len(SVM_ACC_Train))
######mean Accuracy train of SVM model 

SVM_1_Accuuracy_mean_train=np.mean(metric_list_train_SVM_Accuracy[0])
SVM_2_Accuuracy_mean_train=np.mean(metric_list_train_SVM_Accuracy[1])
SVM_3_Accuuracy_mean_train=np.mean(metric_list_train_SVM_Accuracy[2])
SVM_4_Accuuracy_mean_train=np.mean(metric_list_train_SVM_Accuracy[3])
SVM_5_Accuuracy_mean_train=np.mean(metric_list_train_SVM_Accuracy[4])
SVM_6_Accuuracy_mean_train=np.mean(metric_list_train_SVM_Accuracy[5])
SVM_7_Accuuracy_mean_train=np.mean(metric_list_train_SVM_Accuracy[6])




SVM_F1_Test=[]
SVM_F1_Test.extend(metric_list_test_SVM_F1_weighted[0])
SVM_F1_Test.extend(metric_list_test_SVM_F1_weighted[1])
SVM_F1_Test.extend(metric_list_test_SVM_F1_weighted[2])
SVM_F1_Test.extend(metric_list_test_SVM_F1_weighted[3])
SVM_F1_Test.extend(metric_list_test_SVM_F1_weighted[4])
SVM_F1_Test.extend(metric_list_test_SVM_F1_weighted[5])
SVM_F1_Test.extend(metric_list_test_SVM_F1_weighted[6])


###mean F1 weighted test of SVM model

SVM_1_F1_weighted_mean_test=np.mean(metric_list_test_SVM_F1_weighted[0])
SVM_2_F1_weighted_mean_test=np.mean(metric_list_test_SVM_F1_weighted[1])
SVM_3_F1_weighted_mean_test=np.mean(metric_list_test_SVM_F1_weighted[2])
SVM_4_F1_weighted_mean_test=np.mean(metric_list_test_SVM_F1_weighted[3])
SVM_5_F1_weighted_mean_test=np.mean(metric_list_test_SVM_F1_weighted[4])
SVM_6_F1_weighted_mean_test=np.mean(metric_list_test_SVM_F1_weighted[5])
SVM_7_F1_weighted_mean_test=np.mean(metric_list_test_SVM_F1_weighted[6])


print(len(SVM_F1_Test))

SVM_F1_Train=[]

SVM_F1_Train.extend(metric_list_train_SVM_F1_weighted[0])
SVM_F1_Train.extend(metric_list_train_SVM_F1_weighted[1])
SVM_F1_Train.extend(metric_list_train_SVM_F1_weighted[2])
SVM_F1_Train.extend(metric_list_train_SVM_F1_weighted[3])
SVM_F1_Train.extend(metric_list_train_SVM_F1_weighted[4])
SVM_F1_Train.extend(metric_list_train_SVM_F1_weighted[5])
SVM_F1_Train.extend(metric_list_train_SVM_F1_weighted[6])

print(len(SVM_F1_Train))

###mean F1 weighted train of SVM model


SVM_1_F1_weighted_mean_train=np.mean(metric_list_train_SVM_F1_weighted[0])
SVM_2_F2_weighted_mean_train=np.mean(metric_list_train_SVM_F1_weighted[1])
SVM_3_F1_weighted_mean_train=np.mean(metric_list_train_SVM_F1_weighted[2])
SVM_4_F1_weighted_mean_train=np.mean(metric_list_train_SVM_F1_weighted[3])
SVM_5_F1_weighted_mean_train=np.mean(metric_list_train_SVM_F1_weighted[4])
SVM_6_F1_weighted_mean_train=np.mean(metric_list_train_SVM_F1_weighted[5])
SVM_7_F1_weighted_mean_train=np.mean(metric_list_train_SVM_F1_weighted[6])



metric_list_test_SVM_Accuracy
print(len(metric_list_test_SVM_Accuracy))
print(len(metric_list_test_SVM_Accuracy[0]))
metric_list_train_SVM_Accuracy
print(len(metric_list_train_SVM_Accuracy))
print(len(metric_list_train_SVM_Accuracy[0]))

metrics_Acc_F1=[]

###########################################
#metrics_train
RF_ACC_Test
RF_ACC_Train
RF_F1_Test
RF_F1_Train

SVM_ACC_Test
SVM_ACC_Test
SVM_F1_Test
SVM_F1_Train

metrics_Acc_F1.extend(RF_ACC_Test)
metrics_Acc_F1.extend(RF_ACC_Train)
metrics_Acc_F1.extend(RF_F1_Test)
metrics_Acc_F1.extend(RF_F1_Train)
metrics_Acc_F1.extend(SVM_ACC_Test)
metrics_Acc_F1.extend(SVM_ACC_Train)
metrics_Acc_F1.extend(SVM_F1_Test)
metrics_Acc_F1.extend(SVM_F1_Train)

print(len(metrics_Acc_F1))

metric_list_test_RF_Accuracy
metric_list_train_RF_Accuracy

metric_list_test_SVM_F1_weighted
metric_list_train_SVM_F1_weighted

metric_list_test_RF_F1_weighted
metric_list_train_RF_F1_weighted
#### construct metrics list
metrics_Acc_F1.extend(metric_list_test_RF_Accuracy)
metrics_Acc_F1.extend(metric_list_train_RF_Accuracy)
metrics_Acc_F1.extend(metric_list_test_RF_F1_weighted)
metrics_Acc_F1.extend(metric_list_train_RF_F1_weighted)

metrics_Acc_F1.extend(metric_list_test_SVM_Accuracy)
metrics_Acc_F1.extend(metric_list_train_SVM_Accuracy)
metrics_Acc_F1.extend(metric_list_test_SVM_F1_weighted)
metrics_Acc_F1.extend(metric_list_train_SVM_F1_weighted)



#df_metrics_RGS_F1_ACC

##7 create dataframe with produced metrics from last for-loop metrics result and export the dataframe or
#use the dataframe for visualization (boxplots, sigificance, bivariate, univariate statistics,
#multivariate statistics)



for i in metric_list_test_SVM_F1_weighted:
    print(i)
    
cols_dict_NEW={'Model type':Model_NN,
               'metric_type':metric_names_nn,
                   'metric_scores':metrics_Acc_F1}

df_metrics_RGS_F1_ACC = pd.DataFrame(cols_dict_NEW)

#[]
df_metrics_RGS_F1_ACC['metric_scores_%']=df_metrics_RGS_F1_ACC['metric_scores']*100

df_metrics_RGS_F1_ACC[['metric', 'dataset']]=df_metrics_RGS_F1_ACC['metric_type'].str.split('_', expand=True)



df_metrics_RGS_F1_ACC['model_dataset']=df_metrics_RGS_F1_ACC['Model type'].astype(str)+' '+ df_metrics_RGS_F1_ACC['dataset'].astype(str)




df_metrics_RGS_F1_ACC['model_dataset_mt']=df_metrics_RGS_F1_ACC['Model type'].astype(str)+' '+df_metrics_RGS_F1_ACC['metric_type'].astype(str)+' '+ df_metrics_RGS_F1_ACC['dataset'].astype(str)

df_metrics_RGS_F1_ACC['model_dataset']=df_metrics_RGS_F1_ACC['model_dataset'].str.replace('_', ' ')

df_metrics_RGS_F1_ACC[['classifier', 'number' ,'set']]=df_metrics_RGS_F1_ACC['model_dataset'].str.split(' ', expand=True)

df_metrics_RGS_F1_ACC['classifier_set']=df_metrics_RGS_F1_ACC['classifier'].astype(str)+' '+df_metrics_RGS_F1_ACC['set'].astype(str)

df_metrics_RGS_F1_ACC['metrics_sort']=df_metrics_RGS_F1_ACC['number'].astype(str)+'_'+df_metrics_RGS_F1_ACC['metric'].astype(str)


df_metrics_RGS_F1_ACC.columns.tolist()



##########Here Filtering of the dataframe

#classifier_set
#Nach diesen 
print(df_metrics_RGS_F1_ACC.model_dataset_mt.unique().tolist())
#print(df_metrics_RGS_F1_ACC_sub_RF_test.model_dataset_mt.unique().tolist())
print(df_metrics_RGS_F1_ACC.metrics_sort.unique().tolist())

############################
##Visualization of the machine-  learning classification results
#set style/ lcolor of the boxplots

box_colors = {'boxes': 'blue'}

########################

df_metrics_RGS_F1_ACC

df_metrics_RGS_F1_ACC_sub_RF_test=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['classifier_set'].str.contains('RF Test', na=False) ]


df_metrics_RGS_F1_ACC_sub_RF_test_s=df_metrics_RGS_F1_ACC_sub_RF_test.sort_values(by='metrics_sort')

df_metrics_RGS_F1_ACC_sub_RF_test_s['classifier_set_2']=df_metrics_RGS_F1_ACC_sub_RF_test_s['classifier'].astype(str)+' '+df_metrics_RGS_F1_ACC_sub_RF_test_s['metrics_sort'].astype(str)


print(df_metrics_RGS_F1_ACC_sub_RF_test_s.metrics_sort.unique().tolist())

label_M_L=df_metrics_RGS_F1_ACC_sub_RF_test_s.classifier_set_2.unique().tolist()


ax = df_metrics_RGS_F1_ACC_sub_RF_test_s.boxplot(by='model_dataset_mt', column='metric_scores_%', color=dict(boxes='blue', whiskers='red', medians='red'),
                                     boxprops=dict(linestyle='-', linewidth=2.0,
                                                   color=box_colors['boxes']),
                                     flierprops=dict(
                                         linewidth=2.0, markerfacecolor='red'),
                                     medianprops=dict(
                                         linestyle='-', linewidth=2.0, color='red'),
                                     whiskerprops=dict(
                                         linestyle='-', linewidth=2.0),
                                     capprops=dict(
                                         linestyle='-', linewidth=2.0),
                                     widths=0.85
                                     )

ax.set_xlabel('Model Type', fontsize=13, labelpad=14)
ax.set_ylabel(' OA % vs F1 score (weighted) %  Testdata ', fontsize=13, labelpad=10)
ax.title.set_size(17)
ax.set_title('', fontsize=15)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
# plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12, fontweight='bold', color='black')
ax.set_xticklabels(label_M_L, rotation=90)

plt.show()


##############################


df_metrics_RGS_F1_ACC_sub_RF_train=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['classifier_set'].str.contains('RF Train', na=False) ]

df_metrics_RGS_F1_ACC_sub_RF_train_s=df_metrics_RGS_F1_ACC_sub_RF_train.sort_values(by='metrics_sort')


df_metrics_RGS_F1_ACC_sub_RF_train_s['classifier_set_2']=df_metrics_RGS_F1_ACC_sub_RF_train_s['classifier'].astype(str)+' '+df_metrics_RGS_F1_ACC_sub_RF_train_s['metrics_sort'].astype(str)


print(df_metrics_RGS_F1_ACC_sub_RF_test_s.metrics_sort.unique().tolist())

label_M_L=df_metrics_RGS_F1_ACC_sub_RF_test_s.classifier_set_2.unique().tolist()


ax = df_metrics_RGS_F1_ACC_sub_RF_train_s.boxplot(by='model_dataset_mt', column='metric_scores_%', color=dict(boxes='blue', whiskers='red', medians='red'),
                                     boxprops=dict(linestyle='-', linewidth=2.0,
                                                   color=box_colors['boxes']),
                                     flierprops=dict(
                                         linewidth=2.0, markerfacecolor='red'),
                                     medianprops=dict(
                                         linestyle='-', linewidth=2.0, color='red'),
                                     whiskerprops=dict(
                                         linestyle='-', linewidth=2.0),
                                     capprops=dict(
                                         linestyle='-', linewidth=2.0),
                                     widths=0.85
                                     )

ax.set_xlabel('Model Type', fontsize=13, labelpad=14)
ax.set_ylabel(' OA % vs F1 score (weighted) % Traindata', fontsize=13, labelpad=10)
ax.title.set_size(17)
ax.set_title('', fontsize=15)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
# plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12, fontweight='bold', color='black')
ax.set_xticklabels(label_M_L, rotation=90)

plt.show()



#[]

df_metrics_RGS_F1_ACC_sub_SVM_test=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['classifier_set'].str.contains('SVM Test', na=False) ]
#####################
df_M_SVM_test_ACC=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['classifier_M'].str.contains('SVM ', na=False) ]

df_M_SVM_test_ACC.columns.tolist()

df_M_SVM_test_ACC.metrics_sort.unique()

df_M_SVM_Acc=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['classifier_M'].str.contains('Acc', na=False) ]


######################
df_metrics_RGS_F1_ACC_sub_SVM_test_s=df_metrics_RGS_F1_ACC_sub_SVM_test.sort_values(by='metrics_sort')


df_metrics_RGS_F1_ACC_sub_SVM_test_s['classifier_set_2']=df_metrics_RGS_F1_ACC_sub_SVM_test_s['classifier'].astype(str)+' '+df_metrics_RGS_F1_ACC_sub_SVM_test_s['metrics_sort'].astype(str)


print(df_metrics_RGS_F1_ACC_sub_RF_test_s.metrics_sort.unique().tolist())

label_M_L=df_metrics_RGS_F1_ACC_sub_RF_test_s.classifier_set_2.unique().tolist()
label_M_L_SVM=df_metrics_RGS_F1_ACC_sub_SVM_test_s.classifier_set_2.unique().tolist()

ax = df_metrics_RGS_F1_ACC_sub_SVM_test_s.boxplot(by='model_dataset_mt', column='metric_scores_%', color=dict(boxes='blue', whiskers='red', medians='red'),
                                     boxprops=dict(linestyle='-', linewidth=2.0,
                                                   color=box_colors['boxes']),
                                     flierprops=dict(
                                         linewidth=2.0, markerfacecolor='red'),
                                     medianprops=dict(
                                         linestyle='-', linewidth=2.0, color='red'),
                                     whiskerprops=dict(
                                         linestyle='-', linewidth=2.0),
                                     capprops=dict(
                                         linestyle='-', linewidth=2.0),
                                     widths=0.85
                                     )

ax.set_xlabel('Model Type', fontsize=13, labelpad=14)
ax.set_ylabel(' OA % vs F1 score (weighted) % Testdata', fontsize=13, labelpad=10)
ax.title.set_size(17)
ax.set_title('', fontsize=15)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
# plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12, fontweight='bold', color='black')
ax.set_xticklabels(label_M_L_SVM, rotation=90)

plt.show()




df_metrics_RGS_F1_ACC['classifier_M']=df_metrics_RGS_F1_ACC['classifier_set'].astype(str)+'_'+df_metrics_RGS_F1_ACC['metric'].astype(str)

print(df_metrics_RGS_F1_ACC.classifier_M.unique().tolist())
M_LLL=df_metrics_RGS_F1_ACC.classifier_M.unique().tolist()

ax = df_metrics_RGS_F1_ACC.boxplot(by='classifier_M', column='metric_scores_%', color=dict(boxes='blue', whiskers='red', medians='red'),
                                     boxprops=dict(linestyle='-', linewidth=2.0,
                                                   color=box_colors['boxes']),
                                     flierprops=dict(
                                         linewidth=2.0, markerfacecolor='red'),
                                     medianprops=dict(
                                         linestyle='-', linewidth=2.0, color='red'),
                                     whiskerprops=dict(
                                         linestyle='-', linewidth=2.0),
                                     capprops=dict(
                                         linestyle='-', linewidth=2.0),
                                     widths=0.85
                                     )

ax.set_xlabel('Model Type', fontsize=13, labelpad=14)
ax.set_ylabel(' OA % vs F1 score (weighted) % Model- metrics', fontsize=13, labelpad=10)
ax.title.set_size(17)
ax.set_title('', fontsize=15)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
# plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12, fontweight='bold', color='black')
ax.set_xticklabels(M_LLL, rotation=90)

plt.show()


df_metrics_RGS_F1_ACC_sub_SVM_train=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['classifier_set'].str.contains('SVM Train', na=False) ]


df_metrics_RGS_F1_ACC_sub_SVM_train_s=df_metrics_RGS_F1_ACC_sub_SVM_train.sort_values(by='metrics_sort')


df_metrics_RGS_F1_ACC_sub_SVM_train_s['classifier_set_2']=df_metrics_RGS_F1_ACC_sub_SVM_train_s['classifier'].astype(str)+' '+df_metrics_RGS_F1_ACC_sub_SVM_test_s['metrics_sort'].astype(str)




################
############################################################################
############################################################################


# df_metrics_RGS_F1_ACC_sub_RF_test_s.to_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_RF_test.csv')
# df_metrics_RGS_F1_ACC_sub_RF_test_s.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_RF_test.xlsx')


# df_metrics_RGS_F1_ACC_sub_RF_train_s.to_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_RF_train.csv')
# df_metrics_RGS_F1_ACC_sub_RF_train_s.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_AccuracyRF_train.xlsx')



# df_metrics_RGS_F1_ACC_sub_SVM_test_s.to_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_SVM_test.csv')
# df_metrics_RGS_F1_ACC_sub_SVM_test_s.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_SVM_test.xlsx')


# df_metrics_RGS_F1_ACC_sub_SVM_train_s.to_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_SVM_train_2.csv')
# print(len(df_metrics_RGS_F1_ACC_sub_SVM_train_s))
# df_metrics_RGS_F1_ACC_sub_SVM_train_s.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_SVM_train.xlsx')

###boxplot_fig
##################################################################

####df_metrics_RGS_F1_ACC Name of Masterdataframe: df_metrics_RGS_F1_ACC
df_metrics_RGS_F1_ACC_sub_RF_test=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['classifier_set'].str.contains('RF', na=False) ]
df_metrics_rf_Accuracy=df_metrics_RGS_F1_ACC_sub_RF_test[df_metrics_RGS_F1_ACC_sub_RF_test['metric_type'].str.contains('Accuracy', na=False) ]

df_metrics_SVM=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['classifier_set'].str.contains('SVM', na=False) ]

metrics_merged_x_sort_SVM=df_metrics_SVM.sort_values(by='metrics_sort')


df_metrics_rf_Accuracy=df_metrics_RGS_F1_ACC_sub_RF_test[df_metrics_RGS_F1_ACC_sub_RF_test['metric_type'].str.contains('Accuracy', na=False) ]

df_metrics_RF=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['classifier_set'].str.contains('RF', na=False) ]

metrics_merged_x_sort_RF=df_metrics_SVM.sort_values(by='metrics_sort')




###########################################################################

# #############################################################################

# df_metrics_RF=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['classifier'].str.contains('RF', na=False) ]
# df_metrics_RF_Acc=df_metrics_RF[df_metrics_RF['metric_type'].str.contains('Acc', na=False) ]
# ädf_metrics_RF_Acc=df_metrics_RF=df_metrics_RF[df_metrics_RF['metric_type'].str.contains('Acc', na=False) ]
# ##########################sorted dataframes
# metrics_merged_x_sort_RF=df_metrics_RF.sort_values(by='metrics_sort')
# df_metrics_RF_Acc_test=df_metrics_RF[df_metrics_RF['dataset'].str.contains('Test', na=False) ]
# df_metrics_RF_Acc_train=df_metrics_RF[df_metrics_RF['dataset'].str.contains('Train', na=False) ]
# df_metrics_RF_Acc_train_sorted=df_metrics_RF_Acc_train.sort_values(by='model_dataset_mt')
# #############################
# ################select the dataframe and collect  
# df_metrics_SVM=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['classifier'].str.contains('SVM', na=False) ]
# df_metrics_SVM_Acc=df_metrics_SVM[df_metrics_SVM['metric_type'].str.contains('Acc', na=False) ]
# df_metrics_SVM_Acc_test=df_metrics_SVM_Acc[df_metrics_SVM_A['dataset'].str.contains('Test', na=False) ]
# df_metrics_SVM_Acc_test_sorted=df_metrics_SVM_Acc_test.sort_values(by='model_dataset_mt')

# df_metrics_SVM_Acc_train=df_metrics_SVM_Acc[df_metrics_SVM_Acc['dataset'].str.contains('Train', na=False) ]
# df_metrics_SVM_Acc_sorted=df_metrics_SVM_Acc.sort_values(by='model_dataset_mt')
# df_metrics_RF_Acc=df_metrics_SVM[df_metrics_SVM['metric_type'].str.contains('Acc', na=False) ]


# ##########################sorted dataframes
# metrics_merged_x_sort_RF=df_metrics_RF.sort_values(by='metrics_sort')
# df_metrics_RF_Acc_=df_metrics_RF[df_metrics_RF['metric_type'].str.contains('Test', na=False) ]
# df_metrics_RF_Acc_test=df_metrics_RF_Acc[df_metrics_RF_Acc['metric_type'].str.contains('Test', na=False) ]
# df_metrics_RF_Acc_test=df_metrics_RF[df_metrics_RF['dataset'].str.contains('Test', na=False) ]
# df_metrics_RF_Acc_train=df_metrics_RF[df_metrics_RF['dataset'].str.contains('Train', na=False) ]

df_metrics_RF_Acc_sorted=df_metrics_RF_Acc.sort_values(by='model_dataset_mt')
df_metrics_SVM_Acc_train_sorted=df_metrics_SVM_Acc_train.sort_values(by='model_dataset_mt')
df_metrics_SVM_Acc_test_sorted=df_metrics_SVM_Acc_test.sort_values(by='model_dataset_mt')


#####################################################################




# print(len(df_metrics_SVM_Acc_train.model_dataset_mt.unique()))

# print(len(df_metrics_RF_Acc_sorted.model_dataset_mt.unique()))
# print(len(df_metrics_RF_Acc_train_sorted.model_dataset_mt.unique()))


# dfs_Model=[d for _, d in metrics_merged_x_sort_RF.groupby(['model_dataset_mt'])]
# Wuchs_DG_new_1=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_scores_%'], axis=1)
#     WL=W['metric_scores_%'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_new_1.append(WLA)
#     WK=i.filter(['metric_scores_%'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_new_1:
#     print(data)
    
# dfs_Model=[d for _, d in df_metrics_SVM_Acc_test_sorted.groupby(['model_dataset_mt'])]
# Wuchs_DG_new_2=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_scores_%'], axis=1)
#     WL=W['metric_scores_%'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_new_2.append(WLA)
#     WK=i.filter(['metric_scores_%'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_new_2:
#     print(data)

# #######################

# print(len(df_metrics_SVM_Acc_train_sorted))

# title2='NEW title'


###############set labels for the graphs (title , x- label, y label, tick- labels)
#for the boxplot significant function



title_2=" "
ylabel = r'RMSE %  Train- Testdata Repeated K- Fold CV (n=25) '


ylabel_RF_M1=r'Accuracy Testdata RF- Models Repeated K- Fold CV (n=25) '
ylabel_SVM_M1=r'Accuracy Testdata SVM- Models Repeated K- Fold CV (n=25) '
ylabel_RF_M1_2=r'Accuracy Traindata RF- Models Repeated K- Fold CV (n=25) '
ylabel_SVM_M1_2=r'Accuracy Traindata SVM- Models Repeated K- Fold CV (n=25) '


ylabel_RF_M1_RMSE=r'RMSE Testdata RF- Models Repeated K- Fold CV (n=25) '
ylabel_SVM_M1_RMSE=r'RMSE Testdata SVM- Models Repeated K- Fold CV (n=25) '
ylabel_RF_M1_2_RMSE=r'RMSE Traindata RF- Models Repeated K- Fold CV (n=25) '
ylabel_SVM_M1_2_RMSE=r'RMSE Traindata SVM- Models Repeated K- Fold CV (n=25) '

ylabel_RF_M1_RMSE_perc=r'RMSE % Testdata RF- Models Repeated K- Fold CV (n=25) '
ylabel_SVM_M1_RMSE_perc=r'RMSE % Testdata SVM- Models Repeated K- Fold CV (n=25) '
ylabel_RF_M1_2_RMSE_perc=r'RMSE % Traindata RF- Models Repeated K- Fold CV (n=25) '
ylabel_SVM_M1_2_RMSE_perc=r'RMSE % Traindata SVM- Models Repeated K- Fold CV (n=25) '

Train_Test_comp_Accuracy_SVM_RF=r'OA Test- and Traindata RF-SVM- Models Repeated K- Fold CV (n=25) '
Train_Test_comp_RMSE_SVM_RF=r'RMSE Test- and Traindata RF-SVM Models Repeated K- Fold CV (n=25) '
Train_Test_comp_RMSE_perc_SVM_RF=r'RMSE % Test- and Traindata RF- SVM Models Repeated K- Fold CV (n=25)'

Train_Test_comp_Accuracy_RF=r'OA Test- and Traindata RF-Models Repeated K- Fold CV (n=25) '
Train_Test_comp_RMSE_RF=r'- RMSE Test- and Traindata RF-Models Repeated K- Fold CV (n=25) '
Train_Test_comp_RMSE_perc_RF=r'RMSE % Test- and Traindata RF-Models Repeated K- Fold CV (n=25)'


Train_Test_comp_Accuracy_SVM=r'OA Test- and Traindata SVM-Models Repeated K- Fold CV (n=25) '
Train_Test_comp_RMSE_SVM=r'- RMSE Test- and Traindata SVM-Models Repeated K- Fold CV (n=25) '
Train_Test_comp_RMSE_perc_SVM=r'RMSE % Test- and Traindata SVM-Models Repeated K- Fold CV (n=25)'



title=''
ylabel_Acc_F1='Accuracy % vs f1-weighted % Testdatasets RF-CV (n=25)'

ylabel_Acc_F1_RF_train='Accuracy % vs f1-weighted % Traindatasets RF-CV (n=25)'

ylabel_Acc_F1_SVM_test='Accuracy % vs f1-weighted % Testdatasets RF-CV (n=25)'

ylabel_Acc_F1_SVM_train='Accuracy % vs f1-weighted % Traindatasets RF-CV (n=25)'


x_tick_labels_Acc_F1=['RF 1 Accuracy','RF 1 f1 weighted', 'RF 2 Accuracy','RF 2 f1 weighted', 'RF 3 Accuracy','RF 3 f1 weighted',
                      'RF 4 Accuracy','RF 4 f1 weighted', 'RF 5 Accuracy','RF 5 f1 weighted', 'RF 6 Accuracy','RF 6 f1 weighted',
                      'RF 7 Accuracy','RF 7 f1 weighted']

x_tick_labels_Acc_F1_SVM=['SVM 1 Accuracy','SVM 1 f1 weighted', 'SVM 2 Accuracy','SVM 2 f1 weighted', 'SVM 3 Accuracy','SVM 3 f1 weighted',
                      'SVM 4 Accuracy','SVM 4 f1 weighted', 'SVM 5 Accuracy','SVM 5 f1 weighted', 'SVM 6 Accuracy','SVM 6 f1 weighted',
                      'SVM 7 Accuracy','SVM 7 f1 weighted']

x_tick_labels_Acc_DIFF_Train_Test=['Diff Train- Test RF 1','Diff Train- Test SVM 1','Diff Train- Test RF 2','Diff Train- Test SVM 2', 
                                   'Diff Train- Test RF 3','Diff Train- Test SVM 3', 'Diff Train- Test RF 4','Diff Train- Test SVM 4', 'Diff Train- Test RF 5','Diff Train- Test SVM 5',
                                   'Diff Train- Test RF 6','Diff Train- Test SVM 6', 'Diff Train- Test RF 7','Diff Train- Test SVM 7']

print(len(x_tick_labels_Acc_F1))


#xticklabels = ['First_Group', 'Second Group']
#xticklabels_2 = ['0', '1','3', '5', '7', '9']
title=''
xticklabels_2 = ['SVM 1', 'SVM 2','SVM 3', 'SVM 4','SVM 5','RF 1', 'RF 2', 'RF 3', 'RF 4', 'RF 5']
xticklabels_RF = ['RF 1', 'RF 2','RF 3', 'RF 4','RF 5', 'RF 6', 'RF 7']
xticklabels_SVM = ['SVM 1', 'SVM 2','SVM 3', 'SVM 4','SVM 5', 'SVM 6', 'SVM 7']
xticklabels_2 = ['RF 1', 'RF 2','RF 3', 'RF 4','RF 5', 'RF 6', 'RF 7', 'SVM 1', 'SVM 2','SVM 3', 'SVM 4','SVM 5', 'SVM 6', 'SVM 7']
xticklabels_RF_14 = ['RF 1 Test', 'RF 1 Train','RF 2 Test', 'RF 2 Train', 'RF 3 Test', 'RF 3 Train', 'RF 4 Test', 'RF 4 Train', 'RF 5 Test', 'RF 5 Train', 'RF 6 Test', 'RF 6 Train', 'RF 7 Test', 'RF 7 Train']
xticklabels_SVM_14 = ['SVM 1 Test', 'SVM 1 Train', 'SVM 2 Test', 'SVM 2 Train','SVM 3 Test', 'SVM 3 Train', 'SVM 4 Test', 'SVM 4 Train', 'SVM 5 Test', 'SVM 5 Train', 'SVM 6 Test', 'SVM 6 Train', 'SVM 7 Test', 'SVM 7 Train']



##############################################################################
#############################################
x=box_and_whisker_N(Wuchs_DG_18, title_2, Train_Test_comp_RMSE_perc_RF, xticklabels_RF_14 )
box_and_whisker_N(Wuchs_DG_new_2,title_2, Train_Test_comp_RMSE_perc_RF, xticklabels_SVM)
box_and_whisker(Wuchs_DG_new_2,title_2, Train_Test_comp_RMSE_perc_RF, xticklabels_SVM)

box_and_whisker_N(Wuchs_DG_new_1,title_2, Train_Test_comp_RMSE_perc_RF, xticklabels_SVM_14)
box_and_whisker_N(Wuchs_DG_new_2,title_2, Train_Test_comp_RMSE_perc_RF, xticklabels_RF)


box_and_whisker(Wuchs_DG_new_3,title_2, Train_Test_comp_RMSE_perc_RF, xticklabels_SVM)

box_and_whisker(Wuchs_DG_new_1,title_2, Train_Test_comp_RMSE_perc_RF, xticklabels_SVM_14)


box_and_whisker(Wuchs_DG_new_4,title_2, Train_Test_comp_RMSE_perc_RF, xticklabels_SVM)


###################################
#Metrics_Train_test_split_SVM_sub
X_ticklabel_sub_RF=['RF 1 Test','RF 1 Train','RF 2 Test','RF 2 Train', 'RF 3 Test','RF 3 Train', 'RF 7 Test','RF 7 Train' ]
X_ticklabel_sub_SVM=['SVM 1 Test','SVM 1 Train','SVM 2 Test','SVM 2 Train', 'SVM 3 Test','SVM 3 Train', 'SVM 7 Test','SVM 7 Train' ]
print(len(X_ticklabel_sub_RF))
label_sub_RF_Train_test=r'RMSE % Test- and Traindata selected RF-Models Repeated K- Fold CV (n=25)'
label_sub_SVM_Train_test=r'RMSE % Test- and Traindata selected SVM-Models Repeated K- Fold CV (n=25)'
#x=box_and_whisker_NNN(Wuchs_DG_22, title_2, label_sub_RF_Train_test, xticklabels_RF )




#############################################################





###################################################
#df_metrics_RGS_F1_ACC.to_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_ALL.csv')
# df_metrics_RGS_F1_ACC.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_ALL.xlsx')


# df_metrics_RGS_F1_ACC.to_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_ALL.csv')
# df_metrics_RGS_F1_ACC.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_ALL.xlsx')

########################################
###########################################




### nach anderen Metriken grupppieren
###################################################################
########Visualization with boxplots and significance after Mann- Whitney- U Test




#Excel data sheet with the growth class data you have 
#df_excel=pd.read_excel("C:/Users/Ronald/Documents/Boxplot_signif_bar_script/Volumen_gegen_Wuchs_calc.xlsx")


#####
###Neuste verschnittene Geodatendatei

#"C:/Users/ronal/OneDrive/Dokumente/ARENA_DATA_GEO_NEU/ARENA_GEODATA_NEW_intersect_2.csv"
# df_excel_new=pd.read_excel("C:/Users/ronal/OneDrive/Dokumente/ARENA_DATA_GEO_NEU/ARENA_GEODATA_NEW_intersect_Excel.xlsx")

# df_excel_new_2=pd.read_excel("C:/Users/ronal/OneDrive/Dokumente/Masken_with_Supervised_Classes/OSAVI_Mask_ZS_Supervised_Classes_F_DIff_Classes.xlsx"
# )

######################################################################################
#######################################################################################

# df_OSAVI_Masks=pd.read_excel("D:/MASKEN_ZS_FINAL_/OSAVI_ZS_SC_F_2.xlsx")

# df_OBIA_Masks=pd.read_excel("D:/MASKEN_ZS_FINAL_/OBIA_MASK_ZC_SC_F.xlsx")

# df_OBIA_Masks=pd.read_excel("D:/MASKEN_ZS_FINAL_/OBIA_com_OSAVI_MASK_ZS_F.csv")
# ######################################################################################
# ###################################################################################

# df_OSAVI_Masks=pd.read_excel("D:/MASKEN_ZS_FINAL_/OSAVI_ZS_SC_F_2.xlsx")
# print(df_OSAVI_Masks.columns.tolist())


# df_OSAVI_Masks_S = df_OSAVI_Masks[['_NDVI_4_3m', '_NDVI_5_3m', '_OSAVImean', 'W_03_08_i', 'Befall_NEU_Veraison_22',
#                             'Symptom_St']]

# df_OSAVI_drop_NA=df_OSAVI_Masks_S.dropna()

####################################################################################

#####define function for boxplot visualization of the metrics with significance stars and bars for each group combination

from matplotlib.lines import Line2D
#[]
custom_lines=[Line2D([0], [0],color='coral', lw=4),
              Line2D([0], [0],color='blue', lw=4),
              Line2D([0], [0],color='orange', lw=4),
              Line2D([0], [0],color='green', lw=4),
              Line2D([0], [0],color='black', lw=4),
              Line2D([0], [0],color='yellow', lw=4),
              Line2D([0], [0],color='lime', lw=4)]
              

def box_and_whisker(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=14, weight='bold')
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=2, width=1.5)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    ###add legend somewhere here?
    ##ax.legend(['spectral','structural, 'textural', 'spectral+structural', 'spectral+textural', structural+textural', 'spectral+structural+textural'])
    plt.legend(custom_lines,['sp', 'str', 'tex', 'sp+str', 'sp-tex', 'str-tex', 'sp-str-tex'],loc='upper right',
               bbox_to_anchor=(1.27,1.03))
    #ax.legend(loc='upper right')
    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    colors=['coral', 'blue', 'orange', 'green', 'black', 'yellow', 'lime']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()

###define other slightly modified boxlplot significance functions for other groups and labels


def box_and_whisker_3(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90)
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=0)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    colors=['green', 'blue','green', 'blue', 'green', 'blue', 'green', 'blue','green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue' ]
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    # significant_combinations = []
    # # Check from the outside pairs of boxes inwards
    # ls = list(range(1, len(data) + 1))
    # combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    # for c in combinations:
    #     data1 = data[c[0] - 1]
    #     data2 = data[c[1] - 1]
    #     # Significance
    #     U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    #     if p < 0.05:
    #         significant_combinations.append([c, p])

    # # Get info about y-axis
    # bottom, top = ax.get_ylim()
    # yrange = top - bottom

    # # Significance bars
    # for i, significant_combination in enumerate(significant_combinations):
    #     # Columns corresponding to the datasets of interest
    #     x1 = significant_combination[0][0]
    #     x2 = significant_combination[0][1]
    #     # What level is this bar among the bars above the plot?
    #     level = len(significant_combinations) - i
    #     # Plot the bar
    #     bar_height = (yrange * 0.08 * level) + top
    #     bar_tips = bar_height - (yrange * 0.02)
    #     plt.plot(
    #         [x1, x1, x2, x2],
    #         [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    #     # Significance level
    #     p = significant_combination[1]
    #     if p < 0.001:
    #         sig_symbol = '***'
    #     elif p < 0.01:
    #         sig_symbol = '**'
    #     elif p < 0.05:
    #         sig_symbol = '*'
    #     text_height = bar_height + (yrange * 0.01)
    #     plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # # Adjust y-axis
    # bottom, top = ax.get_ylim()
    # yrange = top - bottom
    # ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    # for i, dataset in enumerate(data):
    #     sample_size = len(dataset)
    #     ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    # plt.show()






def box_and_whisker_NN(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=14, weight='bold')
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=0)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    plt.legend(custom_lines_2,['Testdata', 'Traindata'],loc='upper right',
               bbox_to_anchor=(1.27,1.03))    






    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    colors=['green', 'blue','green', 'blue', 'green', 'blue', 'green', 'blue','green', 'blue', 'green', 'blue', 'green', 'blue' ]
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    #Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    #Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()





custom_lines_3=[Line2D([0], [0],color='black', lw=4),
              Line2D([0], [0],color='red', lw=4)]

colors=['black', 'black','red', 'red', 'black', 'black', 'red', 'red' ]

plt.legend(custom_lines_2,['no structural', 'structural'],loc='upper right',bbox_to_anchor=(1.27,1.03))




def box_and_whisker_NNN(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=14, weight='bold')
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=2, width=1.5)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)
    plt.legend(custom_lines_3,['no structural', 'structural'],loc='upper right',bbox_to_anchor=(1.27,1.03))
    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    #colors=['coral', 'blue', 'orange', 'green', 'black', 'yellow', 'lime']
    colors=['black', 'black','red', 'red', 'black', 'black', 'red', 'red' ]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()






def box_and_whisker_2(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=14, weight='bold')
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=2, width=1.5)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    colors=['coral', 'blue', 'orange', 'green', 'black', 'yellow', 'lime']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()


def box_and_whisker_3(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90)
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=0)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    colors=['green', 'blue','green', 'blue', 'green', 'blue', 'green', 'blue','green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue' ]
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    # significant_combinations = []
    # # Check from the outside pairs of boxes inwards
    # ls = list(range(1, len(data) + 1))
    # combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    # for c in combinations:
    #     data1 = data[c[0] - 1]
    #     data2 = data[c[1] - 1]
    #     # Significance
    #     U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    #     if p < 0.05:
    #         significant_combinations.append([c, p])

    # # Get info about y-axis
    # bottom, top = ax.get_ylim()
    # yrange = top - bottom

    # # Significance bars
    # for i, significant_combination in enumerate(significant_combinations):
    #     # Columns corresponding to the datasets of interest
    #     x1 = significant_combination[0][0]
    #     x2 = significant_combination[0][1]
    #     # What level is this bar among the bars above the plot?
    #     level = len(significant_combinations) - i
    #     # Plot the bar
    #     bar_height = (yrange * 0.08 * level) + top
    #     bar_tips = bar_height - (yrange * 0.02)
    #     plt.plot(
    #         [x1, x1, x2, x2],
    #         [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    #     # Significance level
    #     p = significant_combination[1]
    #     if p < 0.001:
    #         sig_symbol = '***'
    #     elif p < 0.01:
    #         sig_symbol = '**'
    #     elif p < 0.05:
    #         sig_symbol = '*'
    #     text_height = bar_height + (yrange * 0.01)
    #     plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # # Adjust y-axis
    # bottom, top = ax.get_ylim()
    # yrange = top - bottom
    # ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    # for i, dataset in enumerate(data):
    #     sample_size = len(dataset)
    #     ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()
    
    

def box_and_whisker_2(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=14, weight='bold')
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=2, width=1.5)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    colors=['coral', 'blue', 'orange', 'green', 'black', 'yellow', 'lime']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()


#############################################
#############################################################

# ls=list(range(1, 7))
# print(ls)

# combinations_test = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
# print(combinations_test)

# ####
# ##adjust the combinations

# ####

# new_comb_list=[(1,2), (3,4), (5,6), (7,8)]

# new_comb_list_2=[(1,2), (3,4), (5,6), (7,8), (9, 10), (11,12), (13, 14)]
# print(len(new_comb_list_2))

# new_comb_list_3=[(1,2), (3,4), (5,6), (7,8), (9, 10), (11,12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24),(25, 26), (27, 28)]


# for i in reversed(ls):
#     print(i)


    



y_L='Difference Accuracy of Test- and Traindata Models compared'



custom_lines_3=[Line2D([0], [0],color='darkgreen', lw=4),
              Line2D([0], [0],color='blue', lw=4)]

custom_lines_3=[Line2D([0], [0],color='coral', lw=5),
              Line2D([0], [0],color='blue', lw=5),
              Line2D([0], [0],color='orange', lw=5),
              Line2D([0], [0],color='green', lw=5),
              Line2D([0], [0],color='black', lw=5),
              Line2D([0], [0],color='yellow', lw=5),
              Line2D([0], [0],color='lime', lw=5)]

custom_lines_4=[Line2D([0], [0],color='darkgreen', lw=4),
              Line2D([0], [0],color='blue', lw=4)]

colors=['black', 'black','red', 'red', 'black', 'black', 'red', 'red' ]

colors=['coral', 'blue', 'orange', 'green', 'black', 'yellow', 'lime']

plt.legend(custom_lines_2,['no structural', 'structural'],loc='upper right',bbox_to_anchor=(1.27,1.03))





def box_and_whisker_mod_c(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=14, weight='bold')
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=2, width=1.5)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    ###add legend somewhere here?
    # plt.legend(custom_lines_3,['spec','spec', 'str', 'str', 'spec+str', 'str+tex', 'spec+str+tex'],
    #           loc='upper right',bbox_to_anchor=(1.33,1.028))
    
    plt.legend(custom_lines_3,['spec','str', 'tex', 'sp+str', 'sp+tex', 'str+tex', 'spec+str+tex'],
              loc='upper right',bbox_to_anchor=(1.34,1.028))
    # plt.legend(custom_lines_3,['Accuracy', 'f1- weighted'],loc='upper right',
    #            bbox_to_anchor=(1.27,1.03))
    #ax.legend(loc='upper right')
    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    
    colors=['coral','coral', 'blue','blue', 'orange','orange', 'green','green', 'black','black', 'yellow','yellow', 'lime', 'lime']
    #colors=['blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green']
    
    #colors=['darkgreen', 'blue','darkgreen', 'blue', 'darkgreen', 'blue', 'darkgreen', 'blue', 'darkgreen', 'blue', 'darkgreen', 'blue', 'darkgreen', 'blue' ]
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    #combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    new_comb_list_x=[(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13, 14)]
    for c in new_comb_list_x:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = 3 
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()


def box_and_whisker(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90)
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=0)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    colors=['coral', 'blue', 'orange', 'green','yellow','lime', 'black', 'coral', 'blue', 'orange', 'green', 'black','yellow','lime']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()
    
    

def box_and_whisker_2(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=14, weight='bold')
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=2, width=1.5)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    colors=['coral', 'blue', 'orange', 'green', 'black', 'yellow', 'lime']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()


def box_and_whisker_3(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90)
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=0)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    colors=['green', 'blue','green', 'blue', 'green', 'blue', 'green', 'blue','green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue' ]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    # significant_combinations = []
    # # Check from the outside pairs of boxes inwards
    # ls = list(range(1, len(data) + 1))
    # combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    # for c in combinations:
    #     data1 = data[c[0] - 1]
    #     data2 = data[c[1] - 1]
    #     # Significance
    #     U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    #     if p < 0.05:
    #         significant_combinations.append([c, p])

    # # Get info about y-axis
    # bottom, top = ax.get_ylim()
    # yrange = top - bottom

    # # Significance bars
    # for i, significant_combination in enumerate(significant_combinations):
    #     # Columns corresponding to the datasets of interest
    #     x1 = significant_combination[0][0]
    #     x2 = significant_combination[0][1]
    #     # What level is this bar among the bars above the plot?
    #     level = len(significant_combinations) - i
    #     # Plot the bar
    #     bar_height = (yrange * 0.08 * level) + top
    #     bar_tips = bar_height - (yrange * 0.02)
    #     plt.plot(
    #         [x1, x1, x2, x2],
    #         [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    #     # Significance level
    #     p = significant_combination[1]
    #     if p < 0.001:
    #         sig_symbol = '***'
    #     elif p < 0.01:
    #         sig_symbol = '**'
    #     elif p < 0.05:
    #         sig_symbol = '*'
    #     text_height = bar_height + (yrange * 0.01)
    #     plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # # Adjust y-axis
    # bottom, top = ax.get_ylim()
    # yrange = top - bottom
    # ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    # for i, dataset in enumerate(data):
    #     sample_size = len(dataset)
    #     ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()
    
    

def box_and_whisker_2(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=14, weight='bold')
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=2, width=1.5)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    colors=['coral', 'blue', 'orange', 'green', 'black', 'yellow', 'lime']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()



Wuchsklassen_label=['0', '1', '3', '5', '7', '9']


y_label='CHM pixel volume against each Growth Class'

 x=box_and_whisker_V(Wuchs_DG_pVolume, title_2, y_label, Wuchsklassen_label )



custom_lines=[Line2D([0], [0],color='red', lw=4),
              Line2D([0], [0],color='orangered', lw=4),
              Line2D([0], [0],color='orange', lw=4),
              Line2D([0], [0],color='cyan', lw=4),
              Line2D([0], [0],color='green', lw=4),
              Line2D([0], [0],color='darkgreen', lw=4)]
              

def box_and_whisker_V(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    # Graph title
    ax.set_title(title, fontsize=14, weight="bold")
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=14, weight='bold')
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=2, width=1.5)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    ###add legend somewhere here?
    ##ax.legend(['spectral','structural, 'textural', 'spectral+structural', 'spectral+textural', structural+textural', 'spectral+structural+textural'])
    plt.legend(custom_lines,['GC 0', 'GC 1', 'GC 3', 'GC 5', 'GC 7', 'GC 9'],loc='upper right',
               bbox_to_anchor=(1.27,1.03))
    #ax.legend(loc='upper right')
    # Change the colour of the boxes to Seaborn's 'pastel' palette
    # colors = sns.color_palette('pastel')
    ##color for Bonitur
    #colors=["red", "orangered", "orange", "cyan", "lime", "green"]
    ##colors for model metrics of first paper
    colors=['red', 'orangered', 'orange', 'cyan', 'green', 'darkgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small', weight="bold")

    plt.show()






#####

# df_metrics_RGS_n_Accuraccy_s=pd.read_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/RCFV__Accuracy_boxplot_1.csv')
# #df_metrics_RGS_n_Accuraccy_s.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/RCFV_Accuracy_boxplot_1.xlsx')

# Metric_Accuracy_df=pd.read_csv("C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/RCFV__2_Accuracy_boxplot_MEW_2_TRUE.csv")
# Metric_Accuracy_df.columns.tolist()



# ###RMSE dataframe
# df_metrics_RGS_n_RMSE_s=pd.read_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/RCFV__RMSE_boxplot_1.csv')
# Metric_RMSE_df=pd.read_csv("C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/RCFV__2_RMSE_boxplot_NEW_2_TRUE.csv")
# Metric_RMSE_df.columns.tolist()
# #df_metrics_RGS_n_RMSE_s.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/RCFV_RMSE_boxplot_1.xlsx')
# ###RMSE % dataframe
# df_metrics_RGS_n_RMSE_s_perc=pd.read_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/RCFV__RMSE_perc_boxplot_1.csv')
# df_metrics_RGS_n_RMSE_s_perc.columns.tolist()
# Metric_RMSE_perc_df=pd.read_csv("C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/RCFV__2_RMSE_perc_boxplot_NEW_2_TRUE.csv")
# #df_metrics_RGS_n_RMSE_s_perc.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/RCFV_RMSE_perc_boxplot_1.xlsx')

# ##gesamt alle Metriken
# df_metrics_RGS_n_all_metrics_s_perc=pd.read_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/.csv')



# Metric_all_metrics_perc_df=pd.read_csv("C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/RCFV__2_Acc_RMSE_RMSE_%_boxplot_NEW_2_TRUE.csv")

# Metric_all_metrics_perc_df.columns.tolist()

# ###Train- Test compare sheet

# Metrics_Train_test_split=pd.read_csv("C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/c.csv")


# df_F1_ALL=pd.read_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_ALL.csv')

######################################metrics dataframes from classification above
##use these for visualization
df_metrics_RGS_F1_ACC_sub_RF_test_s.to_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_RF_test.csv')
df_metrics_RGS_F1_ACC_sub_RF_test_s.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_RF_test.xlsx')
df_metrics_RGS_F1_ACC_sub_RF_test_s

df_metrics_RGS_F1_ACC_sub_RF_train_s.to_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_RF_train.csv')
df_metrics_RGS_F1_ACC_sub_RF_train_s.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_AccuracyRF_train.xlsx')



df_metrics_RGS_F1_ACC_sub_SVM_test_s.to_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_SVM_test.csv')
df_metrics_RGS_F1_ACC_sub_SVM_test_s.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_SVM_test.xlsx')


df_metrics_RGS_F1_ACC_sub_SVM_train_s.to_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_SVM_train_2.csv')
print(len(df_metrics_RGS_F1_ACC_sub_SVM_train_s))
df_metrics_RGS_F1_ACC_sub_SVM_train_s.to_excel('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_SVM_train.xlsx')
##use these for visualization
######################################################
#######################################################################################

###boxplot_fig


# Metrics_Train_test_split.columns.tolist()
# print(len(Metrics_Train_test_split))



#####################################################################


###diese Datenrahmen auch nochmal filtern?

#######
# df_metrics_RGS_n_Accuraccy_s.columns.tolist()
# #df_metrics_RGS_n_Accuraccy_s.Model Type.unique()
# df_metrics_RGS_n_Accuraccy_s_SVM=df_metrics_RGS_n_Accuraccy_s[df_metrics_RGS_n_Accuraccy_s['Model Type'].str.contains('SVM')]
# df_metrics_RGS_n_Accuraccy_s_RF=df_metrics_RGS_n_Accuraccy_s[df_metrics_RGS_n_Accuraccy_s['Model Type'].str.contains('RF')]


#Metrics_Train_test_split_Test=Metrics_Train_test_split[Metrics_Train_test_split['Train_Test_type'].str.contains('Test')]
# print(len(Metrics_Train_test_split_Test))
# index_L=[]
# for i in range(len(Metrics_Train_test_split_Test)):
#     index_L.append(i)
# print(len(index_L))
# Metrics_Train_test_split_Test['I_L']=index_L

# Metrics_Train_test_split_Train=Metrics_Train_test_split[Metrics_Train_test_split['Train_Test_type'].str.contains('Train')]

# print(len(Metrics_Train_test_split_Train))
# Metrics_Train_test_split_Train['I_L']=index_L
#####horizontal merge


# Metrics_merged=pd.merge(Metrics_Train_test_split_Test,Metrics_Train_test_split_Train, on='I_L', how='outer')

# Metrics_merged.columns.tolist()

# Metrics_merged.rename(columns={'Unnamed: 0_x':'UNN', 'Model Type_x':'Model_Typ_test',
#                                'Accuracy_x': 'Accuracy_test', 'RMSE_x':'RMSE_test',
#                                'Train_Test_type_x':'Train_Test_type_TEST', 'Accuracy_n_x':'Accuracy_Test_%',
#                                'Unnamed: 0_y':'UNNN', 'Model Type_y':'Model_Typ_train', 'Accuracy_y':
#                                    'Accuracy_train', 'RMSE_y': 'RMSE_train', 'RMSE_%_y':'RMSE_train',
#                                    'Train_Test_type_y':'Train_Test_type_TRAIN', 'Accuracy_n_y':'Accuracy_Train_%'}, inplace=True)


# COL=Metrics_merged.columns.tolist()

# Metrics_merged['Acc_Train_Test_diff_%']=abs((Metrics_merged['Accuracy_test']-Metrics_merged['Accuracy_train']/Metrics_merged['Accuracy_test'])*100)

# Metrics_merged['Acc_Train_Test_diff']=abs((Metrics_merged['Accuracy_test']-Metrics_merged['Accuracy_train']))

# Metrics_merged[['classifier_model', 'Train/Test']]=Metrics_merged['Model_Typ_test'].str.split(' ', expand=True)

# print(Metrics_merged.reindex(COL, axis='columns'))






# Metric_Accuracy_df_RF=Metric_Accuracy_df[Metric_Accuracy_df['Model Type'].str.contains('RF')]
# Metric_Accuracy_df_SVM=Metric_Accuracy_df[Metric_Accuracy_df['Model Type'].str.contains('SVM')]

# #################################################
# df_metrics_RGS_n_RMSE_s_SVM=df_metrics_RGS_n_RMSE_s[df_metrics_RGS_n_RMSE_s['Model Type'].str.contains('SVM')]
# df_metrics_RGS_n_RMSE_s_RF=df_metrics_RGS_n_RMSE_s[df_metrics_RGS_n_RMSE_s['Model Type'].str.contains('RF')]

# Metric_RMSE_df_RF=Metric_RMSE_df[Metric_RMSE_df['Model Type'].str.contains('RF')]
# Metric_RMSE_df_SVM=Metric_RMSE_df[Metric_RMSE_df['Model Type'].str.contains('SVM')]

# ##############################################################
# df_metrics_RGS_n_RMSE_s_perc_SVM=df_metrics_RGS_n_RMSE_s_perc[df_metrics_RGS_n_RMSE_s_perc['Model Type'].str.contains('SVM')]
# df_metrics_RGS_n_RMSE_s_perc_RF=df_metrics_RGS_n_RMSE_s_perc[df_metrics_RGS_n_RMSE_s_perc['Model Type'].str.contains('RF')]


# Metric_RMSE_perc_df_RF=Metric_RMSE_perc_df[Metric_RMSE_perc_df['Model Type'].str.contains('RF')]
# Metric_RMSE_perc_df_SVM=Metric_RMSE_perc_df[Metric_RMSE_perc_df['Model Type'].str.contains('SVM')]

# #######################################################################################

# Metrics_Train_test_split_RF=Metrics_Train_test_split[Metrics_Train_test_split['Model Type'].str.contains('RF')]
# print(len(Metrics_Train_test_split_RF))
# Metrics_Train_test_split_SVM=Metrics_Train_test_split[Metrics_Train_test_split['Model Type'].str.contains('SVM_1')]
# print(len(Metrics_Train_test_split_SVM))
# #df_metrics_RGS_n_all_metrics_s_perc
# Metrics_merged_x_sort=Metrics_merged_x.sort_values(by='sort_classifier')



# Metrics_Train_test_split_RF_sub=Metrics_Train_test_split[Metrics_Train_test_split['Model Type'].str.contains('RF_1|RF_2|RF_3|RF_7', na=False) ]
# print(len(Metrics_Train_test_split_RF_sub))
# Metrics_Train_test_split_SVM_sub=Metrics_Train_test_split[Metrics_Train_test_split['Model Type'].str.contains('SVM_1|SVM_2|SVM_3|SVM_7', na=False)]
# print(len(Metrics_Train_test_split_SVM_sub))


# ###########################################################################

# ##############8. boxplot visualization

# ####################



# #"C:/Users/ronal/OneDrive/Dokumente/Masken_with_Supervised_Classes/OSAVI_Mask_ZS_Supervised_Classes_diff_classes.csv"


# df_ALL_MASKS_MERGED=pd.read_csv("C:/Users/Ronald/Documents/ALLE_MASKEN_MERGE_NEU/ALLE_MASKEN_ZUSAMMEN_MERGE_NEU_2.csv")
# df_ALL_MASKS_MERGED['layer'].unique().tolist()
# df_ALL_MASKS_MERGED=df_ALL_MASKS_MERGED[df_ALL_MASKS_MERGED['layer']=='OSAVI_ZS_SC_F_2']
# df_ALL_MASKS_MERGED.columns.tolist()
# df_excel=df_ALL_MASKS_MERGED
# #get names of the columns
# df_C=df_excel.columns.tolist()
# print(df_C)

# #Daten in Liste umwandeln/packen, welche auf Signifikanz getestet werden sollen

# dffilt=df_excel.filter(['Wuchswuchs','_ExGmean' ], axis=1)
# dffilt=df_ALL_MASKS_MERGED.filter(['W_03_08_i','_OSAVImean' ], axis=1)
# dffilt=dffilt.dropna()
# 'W_03_08_i','_OSAVImean'

# col_l=dffilt.columns.tolist()
# print(col_l)

##################################################################################
##für Bonituren
# df_Wuchs_canopy=[d for _, d in dffilt.groupby(['W_03_08_i'])]
# Wuchs_DG=[]
# Wuchsklasse=[]
# for i in df_Wuchs_canopy:
#     W=i.filter(['_OSAVImean'], axis=1)
#     WL=W['_OSAVImean'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG.append(WLA)
#     WK=i.filter(['_OSAVImean'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG:
#     print(data)
    
#  ######################################################################
# df_metrics_RGS_n_Accuraccy_s.columns.tolist()
# df_metrics_RGS_n_RMSE_s
# df_metrics_RGS_n_RMSE_s_perc
# ######

# #######für Accuracy



# dfs_Model=[d for _, d in df_metrics_RGS_n_Accuraccy_s.groupby(['Model Type'])]
# Wuchs_DG=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_values'], axis=1)
#     WL=W['metric_values'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG.append(WLA)
#     WK=i.filter(['metric_values'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG:
#     print(data)


# Metric_Accuracy_df_RF.columns.tolist()

# ############################################
# ############################################

# Metric_Accuracy_df_RF['metric_score_test_n']=Metric_Accuracy_df_RF['metric_score_test']*100

# Metric_Accuracy_df_SVM['metric_score_test_n']=Metric_Accuracy_df_SVM['metric_score_test']*100


# Metric_Accuracy_df_RF['metric_score_train_n']=Metric_Accuracy_df_RF['metric_score_train']*100

# Metric_Accuracy_df_RF

# Metrics_Train_test_split.columns.tolist()

####RF_Tesdaten


# dfs_Model=[d for _, d in Metric_Accuracy_df_RF.groupby(['Model Type'])]
# Wuchs_DG=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_test'], axis=1)
#     WL=W['metric_score_test'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG.append(WLA)
#     WK=i.filter(['metric_score_test'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG:
#     print(data)
    
    
# ################SVM_testdaten
# dfs_Model=[d for _, d in Metric_Accuracy_df_SVM.groupby(['Model Type'])]
# Wuchs_DG_2=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_test'], axis=1)
#     WL=W['metric_score_test'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_2.append(WLA)
#     WK=i.filter(['metric_score_test'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_2:
#     print(data)
    
    

# dfs_Model=[d for _, d in Metric_Accuracy_df_RF.groupby(['Model Type'])]
# Wuchs_DG_3=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_train'], axis=1)
#     WL=W['metric_score_train'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_3.append(WLA)
#     WK=i.filter(['metric_score_train'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_3:
#     print(data)
        
    
# dfs_Model=[d for _, d in Metric_Accuracy_df_SVM.groupby(['Model Type'])]
# Wuchs_DG_4=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_train'], axis=1)
#     WL=W['metric_score_train'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_4.append(WLA)
#     WK=i.filter(['metric_score_train'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_4:
#     print(data)

# ###################################################
# Metric_RMSE_df_RF
# Metric_RMSE_df_SVM

# dfs_Model=[d for _, d in Metric_RMSE_df_RF.groupby(['Model Type'])]
# Wuchs_DG_5=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_test'], axis=1)
#     WL=W['metric_score_test'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_5.append(WLA)
#     WK=i.filter(['metric_score_test'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_3:
#     print(data)
        
    
# dfs_Model=[d for _, d in Metric_RMSE_df_SVM.groupby(['Model Type'])]
# Wuchs_DG_6=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_test'], axis=1)
#     WL=W['metric_score_test'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_6.append(WLA)
#     WK=i.filter(['metric_score_test'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_4:
#     print(data)




# dfs_Model=[d for _, d in Metric_RMSE_df_RF.groupby(['Model Type'])]
# Wuchs_DG_7=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_train'], axis=1)
#     WL=W['metric_score_train'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_7.append(WLA)
#     WK=i.filter(['metric_score_train'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_3:
#     print(data)
        
    
# dfs_Model=[d for _, d in Metric_RMSE_df_SVM.groupby(['Model Type'])]
# Wuchs_DG_8=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_train'], axis=1)
#     WL=W['metric_score_train'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_8.append(WLA)
#     WK=i.filter(['metric_score_train'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_4:
#     print(data)




# dfs_Model=[d for _, d in Metric_RMSE_perc_df_RF.groupby(['Model Type'])]
# Wuchs_DG_9=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_test'], axis=1)
#     WL=W['metric_score_test'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_9.append(WLA)
#     WK=i.filter(['metric_score_test'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_3:
#     print(data)
        
    
# dfs_Model=[d for _, d in Metric_RMSE_perc_df_SVM.groupby(['Model Type'])]
# Wuchs_DG_10=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_test'], axis=1)
#     WL=W['metric_score_test'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_10.append(WLA)
#     WK=i.filter(['metric_score_test'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_4:
#     print(data)



# dfs_Model=[d for _, d in Metric_RMSE_perc_df_RF.groupby(['Model Type'])]
# Wuchs_DG_11=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_train'], axis=1)
#     WL=W['metric_score_train'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_11.append(WLA)
#     WK=i.filter(['metric_score_train'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_3:
#     print(data)
        
    
# dfs_Model=[d for _, d in Metric_RMSE_perc_df_SVM.groupby(['Model Type'])]
# Wuchs_DG_12=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['metric_score_train'], axis=1)
#     WL=W['metric_score_train'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_12.append(WLA)
#     WK=i.filter(['metric_score_train'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_4:
#     print(data)

# ##Train-Test splits
# ####################

# Metrics_Train_test_split
# Metrics_Train_test_split_RF
# Metrics_Train_test_split_SVM

# Metrics_Train_test_split.columns.tolist()


# dfs_Model=[d for _, d in Metrics_Train_test_split.groupby(['Model Type'])]
# Wuchs_DG_13=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['Accuracy'], axis=1)
#     WL=W['Accuracy'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_13.append(WLA)
#     WK=i.filter(['Accuracy'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_13:
#     print(data)

# ##################################################################


# dfs_Model=[d for _, d in Metrics_Train_test_split.groupby(['Model Type'])]
# Wuchs_DG_14=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['RMSE'], axis=1)
#     WL=W['RMSE'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_14.append(WLA)
#     WK=i.filter(['RMSE'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_14:
#     print(data)

# dfs_Model=[d for _, d in Metrics_Train_test_split.groupby(['Model Type'])]
# Wuchs_DG_15=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['RMSE_%'], axis=1)
#     WL=W['RMSE_%'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_15.append(WLA)
#     WK=i.filter(['RMSE_%'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_15:
#     print(data)


# Metrics_Train_test_split_RF
# Metrics_Train_test_split_RF.columns.tolist()


# dfs_Model=[d for _, d in Metrics_Train_test_split_RF.groupby(['Model Type'])]
# Wuchs_DG_16=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['Accuracy'], axis=1)
#     WL=W['Accuracy'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_16.append(WLA)
#     WK=i.filter(['Accuracy'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_16:
#     print(data)

# Metrics_Train_test_split_RF
# Metrics_Train_test_split_RF.columns.tolist()
# dfs_Model=[d for _, d in Metrics_Train_test_split_RF.groupby(['Model Type'])]
# Wuchs_DG_17=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['RMSE'], axis=1)
#     WL=W['RMSE'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_17.append(WLA)
#     WK=i.filter(['RMSE'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_17:
#     print(data)

Metrics_Train_test_split_RF
Metrics_Train_test_split_RF.columns.tolist()
print(len(Metrics_Train_test_split_RF))
dfs_Model=[d for _, d in Metrics_Train_test_split_RF.groupby(['Model Type'])]
Wuchs_DG_18=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['RMSE_%'], axis=1)
    WL=W['RMSE_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_18.append(WLA)
    WK=i.filter(['RMSE_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_18:
    print(data)
# ###########################################################################

# Metrics_Train_test_split_SVM
# print(len(Metrics_Train_test_split_SVM))
# Metrics_Train_test_split_SVM.columns.tolist()
# dfs_Model=[d for _, d in Metrics_Train_test_split_SVM.groupby(['Model Type'])]
# Wuchs_DG_19=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['Accuracy'], axis=1)
#     WL=W['Accuracy'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_19.append(WLA)
#     WK=i.filter(['Accuracy'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_19:
#     print(data)

# ############################################################################################

# Metrics_Train_test_split_SVM
# print(len(Metrics_Train_test_split_SVM))
# Metrics_Train_test_split_SVM.columns.tolist()
# dfs_Model=[d for _, d in Metrics_Train_test_split_SVM.groupby(['Model Type'])]
# Wuchs_DG_20=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['RMSE'], axis=1)
#     WL=W['RMSE'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_20.append(WLA)
#     WK=i.filter(['RMSE'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_20:
#     print(data)

# ##########################################



# Metrics_Train_test_split_SVM
# Metrics_Train_test_split_SVM.columns.tolist()
# dfs_Model=[d for _, d in Metrics_Train_test_split_SVM.groupby(['Model Type'])]
# Wuchs_DG_21=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['RMSE_%'], axis=1)
#     WL=W['RMSE_%'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_21.append(WLA)
#     WK=i.filter(['RMSE_%'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_21:
#     print(data)


# Metrics_Train_test_split_RF_sub
# Metrics_Train_test_split_RF_sub.columns.tolist()

# dfs_Model=[d for _, d in Metrics_Train_test_split_RF_sub.groupby(['Model Type'])]
# Wuchs_DG_22=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['RMSE_%'], axis=1)
#     WL=W['RMSE_%'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_22.append(WLA)
#     WK=i.filter(['RMSE_%'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_22:
#     print(data)
    
    
# dfs_Model=[d for _, d in Metrics_Train_test_split_SVM_sub.groupby(['Model Type'])]
# Wuchs_DG_23=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['RMSE_%'], axis=1)
#     WL=W['RMSE_%'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_23.append(WLA)
#     WK=i.filter(['RMSE_%'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_23:
#     print(data)


# dfs_Model=[d for _, d in Metrics_Train_test_split_RF_sub.groupby(['Model Type'])]
# Wuchs_DG_24=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['Accuracy'], axis=1)
#     WL=W['Accuracy'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_24.append(WLA)
#     WK=i.filter(['Accuracy'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_24:
#     print(data)
    
    
# dfs_Model=[d for _, d in Metrics_Train_test_split_SVM_sub.groupby(['Model Type'])]
# Wuchs_DG_25=[]
# Wuchsklasse=[]
# for i in dfs_Model:
#     W=i.filter(['Accuracy'], axis=1)
#     WL=W['Accuracy'].tolist()
#     WLA=np.asarray(WL)
#     Wuchs_DG_25.append(WLA)
#     WK=i.filter(['Accuracy'], axis=1)
#     Wuchsklasse.append(WK)
# for data in Wuchs_DG_25:
#     print(data)
    
    

    
# print(len(Wuchs_DG_25))
# lss=list(range(1,len(Wuchs_DG_25)))
# print(lss)

# combinations_new_1 = [(lss[x], lss[x + y]) for y in reversed(lss) for x in range((len(lss) - y))]

    
# box_and_whisker_mod_c


#############################################################
##groubydataframe 


# dF_ACC_F_1_RF_test=pd.read_csv("C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_RF_test.csv")


# dF_ACC_F_1_RF_train=pd.read_csv("C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_RF_train.csv")

# dF_ACC_F_1_SVM_test=pd.read_csv("C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_SVM_test.csv")


# dF_ACC_F_1_SVM_train=pd.read_csv('C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_SVM_train_2.csv')
# print(len(dF_ACC_F_1_SVM_train))
# #"C:\Users\Ronald\Documents\Repeated_Cross_Fold_validation_plots_1_5_\F_1_Accuracy_SVM_train.csv"
# ####All 

# dF_ACC_F_1_ALL=pd.read_csv("C:/Users/Ronald/Documents/Repeated_Cross_Fold_validation_plots_1_5_/F_1_Accuracy_ALL.csv")


# dF_ACC_F_1_RF_train
# dF_ACC_F_1_RF_test   
# dF_ACC_F_1_SVM_test   
# dF_ACC_F_1_RF_train   
# dF_ACC_F_1_ALL    

# dF_ACC_F_1_RF_train.columns.tolist()

# dF_ACC_F_1_RF_train.classifier_set_2.unique().tolist()

# dF_ACC_F_1_RF_test




# Metrics_merged.reindex(index_L)
# Metrics_merged.columns.tolist()
# Metrics_merged_reset_I=Metrics_merged.reset_index(drop=True)


# Metrics_merged.to_csv('C:/Users/Ronald/Documents/grouped_batch_output_NEW_FFF.csv')
# Metrics_merged_x=pd.read_csv('C:/Users/Ronald/Documents/grouped_batch_output_NEW_FFF.csv')

# Metrics_merged_x[['classifier','number']]=Metrics_merged_x['classifier_model'].str.split('_', expand=True)

# Metrics_merged_x['sort_classifier']=Metrics_merged_x['number'].astype(str)+' '+Metrics_merged_x['classifier'].astype(str)

# Metrics_merged_x_sort=Metrics_merged_x.sort_values(by='sort_classifier')



#####################################
# #####################
# Metric_all_metrics_perc_df

# Metric_all_metrics_perc_df.columns.tolist()

# Metric_all_metrics_perc_df.groupby(['Model Type', 'metric_type'])['metric_score_test'].mean()
# Metric_all_metrics_perc_df.groupby(['Model Type', 'metric_type'])['metric_score_train'].mean()

# ##group
# df_group_out_test=Metric_all_metrics_perc_df.groupby(['Model Type', 'metric_type'])['metric_score_train'].agg(['mean','median', 'std','var']).unstack(level='metric_type')

# df_group_out_train=Metric_all_metrics_perc_df.groupby(['Model Type', 'metric_type'])['metric_score_test'].agg(['mean','median', 'std','var']).unstack(level='metric_type')

# print(len(df_group_out_test))

###############################################################


# df_F1_ALL_sub_f1=df_F1_ALL[df_F1_ALL['metric_type']=='f1_weighted']

# df_F1_ALL_group_by=df_F1_ALL.groupby(['Model type', 'metric_type'])['metric_scores'].agg(['mean','median', 'std','var']).unstack(level='metric_type')




# df_F1_ALL_group_by_sub_F1=df_F1_ALL_sub_f1.groupby(['Model Type', 'metric_type'])['metric_scores'].agg(['mean','median', 'std','var']).unstack(level='metric_type')


# df_F1_ALL_group_by.to_csv('C:/Users/Ronald/Documents/metrics_grouped_metrics_NEW/Metrics_NEW_1_grouped.csv')

# df_F1_ALL_group_by.to_excel('C:/Users/Ronald/Documents/metrics_grouped_metrics_NEW/Metrics_NEW_1_grouped.xlsx')

# ###########################################################################################

# df_group_out=Metric_all_metrics_perc_df.groupby(['Model Type', 'metric_type'])['metric_score_train'].agg(['mean','median', 'count','sum', 'std','var']).unstack(level='metric_type')



# df_group_out_ALL=Metric_all_metrics_perc_df.groupby(['Model Type', 'metric_type'])['metric_score_train','metric_score_test' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')


# index_group_new=['1', '2', '3', 'RF_1', 'RF_2', 'RF_3', 'RF_4', 'RF_5', 'RF_6', 'RF_7',
#                  'SVM_1', 'SVM_2', 'SVM_3','SVM_4', 'SVM_5', 'SVM_6', 'SVM_7']

# df_group_out_ALL.to_excel('C:/Users/Ronald/Documents/metrics_grouped_metrics_NEW/other_mretrics_from_cross_validate.xlsx')
# df_group_out_ALL.to_csv('C:/Users/Ronald/Documents/metrics_grouped_metrics_NEW/other_mretrics_from_cross_validate.csv')

# ####################################################


# df_grouped_j_metrics_cross_val_1=pd.merge(df_group_out_ALL,df_F1_ALL_group_by, left_index=True,
#          right_index=True)

# df_grouped_j_metrics_cross_val_2=df_group_out_ALL.join(df_F1_ALL_group_by)

# df_grouped_j_metrics_cross_val_3=pd.concat([df_group_out_ALL,df_F1_ALL_group_by ], axis=1)


# df_group_out_ALL
# df_F1_ALL_group_by


# Metric_all_metrics_perc_df.columns.tolist()

# df_group_out_ALL.unique



# df_group_out_ALL
# print(len(df_group_out_ALL))
# #######################################################################
# ##############################################################


# dF_ACC_F_1_ALL.columns.tolist()

# dF_ACC_F_1_ALL.rename(columns={'Model type':'Model_type'}, inplace=True)
# #####
# dF_ACC_F_1_ALL.Model_type.unique()

# dF_ACC_F_1_ALL.metric_type.unique()

# dF_ACC_F_1_ALL.metric.unique()

# dF_ACC_F_1_ALL.dataset.unique()

# dF_ACC_F_1_ALL.model_dataset.unique()

# dF_ACC_F_1_ALL.model_dataset_mt.unique()

# dF_ACC_F_1_ALL.classifier.unique()

# dF_ACC_F_1_ALL.model_dataset.unique()

# dF_ACC_F_1_ALL.numer.unique()

# dF_ACC_F_1_ALL.set.unique()

# dF_ACC_F_1_ALL.classfier_set.unique()

# dF_ACC_F_1_ALL.metrics_sort.unique()

# dF_ACC_F_1_ALL.classifier_M.unique()



# dF_ACC_F_1_ALL.columns.tolist()

# col_names_L=dF_ACC_F_1_ALL.select_dtypes(include=['category', 'object']).columns.tolist()

# print(dF_ACC_F_1_ALL.select_dtypes(include=['number']).columns.tolist())


# df_group_out_ALL=Metric_all_metrics_perc_df.groupby(['Model Type', 'metric_type'])['metric_score_train','metric_score_test' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')


# df_group_out_ALL.set_axis(index_group_new)

# df_group_out_ALL_cols_grouped=Metric_all_metrics_perc_df.groupby(['Model Type', 'metric_type'])['metric_score_train','metric_score_test' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')
# df_group_out_ALL.round(2)

# df_group_out_ALL.to_csv('C:/Users/Ronald/Documents/grouped_NEW_metrics_Rep_FOLD_CV/grouped_metrics_Repeated_Fold_CV_round.csv')

# df_group_out_ALL.to_excel('C:/Users/Ronald/Documents/grouped_NEW_metrics_Rep_FOLD_CV/grouped_metrics_Repeated_Fold_CVround.xlsx')


# df_group_out_ALL.to_excel('C:/Users/Ronald/Documents/grouped_NEW_metrics_Rep_FOLD_CV/grouped_metrics_Repeated_Fold_CVround.xlsx')


# df_metrics_RGS_F1_ACC_sub_RF_test




# ################################################################
# df_metrics_RGS_F1_ACC_sub_SVM_train_s.columns.tolist()
# ###############################
# #[]
# df_group_out_f1_train_test_SVM_sub=df_metrics_RGS_F1_ACC_sub_SVM_train_s[df_metrics_RGS_F1_ACC_sub_SVM_train_s['metric']=='f1-weighted']
# df_metrics_RGS_F1_ACC_sub_SVM_train_s.columns.tolist()

# df_group_out_f1_train_test_SVM=df_metrics_RGS_F1_ACC_sub_SVM_train_s.groupby(['Model type', 'metric_type'])['metric_scores' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')

# df_group_out_f1_train_test_SVM_sub=df_group_out_f1_train_test_SVM_sub.groupby(['Model Type', 'metric_type'])['metric_score_train','metric_score_test' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')

# ##################

# df_group_out_f1_train_test_RF_sub=df_metrics_RGS_F1_ACC_sub_RF_train_s[df_metrics_RGS_F1_ACC_sub_SVM_train_s['metric']=='f1-weighted']
# df_metrics_RGS_F1_ACC_sub_RF_train_s.columns.tolist()
# df_group_out_f1_train_test_RF=df_metrics_RGS_F1_ACC_sub_RF_train_s.groupby(['Model type', 'metric_type'])['metric_scores' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')

# df_group_out_f1_train_test_RF_sub=df_group_out_f1_train_test_RF_sub.groupby(['Model Type', 'metric_type'])['metric_score_train','metric_score_test' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')

# ####Master dataframe 

# df_metrics_RGS_F1_Acc
# df_Metrics_filtered_f1_weighted_=df_metrics_RGS_F1_Accdf_metrics_RGS_F1_Acc['metric']=='f1-weighted']

# df_group_out_f1_grouped_train_test=df_Metrics_filtered_f1_weighted_.groupby(['model_dataset_mt', 'metric_type'])['metric_scores_%' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')


# ####################################################################################


# df_metrics_RGS_F1_ACC_sub_SVM_test_s
# df_metrics_RGS_F1_ACC_sub_SVM_test_s.columns.tolist()
# df_metrics_RGS_F1_ACC_sub_SVM_train_s
# df_metrics_RGS_F1_ACC_sub_SVM_train_s.columns.tolist()

# df_metrics_RGS_F1_ACC_sub_RF_test_s
# df_metrics_RGS_F1_ACC_sub_RF_test_s.columns.tolist()
# df_metrics_RGS_F1_ACC_sub_RF_train_s
# df_metrics_RGS_F1_ACC_sub_RF_train_s.columns.tolist()

# #######################################################################################

# df_group_out_f1_train_test_RF_sub=df_metrics_RGS_F1_ACC_sub_RF_train_s[df_group_out_f1_train_test_RF['metric']='f1-weighted']


# df_group_out_f1_train_test_RF=df_metrics_RGS_F1_ACC_sub_RF_train_s.groupby(['Model Type', 'metric_type'])['metric_score_train','metric_score_test' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')

# df_group_out_f1_train_test_RF_sub=df_group_out_f1_train_test_RF_sub.groupby(['Model Type', 'metric_type'])['metric_score_train','metric_score_test' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')


# df_group_out_f1_train_test_RF_sub=df_metrics_RGS_F1_ACC_sub_RF_test_s[df_group_out_f1_train_test_RF['metric']='f1-weighted']


# df_group_out_f1_train_test_RF=df_metrics_RGS_F1_ACC_sub_RF_test_s.groupby(['Model Type', 'metric_type'])['metric_score_train','metric_score_test' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')

# df_group_out_f1_train_test_RF_sub=df_group_out_f1_train_test_RF_sub.groupby(['Model Type', 'metric_type'])['metric_score_train','metric_score_test' ].agg(['mean','median', 'std','var']).unstack(level='metric_type')





# concat_f1_frames=pd.concat([df_group_out_f1_train_test_RF,df_group_out_f1_train_test_SVM ], axis=1)


# ###################


# df_metrics_RGS_F1_ACC_sub_SVM_train_s


# df_group_out_ALL_cols_grouped.set_axis(index_group_new)

# df_group_out_ALL_cols_grouped_I=df_group_out_ALL_cols_grouped.index=inde

# for i in col_names_L:
#     dF_ACC_F_1_ALL_cols_grouped=dF_ACC_F_1_ALL.groupby([i])['metric_scores','metric_scores_%' ].agg(['mean','median', 'std','var']).unstack(level=i)
#     #dF_ACC_F_1_ALL_cols_grouped.index=index_group_new
#     dF_ACC_F_1_ALL_cols_grouped.to_csv('C:/Users/Ronald/Documents/grouped_batch_output_NEW'+str(i)+'_grouped.csv')
#     dF_ACC_F_1_ALL_cols_grouped.to_excel('C:/Users/Ronald/Documents/grouped_NEW_metrics_Rep_FOLD_CV/grouped_metrics_Repeated_Fold_CV.xlsx')



# #########################################################################
# ########################################################################

# ####box box

# box_and_whisker_mod_c(Wuchs_DG_16, title_2, Train_Test_comp_Accuracy_RF, xticklabels_RF_14 )


# box_and_whisker_mod_c(Wuchs_DG_19, title_2, Train_Test_comp_Accuracy_SVM, xticklabels_SVM_14)

##############################################################################

##################################################################################

##important dataframes for visualization

# dF_ACC_F_1_RF_train
# dF_ACC_F_1_RF_test   
# dF_ACC_F_1_SVM_test   
# dF_ACC_F_1_RF_train   
# dF_ACC_F_1_ALL 
# df_metrics_RGS_F1_ACC


###Datenrahmen zum Vergleich
###########################################

df_metrics_RGS_F1_ACC_sub_RF_test_s.columns.tolist()

df_metrics_RGS_F1_ACC_sub_RF_test_s_ACC=df_metrics_RGS_F1_ACC_sub_RF_test_s[df_metrics_RGS_F1_ACC_sub_RF_test_s['metric']=='Accuracy']

df_metrics_RGS_F1_ACC_sub_RF_test_s_ACC=df_metrics_RGS_F1_ACC_sub_RF_test_s[df_metrics_RGS_F1_ACC_sub_RF_test_s['metric']=='Accuracy']


df_metrics_RGS_F1_ACC_train_test_=df_metrics_RGS_F1_ACC[df_metrics_RGS_F1_ACC['metric']=='Accuracy']

df_metrics_RGS_F1_ACC_train_test_RF=df_metrics_RGS_F1_ACC_train_test_[df_metrics_RGS_F1_ACC_train_test_['model_dataset'].str.contains('RF')]
df_metrics_RGS_F1_ACC_train_test_SVM=df_metrics_RGS_F1_ACC_train_test_[df_metrics_RGS_F1_ACC_train_test_['model_dataset'].str.contains('SVM')]


# print(len(Metrics_Train_test_split_RF))
# Metrics_Train_test_split_SVM=Metrics_Train_test_split[Metrics_Train_test_split['Model Type'].str.contains('SVM_1')]
# print(len(Metrics_Train_test_split_SVM))
# #df_metrics_RGS_n_all_metrics_s_perc
# Metrics_merged_x_sort=Metrics_merged_x.sort_values(by='sort_classifier')

######################################################
######################################################


dF_ACC_F_1_RF_train
df_metrics_RGS_F1_ACC_sub_RF_train_s_ACC=dF_ACC_F_1_RF_train[dF_ACC_F_1_RF_train['metric']=='Accuracy']


#############################################################################
##group the dataframe and save results in lists for each individual group

dfs_Model=[d for _, d in df_metrics_RGS_F1_ACC_sub_RF_test_s_ACC.groupby(['classifier_set_2'])]
Wuchs_DG_26_xx=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_26_xx.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_26_xx:
    print(data)

len(Wuchs_DG_26_xx)




dfs_Model=[d for _, d in df_metrics_RGS_F1_ACC_sub_RF_train_s_ACC.groupby(['classifier_set_2'])]
Wuchs_DG_27_xx=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_27_xx.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_26_xx:
    print(data)

len(Wuchs_DG_27_xx)


box_and_whisker(Wuchs_DG_27_xx,title,ylabel_Acc_F1, xticklabels_RF)


dfs_Model=[d for _, d in df_metrics_RGS_F1_ACC_sub_RF_train_s_ACC.groupby(['classifier_set_2'])]
Wuchs_DG_27_xx=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_27_xx.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_26_xx:
    print(data)

len(Wuchs_DG_27_xx)


df_metrics_RGS_F1_ACC_train_test_SVM





######################################################################


dfs_Model=[d for _, d in df_metrics_RGS_F1_ACC_train_test_RF.groupby(['model_dataset'])]
Wuchs_DG_27_xxx=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_27_xxx.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_27_xxx:
    print(data)
    


dfs_Model=[d for _, d in df_metrics_RGS_F1_ACC_train_test_SVM.groupby(['model_dataset'])]
Wuchs_DG_27_SVM_x=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_27_SVM_x.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)




df_metrics_RGS_F1_ACC_train_test_SVM

#df_metrics_RGS_F1_ACC_train_test_RF

#Metrics_merged_x_sort=Metrics_merged_x.sort_values(by='sort_classifier')

######################################################################
df_metrics_RGS_F1_ACC_sub_RF_test_s.metric.unique()

df_metrics_RGS_F1_ACC_sub_RF_train_s


df_metrics_RGS_F1_ACC_sub_SVM_test_s

df_metrics_RGS_F1_ACC_sub_SVM_train_s

df_metrics_RGS_F1_ACC_sub_SVM_train_s


#############################################################################

dfs_Model=[d for _, d in dF_ACC_F_1_RF_test.groupby(['classifier_set_2'])]
Wuchs_DG_26_xx=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_26_x.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_26_x:
    print(data)

len(Wuchs_DG_26_x)





dfs_Model=[d for _, d in dF_ACC_F_1_RF_test.groupby(['classifier_set_2'])]
Wuchs_DG_26_x=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_26_x.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_26_x:
    print(data)

len(Wuchs_DG_26_x)


dfs_Model=[d for _, d in dF_ACC_F_1_RF_train.groupby(['classifier_set_2'])]
Wuchs_DG_27_x=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_27_x.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_27_x:
    print(data)

len(Wuchs_DG_27_x)


dfs_Model=[d for _, d in dF_ACC_F_1_SVM_test.groupby(['classifier_set_2'])]
Wuchs_DG_28_x=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_28_x.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_28_x:
    print(data)

len(Wuchs_DG_28_x)
dF_ACC_F_1_SVM_train.columns.tolist()


dF_ACC_F_1_SVM_train['classifier_set_2_2']=dF_ACC_F_1_SVM_train['classifier'].astype(str)+' '+dF_ACC_F_1_SVM_train['number'].astype(str)+' '+dF_ACC_F_1_SVM_train['metric'].astype(str)

dfs_Model=[d for _, d in dF_ACC_F_1_SVM_train.groupby(['classifier_set_2_2'])]
Wuchs_DG_29_x=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_29_x.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_29_x:
    print(data)

len(Wuchs_DG_29_x)
dF_ACC_F_1_SVM_train.columns.tolist()

dfs_Model=[d for _, d in Metrics_merged_x_sort.groupby(['classifier_model'])]
Wuchs_DG_30_x=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['Acc_Train_Test_diff'], axis=1)
    WL=W['Acc_Train_Test_diff'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_30_x.append(WLA)
    WK=i.filter(['Acc_Train_Test_diff'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_30_x:
    print(data)

len(Wuchs_DG_30_x)
dF_ACC_F_1_SVM_train.columns.tolist()
#############################################################


dfs_Model=[d for _, d in df_metrics_SVM_Acc_train_sorted.groupby(['model_dataset_mt'])]
Wuchs_DG_new_3=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_new_3.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_new_3:
    print(data)

#!!!
title2='NEW title'

dfs_Model=[d for _, d in df_metrics_RF_Acc_train.groupby(['model_dataset_mt'])]
Wuchs_DG_new_4=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['metric_scores_%'], axis=1)
    WL=W['metric_scores_%'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_new_4.append(WLA)
    WK=i.filter(['metric_scores_%'], axis=1)
    Wuchsklasse.append(WK)
for data in Wuchs_DG_new_4:
    print(data)


##put the grouped list outputs for the metric group outputs in the boxplot function for visualization

box_and_whisker(Wuchs_DG_26_xx,title,ylabel_Acc_F1, xticklabels_RF)


box_and_whisker(Wuchs_DG_27_xxx,title,ylabel_Acc_F1, xticklabels_RF_14)

box_and_whisker_NN(Wuchs_DG_27_xxx,title,ylabel_Acc_F1, xticklabels_RF_14)

box_and_whisker(Wuchs_DG_27_xxx,title,ylabel_Acc_F1, xticklabels_SVM_14)


box_and_whisker_3(Wuchs_DG_27_SVM_x,title,ylabel_Acc_F1, xticklabels_SVM_14)


box_and_whisker_mod_c(Wuchs_DG_26_x,title,ylabel_Acc_F1, x_tick_labels_Acc_F1)

box_and_whisker_mod_c(Wuchs_DG_27_x,title,ylabel_Acc_F1_RF_train, x_tick_labels_Acc_F1)

box_and_whisker_mod_c(Wuchs_DG_28_x,title,ylabel_Acc_F1_SVM_test, x_tick_labels_Acc_F1_SVM)

box_and_whisker_mod_c(Wuchs_DG_29_x,title,ylabel_Acc_F1_SVM_train, x_tick_labels_Acc_F1_SVM)

#box_and_whisker_mod_c(Wuchs_DG_30_x,title,y_L, x_tick_labels_Acc_DIFF_Train_Test)

box_and_whisker(Wuchs_DG_new_4,title,ylabel_Acc_F1, xticklabels_RF)

box_and_whisker(Wuchs_DG_new_3,title,ylabel_Acc_F1, xticklabels_RF)

print(len(Wuchs_DG_30_x))



dfs_Model=[d for _, d in df_OBIA_comb_OSAVI_FINAL_texture_sub_3.groupby(['W_03_08_i'])]
Wuchs_DG_pVolume=[]
Wuchsklasse=[]
for i in dfs_Model:
    W=i.filter(['V_OM'], axis=1)
    WL=W['V_OM'].tolist()
    WLA=np.asarray(WL)
    Wuchs_DG_pVolume.append(WLA)
    WK=i.filter(['V_OM'], axis=1)
    Wuchsklasse.append(WK)























