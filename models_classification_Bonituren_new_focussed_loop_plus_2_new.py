# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 10:17:33 2024

@author: ronal
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 10:05:56 2024

@author: ronal
"""
#import relevant python packages and associated functions
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
from sklearn.metrics import mean_absolute_error, mean_squared_error,matthews_corrcoef


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
from sklearn.model_selection import learning_curve
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import validation_curve

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  roc_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib
import sys
import os
import glob
import sklearn as sk
import skimage as ski
#check version of sklearn
print(sk.__version__)
print(ski.__version__)
#################################################



# for file in glob.glob()
#here use file OBIA_OSAVI_mit_Texture_NEW.csv as input path to create the input dataframe, also provided in the github repository/ exchange the path 
#-the file contains the spectral, structural+texture features after the geoprocessing routines
#this is the foundation of the column selection, preprocessing and classification process

df_OBIA_comb_OSAVI_FINAL_texture=pd.read_csv("C:/Users/ronal/OneDrive/Dokumente/OBIA_OSAVI_mit_Texture_NEW/OBIA_OSAVI_mit_Texture_NEW.csv")

df_OBIA_comb_OSAVI_FINAL_texture.columns.tolist()
#define standard scaler for pre-processing
scaler=StandardScaler()
df_OBIA_comb_OSAVI_FINAL_texture_all_columns=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean','_CHM_max','_CHM_stdev','V_OM','_Contrastmean','_Correlationmean', '_Entropymean','_ASMmean', 'W_03_08_i']]
df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean','_CHM_max','_CHM_stdev','V_OM','_Contrastmean','_Correlationmean', '_Entropymean','_ASMmean']]==scaler.fit_transform(df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean','_CHM_max','_CHM_stdev','V_OM','_Contrastmean','_Correlationmean', '_Entropymean','_ASMmean']])

df_OBIA_comb_OSAVI_FINAL_texture_all_columns.to_excel("C:/Users/ronal/OneDrive/Dokumente/final_dataframe_all_cols/df_all_columns_features_Wuchs.xlsx")


#Hier dataframes with spectral-, structural and texture feature combination
##spectral
df_OBIA_comb_OSAVI_FINAL_texture.shape[0]
df_OBIA_comb_OSAVI_FINAL_texture_sub_1=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean', 'W_03_08_i']].dropna()
#df_OBIA_comb_OSAVI_FINAL_texture_sub_1.shape[0]
df_OBIA_comb_OSAVI_FINAL_texture_sub_1.columns.tolist()

Y_1=df_OBIA_comb_OSAVI_FINAL_texture_sub_1[['W_03_08_i']].values
Y_1.shape[0]
#spectral+ structural
X_1=df_OBIA_comb_OSAVI_FINAL_texture_sub_1.drop('W_03_08_i', axis=1).values
# feats_1=df_OBIA_comb_OSAVI_FINAL_texture_sub_1.drop('W_03_08_i', axis=1).columns.tolist()
# print(feats_1)
# X_1.shape[0]
# X_1.shape[1]
print(X_1)
feature_type=['spectral', 'spectral+ structural','spectral+structural+textural', 'structural', 'textural', 'spectral+textural', 'structural+textural']
####################################################################

df_OBIA_comb_OSAVI_FINAL_texture_sub_2=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean','_CHM_max','_CHM_stdev','_CHM_range', '_CHM_count', 'V_OM', 'W_03_08_i']].dropna()
feats_2=df_OBIA_comb_OSAVI_FINAL_texture_sub_2.drop('W_03_08_i', axis=1).columns.tolist()
#print(feats_2)
Y_2=df_OBIA_comb_OSAVI_FINAL_texture_sub_2[['W_03_08_i']].values
Y_2.shape[0]
####################################################################
X_2=df_OBIA_comb_OSAVI_FINAL_texture_sub_2.drop('W_03_08_i', axis=1).values
X_2.shape[0]
##only structural Y_4 und X_4
##############################################################################
df_OBIA_comb_OSAVI_FINAL_texture_sub_4=df_OBIA_comb_OSAVI_FINAL_texture[['_CHM_mean','_CHM_max','_CHM_stdev','_CHM_range', '_CHM_count', 'V_OM', 'W_03_08_i']].dropna()
df_OBIA_comb_OSAVI_FINAL_texture_sub_4.columns.tolist()
#feats_4=df_OBIA_comb_OSAVI_FINAL_texture_sub_4.drop('W_03_08_i', axis=1).columns.tolist()
#print(feats_2)
Y_4=df_OBIA_comb_OSAVI_FINAL_texture_sub_4[['W_03_08_i']].values
Y_4.shape[0]
####################################################################
X_4=df_OBIA_comb_OSAVI_FINAL_texture_sub_4.drop('W_03_08_i', axis=1).values
X_4.shape[0]
#spectral+ structural

#########################################################################
############################################################################

###spectral+ strcutural+ textural

df_OBIA_comb_OSAVI_FINAL_texture_sub_3=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean','_CHM_max','_CHM_stdev','V_OM','_Contrastmean','_Correlationmean', '_Entropymean','_ASMmean', 'W_03_08_i']].dropna()

Y_3=df_OBIA_comb_OSAVI_FINAL_texture_sub_3[['W_03_08_i']].values
Y_3.shape[0]



#df_OBIA_comb_OSAVI_FINAL_texture_sub_3=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean','_CHM_max','_CHM_stdev','V_OM','_Contrastmean','_Correlationmean', '_Entropymean','_ASMmean', 'W_03_08_i']].dropna()

Y_3=df_OBIA_comb_OSAVI_FINAL_texture_sub_3[['W_03_08_i']].values
Y_3.shape[0]

X_3=df_OBIA_comb_OSAVI_FINAL_texture_sub_3.drop('W_03_08_i', axis=1).values
#feats_3=X_1=df_OBIA_comb_OSAVI_FINAL_texture_sub_3.drop('W_03_08_i', axis=1).columns.tolist()
#print(feats_3)
X_3.shape[0]


#####only textural###Y_5-und X_5

df_OBIA_comb_OSAVI_FINAL_texture_sub_5=df_OBIA_comb_OSAVI_FINAL_texture[['_Contrastmean','_Correlationmean', '_Entropymean','_ASMmean', 'W_03_08_i']].dropna()
df_OBIA_comb_OSAVI_FINAL_texture_sub_5.columns.tolist()
Y_5=df_OBIA_comb_OSAVI_FINAL_texture_sub_5[['W_03_08_i']].dropna().values
Y_5.shape[0]
#feats_5=X_1=df_OBIA_comb_OSAVI_FINAL_texture_sub_5.drop('W_03_08_i', axis=1).columns.tolist()
#Y_5=df_OBIA_comb_OSAVI_FINAL_texture_sub_5[['W_03_08_i']].values
Y_5.shape[0]
Y_5.shape[1]

X_5=df_OBIA_comb_OSAVI_FINAL_texture_sub_5.drop('W_03_08_i', axis=1).dropna().values
X_5.shape[0]
X_5.shape[1]
# X_1=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean', 'W_03_08_i']].dropna().values
# X_2=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean','_CHM_max','_CHM_stdev','_CHM_range', '_CHM_count', 'V_OM','W_03_08_i']].dropna().values
# X_3=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean','_CHM_max','_CHM_stdev','V_OM','_Contrastmean','_Correlationmean', '_Entropymean','_ASMmean','W_03_08_i']].dropna().values

##Y_6 spectral + texture
#df_OBIA_comb_OSAVI_FINAL_texture_sub_3=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean','_CHM_max','_CHM_stdev','V_OM','_Contrastmean','_Correlationmean', '_Entropymean','_ASMmean', 'W_03_08_i']].dropna()
df_OBIA_comb_OSAVI_FINAL_texture_sub_6=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean','_Contrastmean','_Correlationmean', '_Entropymean','_ASMmean', 'W_03_08_i']].dropna()
df_OBIA_comb_OSAVI_FINAL_texture_sub_6.columns.tolist()
Y_6=df_OBIA_comb_OSAVI_FINAL_texture_sub_6[['W_03_08_i']].dropna().values
Y_6.shape[0]
#feats_6=X_1=df_OBIA_comb_OSAVI_FINAL_texture_sub_6.drop('W_03_08_i', axis=1).columns.tolist()
#Y_5=df_OBIA_comb_OSAVI_FINAL_texture_sub_5[['W_03_08_i']].values
Y_6.shape[0]
Y_6.shape[1]

X_6=df_OBIA_comb_OSAVI_FINAL_texture_sub_6.drop('W_03_08_i', axis=1).dropna().values
X_6.shape[0]
X_6.shape[1]


##Y_7 structural+ texture
#df_OBIA_comb_OSAVI_FINAL_texture_sub_3=df_OBIA_comb_OSAVI_FINAL_texture[['_NDVI_4_3mean','_NDVI_5_3mean','_OSAVImean','_GNDVImean', '_NDWImean', '_TSAVImean','_CHM_mean','_CHM_max','_CHM_stdev','V_OM','_Contrastmean','_Correlationmean', '_Entropymean','_ASMmean', 'W_03_08_i']].dropna()
df_OBIA_comb_OSAVI_FINAL_texture_sub_7=df_OBIA_comb_OSAVI_FINAL_texture[['_CHM_mean','_CHM_max','_CHM_stdev','V_OM','_Contrastmean','_Correlationmean', '_Entropymean','_ASMmean', 'W_03_08_i']].dropna()
df_OBIA_comb_OSAVI_FINAL_texture_sub_7.columns.tolist()
Y_7=df_OBIA_comb_OSAVI_FINAL_texture_sub_7[['W_03_08_i']].dropna().values
Y_7.shape[0]
#feats_7=X_1=df_OBIA_comb_OSAVI_FINAL_texture_sub_7.drop('W_03_08_i', axis=1).columns.tolist()
#Y_5=df_OBIA_comb_OSAVI_FINAL_texture_sub_5[['W_03_08_i']].values
Y_7.shape[0]
Y_7.shape[1]
type(Y_7)
X_7=df_OBIA_comb_OSAVI_FINAL_texture_sub_7.drop('W_03_08_i', axis=1).dropna().values
X_7.shape[0]
X_7.shape[1]
type(X_7)
#################################################################


#X_1=X_1.dropna()

Y=df_OBIA_comb_OSAVI_FINAL_texture[['W_03_08_i']].values




###################ohne ravel
x_train_1_1, x_test_1_1, y_train_1_1,y_test_1_1=train_test_split(X_1,Y_1,random_state=0, test_size=0.25)

x_train_1_2_, x_test_1_2_, y_train_1_2_,y_test_1_2_=train_test_split(X_2,Y_2,random_state=0, test_size=0.25)

x_train_1_3_, x_test_1_3_, y_train_1_3_,y_test_1_3_=train_test_split(X_3,Y_3,random_state=0, test_size=0.25)

######################################## mit ravel
x_train_1_1, x_test_1_1, y_train_1_1,y_test_1_1=train_test_split(X_1,Y_1.ravel(),random_state=0, test_size=0.25)

x_train_1_2_, x_test_1_2_, y_train_1_2_,y_test_1_2_=train_test_split(X_2,Y_2.ravel(),random_state=0, test_size=0.25)

x_train_1_3_, x_test_1_3_, y_train_1_3_,y_test_1_3_=train_test_split(X_3,Y_3.ravel(),random_state=0, test_size=0.25)
### 4 (only structural)--5- only textural
x_train_1_4_, x_test_1_4_, y_train_1_4_,y_test_1_4_=train_test_split(X_4,Y_4.ravel(),random_state=0, test_size=0.25)

x_train_1_5_, x_test_1_5_, y_train_1_5_,y_test_1_5_=train_test_split(X_5,Y_5.ravel(),random_state=0, test_size=0.25)



#create dataframe
##create dataframes to hold the classification metrics

arr_random_2=np.random.default_rng().uniform(low=5,high=10, size=[14, 12])

#model_list_2=['LR_1', 'LR_2', 'LR_3', 'SVM_1', 'SVM_2', 'SVM_3', 'RF_1', 'RF_2', 'RF_3']

model_list_2=['RF_1', 'RF_2', 'RF_3', 'RF_4', 'RF_5', 'RF_6', 'RF_7', 'SVM_1', 'SVM_2', 'SVM_3','SVM_4', 'SVM_5', 'SVM_6','SVM_7']

df_metrics_test_2_=pd.DataFrame(arr_random_2, columns=['R2', "MSE", 'RMSE', 'RMSE%', 'MAE', 'Cohens Kappa', 'Accuracy', 'BA', 'F1-Macro', 'Precision','input_features','best_params'])
##
df_metrics_test_2_['models']=model_list_2
df_metrics_test_2_.index=list(df_metrics_test_2_['models'])
#assign value to specific cell

R2_test=-34.8


df_metrics_test_2_.iloc[0:0,0:0]=R2_test
df_metrics_test_2_.iat[0,0]=R2_test
####

#model_list_2=['LR_1', 'LR_2', 'LR_3', 'SVM_1', 'SVM_2', 'SVM_3', 'RF_1', 'RF_2', 'RF_3']


df_metrics_train_2_=pd.DataFrame(arr_random_2, columns=['R2', "MSE", 'RMSE', 'RMSE%', 'MAE', 'Cohens Kappa', 'Accuracy', 'BA', 'F1-Score', 'Precision', 'input_features','best_params'])    


df_metrics_train_2_['models']=model_list_2
df_metrics_train_2_.index=list(df_metrics_train_2_['models'])

################################################################################################
##dataframe for F_1-score metrics for model evaluation
#arr_random_n=np.random.default_rng().uniform(low=5,high=10, size=[10, 8])
arr_random_n=np.random.default_rng().uniform(low=5,high=10, size=[14, 8])
model_list_F=['RF_1', 'RF_2', 'RF_3','RF_4', 'RF_5', 'SVM_1', 'SVM_2', 'SVM_3','SVM_4', 'SVM_5' ]

model_list_F_2=['RF_1', 'RF_2', 'RF_3','RF_4', 'RF_5','RF_6', 'RF_7', 'SVM_1', 'SVM_2', 'SVM_3','SVM_4', 'SVM_5', 'SVM_6', 'SVM_7' ]
print(len(model_list_F))
df_metrics_test_F_scores_=pd.DataFrame(arr_random_n, columns=['Accuracy_Test', 'Accuracy_Train', "F1-Macro_Test", "F1-Macro_Train", 'F1-Weighted_Test', 'F1-Weighted_Train' ,'F_1_Micro_Test','F_1_Micro_Train'])
##
df_metrics_test_F_scores_['models']=model_list_F_2
df_metrics_test_F_scores_.index=list(df_metrics_test_F_scores_['models'])
#assign value to specific cell


###hier die Kernel Funktionen noch ändern/ durchtesten
#### ['kernels: ','linear, 'poly', 'rbf', 'sigmoid, ''precomputed']
pipeline=Pipeline([
("scaler", StandardScaler()), 
("svm", SVC()),

])


clf=GridSearchCV(pipeline, param_grid={
"svm__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
"svm__gamma": [0.001, 0.01, 0.1, 1, 10],


})


#######################################################################
##create pipeline for hyperparameter tuning


pipeline=Pipeline([
("scaler", StandardScaler()), 
("svm", SVC()),

])

clf=GridSearchCV(pipeline, param_grid={
"svm__C":[0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
"svm__gamma": [0.001, 0.01, 0.1, 1, 10],
"kernel":['linear', 'rbf']

})




param_grid2=[{'n_estimators':[25, 50, 100, 150, 300, 500],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [3,6, 9, 15, 20, 30],
                'max_leaf_nodes': [3,6,9],
                'max_samples':[2,4,6],
                'min_samples_leaf':[1,2,4],
                'criterion':['entropy', 'gini']
}]



# param_grid_RF_random_search=RandomizedSearchCV(RandomForestClassifier(),
#                                                 param_grid2,
#                                                 cv=5,   
#                                                 scoring='accuracy',
#                                                 n_jobs=-1
#                                                 )


#################################################################################
#####################################################

##Hyperparameters selection etc.
###################################################################
###SVM
pipeline=Pipeline([
("scaler", StandardScaler()), 
("svm", SVC()),

])

clf=GridSearchCV(pipeline, param_grid={
"svm__C":[0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
"svm__gamma": [0.001, 0.01, 0.1, 1, 10],
"kernel":['linear', 'rbf']

})


###################
#####RFC

param_grid={'n_estimators':[25, 50, 100, 150],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [3,6, 9],
                'max_leaf_nodes': [3,6,9],
        
}

grid_search=GridSearchCV(RandomForestClassifier(),
                         param_grid=param_grid)


###################################################################################
################################################################################
print(np.mean(Y_1))
mean_GC=np.mean(Y_1)
('The average Growth Class of the vine stocks\
 in the investigation area is', mean_GC)
############################
dict_metrics_={}
np.random.seed(42)###To get the same random seed
#############################################
##define repeated k-fold cross validation
kf_1_1=KFold(n_splits=10, shuffle=True)


('The average Growth Class of the grapevines\
 in the investigation area is', mean_GC)
############################

np.random.seed(42)###To get the same random seed
#############################################

##create the for loop for getting all the metrics including splitting into
##test and training data, model fit with the features, metric computation and 
##storing of the data into specified dictionary
rfc=RandomForestClassifier()
lr=LogisticRegression()
gnb=GaussianNB()
X_list=[X_1, X_2, X_3, X_4, X_5, X_6, X_7]
Y_list=[Y_1, Y_2, Y_3, Y_4, Y_5, Y_6,Y_7]
Y_list_ravel=[Y_1.ravel(), Y_2.ravel(), Y_3.ravel(), Y_4.ravel(), Y_5.ravel(), Y_6.ravel(),Y_7.ravel()]
count_list_1=['1','2','3','4','5','6','7']
count_list_2=['1','4','7','2','3','5','6']

counter=[0,1,2,3,4,5,6]

RF_classifier=['RF']*7
print(RF_classifier)
SVM_classifier=['SVM']*7
feature_type=['spectral', 'spectral+ structural','spectral+structural+textural', 'structural', 'textural', 'spectral+textural', 'structural+textural']

type(X_1)
type(Y_1)
print(np.mean(Y_1))
print(np.mean(Y_2))
mean_GC=np.mean(Y_1)
############################
#zip lists
zipped_L=list(zip(X_list, Y_list, count_list_2,RF_classifier, SVM_classifier, feature_type, counter))
for j in zipped_L:
    print( j[2])
    rep=np.array(j[0]).reshape((1,-1))
print(len(zipped_L[0][1]))


L_array=[]
for j in range(len(zipped_L)):
    print(zipped_L[0])
    L_array.append(zipped_L[0])
print(len(L_array[0][0]))   

dict_metrics_={}
dic_List_3=[]

kf_1_1=KFold(n_splits=5, shuffle=True)    

x_times=3

for i in range(x_times):
    for j in zipped_L:
        for train_index, test_index in kf_1_1.split(np.array(j[0]).astype(int)):
            print("train:"+ str(train_index))
            print("test:"+ str(test_index))
            print('-----------------------')
            try:
                X_train_K1=j[0][train_index]
                X_test_K1=j[0][test_index]
                #Y_train_K1=Y_3_1_T_L_1[train_index]
                Y_train_K1=j[1][train_index]
                Y_test_K1=j[1][test_index]
            except IndexError:
                    pass
            
            
            
            #####linear Model##########################################
            print('Model scores for spectral feature inputs')
            
            
            model_LL=LinearRegression()
            model_LL.fit(X_train_K1, Y_train_K1.ravel())
            print('Model Spectral score Linear Regression')
            print(model_LL.score(X_test_K1, Y_test_K1))
            y_hatt=model_LL.predict(X_test_K1)
            
        
            
            #y_hatt_prob=model_LL.predict_proba(X_test_K1)[:,1]
            ############################################
            # fpr, tpr, threshols=roc_curve(Y_test_K1, y_hatt_prob)
            
            # plt.plot(fpr, tpr)
            
            #plt.show()
            ###################################################
            
            y_hatt_train=model_LL.predict(X_train_K1)
            ########################################################
            
            ##Classification Report Lineare Regression
            # print('Classification Report Lineare Regression Testdata')
            # classification_report(Y_test_K1, y_hatt)
            # print('Classification Report Lineare Regression Traindata')
            # classification_report(Y_test_K1, y_hatt_train)
            
            ###############
            
            print('model parameters Linear Regression')
            
            print('model intercept :', model_LL.intercept_) 
            print('model coefficients : ', model_LL.coef_) 
            print('Model score : ', model_LL.score(X_test_K1,Y_test_K1)) 
            print(model_LL.score(X_test_K1, Y_test_K1))
            
            #np.mean
            
            #df_metrics_test_2_.iat[0,0]=np.mean(model_LL.score(X_test_K1, Y_test_K1))
            
            
            #################################################################
            ############################################
            
            print('RMSE Lineare Regression Testdaten')
            rmse_test=root_mean_squared_error(Y_test_K1, y_hatt)
            #df_metrics_test_2_.iat[0,2]=rmse_test
            print('RMSE % Lineare Regression Testdaten')
            print((rmse_test/mean_GC)*100)
            #df_metrics_test_2_.iat[0,3]=(rmse_test/mean_GC)*100
        
            print('MAE Linear Regression ')
            
            print('RMSE Lineare Regression Traininsdaten')
            rmse_train=root_mean_squared_error(Y_train_K1, y_hatt_train)
            #df_metrics_test_2_.iat[0,2]=rmse_test
            
            print('RMSE % Lineare Regression Traininsdaten')
            print((rmse_train/mean_GC)*100)
            #df_metrics_train_2_.iat[0,3]=(rmse_train/mean_GC)*100
            mae=mean_absolute_error(Y_test_K1, y_hatt)
            
        
            print(mae)
            mae_train=mean_absolute_error(Y_train_K1, y_hatt_train)
        
            print('MAE Linear Regression Trainingsdaten ')
            print(mae_train)
            
            print('Mean Squared Error Linear Regression')
            print(mean_squared_error(Y_test_K1, y_hatt))
            print('R2 Lineare Regression Testdaten')
            #df_metrics_test_2_.iat[0,1]=mean_squared_error(Y_test_K1, y_hatt)
            
            print('Mean Squared Error Linear Regression Trainingsdaten')
            print(mean_squared_error(Y_train_K1, y_hatt_train))
            #df_metrics_train_2_.iat[0,1]=mean_squared_error(Y_train_K1, y_hatt_train)
            print('R2 Lineare Regression Testdaten')
            
            r2=r2_score(Y_test_K1, y_hatt)
            r2_train=r2_score(Y_train_K1, y_hatt_train)
            #df_metrics_test_2_.iat[0,0]=r2
            print(r2)
            
            
            
            
        
            print('R2 Lineare Regression Trainingsdaten')
            print(r2_train)
            

        
            svm_l=svm.SVC(kernel="linear")
            svm_l.fit(X_train_K1, Y_train_K1.ravel())
            print("SVM linear Kernel Score ")
            print(svm_l.score(X_test_K1, Y_test_K1))
            print("SVM linear Kernel Score Trainingsdaten")
            print((svm_l.score(X_train_K1, Y_train_K1)))
            y_hatt_svm=svm_l.predict(X_test_K1)
            y_hatt_svm_train=svm_l.predict(X_train_K1)
            print('SVM  linear Kernel MAE ')
            mae_svm=mean_absolute_error(Y_test_K1, y_hatt_svm)
            print(mae_svm)
            print('SVM  linear Kernel MAE Trainingsdaten ')
            mae_svm=mean_absolute_error(Y_train_K1,y_hatt_svm_train)
            print(mae_svm)
            
            print('Mean Squared Error SVM linear Kernel')
            print(mean_squared_error(Y_test_K1, y_hatt))
            print('Mean Squared Error SVM linear Kernel Trainingsdaten')
            print(mean_squared_error(Y_train_K1,y_hatt_svm_train))
            print('R2 SVM linear Kernel')
            r2_svm=r2_score(Y_test_K1, y_hatt)
            r2_svm_train=r2_score(Y_train_K1,y_hatt_svm_train)
            print(r2_svm)
            print('R2 SVM linear Kernel Trainingsdaten')
            print(r2_svm_train)
            
            ##############################SVM kernel=rbf
            
            svm_l_rbf=svm.SVC(kernel="rbf")
            svm_l_rbf.fit(X_train_K1, Y_train_K1.ravel())
            print("SVM RBF Kernel Score ")
            print(svm_l_rbf.score(X_test_K1, Y_test_K1))
            print("SVM RBF Kernel Score Trainingsdaten")
            print(svm_l_rbf.score(X_train_K1, Y_train_K1))
            
            y_hatt_svm_rbf=svm_l_rbf.predict(X_test_K1)
            
            y_hatt_svm_rbf_train=svm_l_rbf.predict(X_train_K1)
            
            print('SVM  RBF Kernel MAE ')
            mae_svm_rbf=mean_absolute_error(Y_test_K1, y_hatt_svm_rbf)
            print(mae_svm_rbf)
            print('SVM  RBF Kernel MAE Trainingsdaten ')
            mae_svm_rbf_train=mean_absolute_error(Y_train_K1, y_hatt_svm_rbf_train)
            print(mae_svm_rbf_train)
            
            
            print('Mean Squared Error SVM RBF Kernel')
            print(mean_squared_error(Y_test_K1, y_hatt_svm_rbf))
            print('Mean Squared Error SVM RBF Kernel Trainingsdaten')
            print(mean_squared_error(Y_train_K1, y_hatt_svm_rbf_train))
            print('R2 SVM RBF Kernel')
            r2_svm_rbf=r2_score(Y_test_K1, y_hatt_svm_rbf)
            print(r2_svm_rbf)
            print('R2 SVM RBF Kernel Trainingsdaten')
            r2_svm_rbf_train=r2_score(Y_train_K1,y_hatt_svm_rbf_train)
            print(r2_svm_rbf)
            
 
            
            neigh=KNeighborsClassifier(n_neighbors=3)
            
            
            neigh.fit(X_train_K1, Y_train_K1.ravel())
            print("KNN Score ")
            print(neigh.score(X_test_K1, Y_test_K1))
            print("KNN Score Training")
            print(neigh.score(X_train_K1, Y_train_K1))
            y_hatt_knn=neigh.predict(X_test_K1)
            y_hatt_knn_train=neigh.predict(X_train_K1)
            
            print('KNN MAE ')
            mae_knn=mean_absolute_error(Y_test_K1, y_hatt_knn)
            mae_knn_train=mean_absolute_error(Y_train_K1, y_hatt_knn_train)
            print(mae_knn)
            print('KNN MAE Trainingsdaten')
            print(mae_knn_train)
            print('Mean Squared Error KNN')
            print(mean_squared_error(Y_test_K1, y_hatt_knn))
            print('Mean Squared Error KNN Trainingsdaten')
            print(mean_squared_error(Y_train_K1, y_hatt_knn_train))
            print('R2 KNN')
            r2_knn=r2_score(Y_test_K1, y_hatt_knn)
            
            print(r2_knn)
            print('R2 KNN Trainingsdaten')
            r2_knn=r2_score(Y_train_K1,y_hatt_knn_train)
            
            ################################################
        
            
            
            ############################################################
            ####SVM hyper tuned
            
            pipeline=Pipeline(steps=[
            ("scaler", StandardScaler()), 
            ("svm", SVC())])
            
            

            param_grid={
                "svm__C":[0.001, 0.01, 0.1, 1, 10],
                "svm__gamma": [0.001, 0.01, 0.1, 1, 10]}
                
                
             # })
             
             
            
            # param_grid_SVC=[
            # {"C":[0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
            # "gamma": [0.001, 0.01, 0.1, 1, 10],
            # "kernel":['linear']},
            # {"C":[0.001, 0.01, 0.1, 1, 10, 15, 20, 100, 1000],
            # "gamma": [0.001, 0.01, 0.1, 1, 10],
            # "kernel":['rbf']},
            # ]
            
            param_grid_SVC={'C':[0.1, 1, 10, 100, 100],
                              'gamma':[1,0.1, 0.01, 0.001, 0.0001],
                            'kernel':['linear', 'rbf']}
                
                
        
            clf=GridSearchCV(pipeline,param_grid)
            clf=GridSearchCV(SVC(), param_grid_SVC, refit=True, verbose=3)
            #     )
        
            m1k=clf.fit(X_train_K1, Y_train_K1.ravel())
            
            
            print('best params for SVM HT Model for X1 features')
            print(m1k.best_params_)
            
            print(m1k.best_params_)
            print(clf.best_params_)

            y_hatk=m1k.predict(X_test_K1)
            y_hatk_train=m1k.predict(X_train_K1)
            print(len(y_hatk))
            print(len(Y_test_K1)) 
            
            
            
            ###########################################################################
            ###########################################################################
            
            
            print('SVM Hyper- tuned Grid Search')
            print('Score SVM Hyper- tuned Grid Search:'+str(m1k.score(X_test_K1, Y_test_K1)))
            print('Score SVM Hyper- tuned Grid Search Trainingsdaten:'+str(m1k.score(X_train_K1, Y_train_K1)))
            print('RMSE SVM Hyper- tuned Grid Search:'+str(root_mean_squared_error(Y_train_K1,y_hatk_train)))
            df_metrics_test_2_.iat[7+j[6],2]=root_mean_squared_error(Y_test_K1, y_hatk)
            rmse_svm_1_test=root_mean_squared_error(Y_test_K1, y_hatk)
            print('RMSE % SVM HT Testdata')
            print((rmse_svm_1_test/mean_GC)*100)
            #rmse_svm_1_test=(rmse_svm_1_test/mean_GC)*100
            
            df_metrics_test_2_.iat[7+j[6],3]=(root_mean_squared_error(Y_test_K1, y_hatk)/mean_GC)*100
            rmse_percent_SVM_1_test=((root_mean_squared_error(Y_test_K1, y_hatk))/mean_GC)*100

            df_SVM_1_HT_Train_df=pd.DataFrame(classification_report(Y_train_K1,y_hatk_train, output_dict=True))
            print('RMSE Train SVM Hyper- tuned Grid Search:'+str(root_mean_squared_error(Y_train_K1,y_hatk_train)))
            df_metrics_train_2_.iat[7+j[6],2]=root_mean_squared_error(Y_train_K1,y_hatk_train)
            rmse_train_svm_1=root_mean_squared_error(Y_train_K1,y_hatk_train)
            print('RMSE % SVM HT Traintdata')
            print((root_mean_squared_error(Y_train_K1,y_hatk_train)/mean_GC)*100)
            rmse_percent_SVM_1_train=(root_mean_squared_error(Y_train_K1,y_hatk_train)/mean_GC)*100
            df_metrics_train_2_.iat[7+j[6],3]=(root_mean_squared_error(Y_train_K1,y_hatk_train)/mean_GC)*100
            
            
            
            print('Accuracy Score SVM Hyper- tuned Grid Search:'+str(accuracy_score(Y_test_K1, y_hatk)))
            accuracy_svm_test_1=accuracy_score(Y_test_K1, y_hatk)
            
            print('Accuracy Score SVM Hyper- tuned Grid Search Trainingsdaten:'+str(accuracy_score(Y_train_K1,y_hatk_train)))
            #print('Average Precision score:'+str(average_precision_score(Y_test_K1,y_hatk)))
           # print('Balanced Accuracy Score:'+str(balanced_accuracy_score(Y_test_K1,y_hatk).reshape(1,-1)))
            accuracy_SVM_1_train=accuracy_score(Y_train_K1,y_hatk_train)
            print('MAE SVM Hyper- tuned Grid Search:'+ str(mean_absolute_error(Y_test_K1, y_hatk)))
            print('MAE SVM Hyper- tuned Grid Search Trainingsdaten:'+ str(mean_absolute_error(Y_train_K1,y_hatk_train)))
            print('MSE SVM Hyper- tuned Grid Search:'+str(mean_squared_error(Y_test_K1,y_hatk)))
            df_metrics_test_2_.iat[7+j[6],1]=mean_squared_error(Y_test_K1,y_hatk)
            print('MSE SVM Hyper- tuned Grid Search Trainingsdaten:'+str(mean_squared_error(Y_train_K1,y_hatk_train)))
            r2=r2_score(Y_test_K1,y_hatk)
            df_metrics_test_2_.iat[7+j[6],0]=r2
            
            r2_train=r2_score(Y_train_K1, y_hatk_train)
            df_metrics_train_2_.iat[7+j[6],0]=r2
            print('R2 SVM Hyper- tuned Grid Search:'+str(r2))
            
            print('R2 SVM Hyper- tuned Grid Search Trainingsdaten:'+str(r2_train))
            print('Cohens Kappa score SVM Hyper- tuned Grid Search:'+str(cohen_kappa_score(Y_test_K1,y_hatk)))
            df_metrics_test_2_.iat[7+j[6],5]=cohen_kappa_score(Y_test_K1,y_hatk)
            print('Cohens Kappa score SVM Hyper- tuned Grid Search Traininsdaten:'+str(cohen_kappa_score(Y_train_K1,y_hatk_train)))
            df_metrics_train_2_.iat[7+j[6],5]=cohen_kappa_score(Y_train_K1,y_hatk_train)
            print('SVM HT Accuracy Score Testdata')
            print(accuracy_score(Y_test_K1, y_hatk))
            df_metrics_test_2_.iat[7+j[6],6]=accuracy_score(Y_test_K1, y_hatk)
            accuracy_svm=accuracy_score(Y_test_K1, y_hatk)
            #accuracy_svm_train=accuracy_score(Y_test_K1, y_hatk)
            accuraccy_svm_train=accuracy_score(Y_train_K1,y_hatk_train)
            print('SVM HT Balanced Accuracy Score Testdata')
            print(balanced_accuracy_score(Y_test_K1, y_hatk))
            df_metrics_test_2_.iat[7+j[6],7]=balanced_accuracy_score(Y_test_K1, y_hatk)
            
            metricslist1=[]
            metricslist1+=r2
            metricslist1+=rmse_train
            metricslist2=[]
            metricslist2+=r2_train
            metricslist2+=r2_train
            #### who do you eant here 2?
            #dict_metrics__={}
        #####dictiionary definieren und key mit den model metriken verbinden
            dict_metrics_['rmse_SVM_test_'+'-'+str(j[2])]=rmse_svm_1_test
            dict_metrics_['rmse_SVM_train_'+'-'+str(j[2])]=rmse_train_svm_1
            dict_metrics_['rmse_SVM_test_accuracy'+'-'+str(j[2])]=accuracy_svm_test_1
            dict_metrics_['rmse_SVM_train_accuracy'+'-'+str(j[2])]=accuracy_SVM_1_train
            dict_metrics_['rmse_SVM_test_rmse_percent'+'-'+str(j[2])]=rmse_percent_SVM_1_test
            dict_metrics_['rmse_SVM_train_rmse_percent'+'-'+str(j[2])]=rmse_percent_SVM_1_train
            dict_metrics_['SVM_metrics'+'-'+str(j[2])]=[rmse_svm_1_test]

        
            ################################################
            for i in metricslist1:
                print(i)
            compare_metrics=pd.DataFrame({'SVM_2_HT_Test':[r2, rmse_train,accuracy_svm_test_1],
                                           'SVM_2_HT_Train':[r2_train, rmse_train, accuracy_svm_test_1],
                                           'SVM_1_HT_Test':[r2, rmse_train, accuracy_svm_test_1],
                                           'SVM_1_HT_Train':[r2_train, rmse_train_svm_1,accuracy_svm_test_1]}, index=[0,1,2])
            
            
            compare_metrics.plot.bar()
            
            
            for i in metricslist1:
                print(i)
            compare_metrics=pd.DataFrame({'SVM_2_HT_Test':[r2, rmse_train,accuracy_svm_test_1],
                                           'SVM_2_HT_Train':[r2_train, rmse_train, accuracy_svm_test_1],
                                           'SVM_1_HT_Test':[r2, rmse_train, accuracy_svm_test_1],
                                           'SVM_1_HT_Train':[r2_train, rmse_train_svm_1,accuracy_svm_test_1]}, index=['r2','rmse','overall accuracy'])
            
            
            compare_metrics.plot.bar()
            
            
            print(accuracy_score(Y_train_K1, y_hatk_train))
            df_metrics_train_2_.iat[7+j[6],6]=accuracy_score(Y_train_K1, y_hatk_train)
            accuracy_svm_train=accuracy_score(Y_train_K1, y_hatk_train)
            print('SVM HT Balanced Accuracy Score Traindata')
            print(balanced_accuracy_score(Y_train_K1, y_hatk_train))
            df_metrics_train_2_.iat[7+j[6],7]=balanced_accuracy_score(Y_train_K1, y_hatk_train)
            
            print('SVM HT Precision Score Testdata')
            print(precision_score(Y_test_K1, y_hatk, average='macro'))
            df_metrics_test_2_.iat[7+j[6],9]=precision_score(Y_test_K1, y_hatk, average='macro')
            
            print('SVM HT Precision Score Traindata')
            print(precision_score(Y_train_K1, y_hatk_train, average='macro'))
            df_metrics_train_2_.iat[7+j[6],9]=precision_score(Y_train_K1, y_hatk_train, average='macro')
            
            
            print('SVM HT F1 Score macro Testdata')
            print(f1_score(Y_test_K1, y_hatk, average='macro'))
            
            f_1_score_macro_test=f1_score(Y_test_K1, y_hatk, average='macro')
            print('SVM HT F1 Score- macro Traindata')
            print(f1_score(Y_train_K1, y_hatk_train, average='macro'))
            f_1_score_macro_train=f1_score(Y_train_K1, y_hatk_train, average='macro')
            f_1_score_macro_train
            ##############################evaluation difference between the metrics
            f_1_score_macro_train_weighted=f1_score(Y_train_K1, y_hatk_train, average='weighted')
            f_1_score_macro_train_weighted
            f_1_score_macro_train_micro=f1_score(Y_train_K1, y_hatk_train, average='micro')
            f_1_score_macro_train_micro
            ############################
            print('SVM HT F1 Score- micro Testdata')
            print(f1_score(Y_test_K1, y_hatk, average='micro'))
            
            print('SVM HT F1 Score micro Traindata')
            print(f1_score(Y_train_K1, y_hatk_train, average='micro'))
            
            print('SVM HT F1 Score- weighted Testdata')
            print(f1_score(Y_test_K1, y_hatk, average='weighted'))
        
            print('SVM HT F1 Score weighted Traindata')
            print(f1_score(Y_train_K1, y_hatk_train, average='weighted'))
            
            
            print('SVM HT F1 Score- samples Testdata')
            print(f1_score(Y_test_K1, y_hatk, average='weighted'))
            
            ##F-1 scores test
            df_metrics_test_F_scores_.iat[7+j[6],8]
            accuracy_svm_test_1
            
            F1_macro_test_SVM=f1_score(Y_test_K1, y_hatk, average='macro')
        
            F1_weighted_test_SVM=f1_score(Y_test_K1, y_hatk, average='weighted')
            
            F1_micro_test_SVM=f1_score(Y_test_K1, y_hatk, average='micro')
            
            
            #F1_samples_test_SVM=f1_score(Y_test_K1, y_hatk, average='samples')
            
            df_metrics_test_F_scores_.iat[7+j[6],0]=accuracy_svm_test_1
            df_metrics_test_F_scores_.iat[7+j[6],2]=F1_macro_test_SVM
            df_metrics_test_F_scores_.iat[7+j[6],4]=F1_weighted_test_SVM
            df_metrics_test_F_scores_.iat[7+j[6],6]=F1_micro_test_SVM
            
        
            print('SVM HT F1 Score samples Traindata')
            print(f1_score(Y_train_K1, y_hatk_train, average='weighted'))
            ########F1 scores Train
            accuracy_svm_train
            
            F1_macro_train_SVM=f1_score(Y_train_K1, y_hatk_train, average='macro')
            
            F1_weighted_train_SVM=f1_score(Y_train_K1, y_hatk_train, average='weighted')
            
            #F1_macro_train_SVM=f1_score(Y_train_K1, y_hatk_train, average='samples')
            
            F1_micro_train_SVM=f1_score(Y_train_K1, y_hatk_train, average='micro')
            
            
            df_metrics_test_F_scores_.iat[7+j[6],1]=accuracy_svm_train
            df_metrics_test_F_scores_.iat[7+j[6],3]=F1_macro_train_SVM
            df_metrics_test_F_scores_.iat[7+j[6],5]=F1_weighted_train_SVM
            df_metrics_test_F_scores_.iat[7+j[6],7]=F1_micro_train_SVM
            
            
            ####################Hier Test des Erzeugens der Confuksion- Matrix
            
            ###müsste hier nochmal umgedreht werden um zu passen (Lablel zu vorhergesagten Daten)
            matrix_SVM=confusion_matrix(y_hatk, Y_test_K1)
            matrix_df=pd.DataFrame(matrix_SVM, columns=['0 W P', '1 W P', '3W P', '5W P', '7 W P', '9 W P'])
            ###matrix to dataframe
            ####
           ###inverted matrix to test which site of this is the right matrix
            matrix_SVM=confusion_matrix(Y_test_K1, y_hatk)
            matrix_df=pd.DataFrame(matrix_SVM, columns=['0 W P', '1 W P', '3W P', '5W P', '7 W P', '9 W P'])
        
           
            
            sumpa=matrix_df.sum(axis=0)
            sumpa=sumpa.tolist()
            print(matrix_df)
            sumpa.append(1.0)
            print(len(sumpa))
            print(matrix_df.sum(axis=0))
            sumna=matrix_df.sum(axis=1)
            print(matrix_df.sum(axis=1))
            ###Hier in Confusion matrix einfügen
            matrix_df['Sum']=sumna
            
            matrix_df.loc[len(matrix_df.index)]=sumpa
            
            ###Useraccuracy
            UseraccL=[]
            Useracc1=(matrix_df.iat[0,0]/matrix_df.iat[0,6])*100
            print(Useracc1)
            print(matrix_df.iat[0,0])
            print(matrix_df.iat[0,6])
            UseraccL.append(Useracc1)
            Useracc2=(matrix_df.iat[1,1]/matrix_df.iat[1,6])*100
            UseraccL.append(Useracc2)
            Useracc3=(matrix_df.iat[2,2]/matrix_df.iat[2,6])*100
            UseraccL.append(Useracc3)
            Useracc4=(matrix_df.iat[3,3]/matrix_df.iat[3,6])*100
            UseraccL.append(Useracc4)
            Useracc5=(matrix_df.iat[4,4]/matrix_df.iat[4,6])*100
            UseraccL.append(Useracc5)
            Useracc6=(matrix_df.iat[5,5]/matrix_df.iat[5,6])*100
            UseraccL.append(Useracc6)
            UseraccL.append('-')
            matrix_df['UA']=UseraccL
            
            ####Producer Accuraccy
            
            ProduceraccL=[]
            PAacc1=(matrix_df.iat[0,0]/matrix_df.iat[6,0])*100
            print(PAacc1)
            ProduceraccL.append(PAacc1)
            PAacc2=(matrix_df.iat[1,1]/matrix_df.iat[6,1])*100
            print(matrix_df.iat[1,1])
            print(matrix_df.iat[6,1])
            ProduceraccL.append(PAacc2)
            PAacc3=(matrix_df.iat[2,2]/matrix_df.iat[6,2])*100
            ProduceraccL.append(PAacc3)
            PAacc4=(matrix_df.iat[3,3]/matrix_df.iat[6,3])*100
            ProduceraccL.append(PAacc4)
            PAacc5=(matrix_df.iat[4,4]/matrix_df.iat[6,4])*100
            ProduceraccL.append(PAacc5)
            PAacc6=(matrix_df.iat[5,5]/matrix_df.iat[6,5])*100
            ProduceraccL.append(PAacc6)
            ProduceraccL.append('-')
            ProduceraccL.append('-')    
            
            matrix_df.loc[len(matrix_df.index)]=ProduceraccL
            ###
            ### Overall accuracy add
            ###
            matrix_df.iat[6,7]='Overall Accuracy/F-1 '
            matrix_df.iat[7,7]=str(np.round(accuracy_svm,2))+'/ '+str(np.round(f_1_score_macro_test,2))
            matrix_df.iat[7,6]=str('SVM-1-Test')
            matrix_df.index=['0 W L','1 WL', '3 W L', '5W L', '7 WL', '9 WL', 'Sum', 'PA' ]
            
            
            
            # matrix_df.to_excel('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1.xlsx')
            # matrix_df.to_csv('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1.csv')
           
            
           
            matrix_SVM.diagonal()/matrix_SVM.sum(axis=1)
            cm_1=matrix_SVM.astype('float')/matrix_SVM.sum(axis=1)[:, np.newaxis]
            ##Confusion Matrix List
            CL_list=['0','1','3','5','7','9']
        
            cm_1_df=pd.DataFrame.from_records(cm_1)
            cm_1_df.rename(columns={0:'0_W', 1:'1_W',2:'3_W', 3:'W_5', 4:'W_7', 5:'W_9'}, inplace=True)
            cm_1_df.index=list(CL_list)
            ################
            ##Calculate User and Produser Accuraccy
            #############################################
            sumpa=cm_1.sum(axis=0)
            sumpa=sumpa.tolist()
            print(sumpa)
            sumpa.append(1.0)
            print(len(sumpa))
            print(cm_1.sum(axis=0))
            sumna=cm_1.sum(axis=1)
            print(cm_1.sum(axis=1))
            ###Hier in Confusion matrix einfügen
            cm_1_df['Sum']=sumna
            
            cm_1_df.loc[len(cm_1_df.index)]=sumpa
            
            #cm_1_df.at['6']=sumpa
            
            ###Useraccuracy
            UseraccL=[]
            Useracc1=(cm_1_df.iat[0,0]/cm_1_df.iat[0,6])*100
            print(Useracc1)
            print(cm_1_df.iat[0,0])
            print(cm_1_df.iat[0,6])
            UseraccL.append(Useracc1)
            Useracc2=(cm_1_df.iat[1,1]/cm_1_df.iat[1,6])*100
            UseraccL.append(Useracc2)
            Useracc3=(cm_1_df.iat[2,2]/cm_1_df.iat[2,6])*100
            UseraccL.append(Useracc3)
            Useracc4=(cm_1_df.iat[3,3]/cm_1_df.iat[3,6])*100
            UseraccL.append(Useracc4)
            Useracc5=(cm_1_df.iat[4,4]/cm_1_df.iat[4,6])*100
            UseraccL.append(Useracc5)
            Useracc6=(cm_1_df.iat[5,5]/cm_1_df.iat[5,6])*100
            UseraccL.append(Useracc6)
            UseraccL.append('-')
            cm_1_df['UA']=UseraccL
            
            ###
            
            ####Producer Accuraccy
            
            ProduceraccL=[]
            PAacc1=(cm_1_df.iat[0,0]/cm_1_df.iat[6,0])*100
            print(PAacc1)
            ProduceraccL.append(PAacc1)
            PAacc2=(cm_1_df.iat[1,1]/cm_1_df.iat[6,1])*100
            print(cm_1_df.iat[1,1])
            print(cm_1_df.iat[6,1])
            ProduceraccL.append(PAacc2)
            PAacc3=(cm_1_df.iat[2,2]/cm_1_df.iat[6,2])*100
            ProduceraccL.append(PAacc3)
            PAacc4=(cm_1_df.iat[3,3]/cm_1_df.iat[6,3])*100
            ProduceraccL.append(PAacc4)
            PAacc5=(cm_1_df.iat[4,4]/cm_1_df.iat[6,4])*100
            ProduceraccL.append(PAacc5)
            PAacc6=(cm_1_df.iat[5,5]/cm_1_df.iat[6,5])*100
            ProduceraccL.append(PAacc6)
            ProduceraccL.append('-')
            ProduceraccL.append('-')    
            
            cm_1_df.loc[len(cm_1_df.index)]=ProduceraccL
            
            ### Overall accuracy add
            
            cm_1_df.iat[6,7]='Overall Accuracy/F-1 '
            cm_1_df.iat[7,7]=str(np.round(accuracy_svm_train,2))+'/ '+str(np.round(f_1_score_macro_train,2))
            cm_1_df.iat[7,6]=str('SVM-Train')
            cm_1_df.index=['0 W L','1 WL', '3 W L', '5W L', '7 WL', '9 WL', 'Sum', 'PA' ]
            
            cm1_test=cm_1_df
            matrix_df_test=matrix_df
            
            # cm_1_df.to_excel('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1_prob.xlsx')
            # cm_1_df.to_csv('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1_prob.csv')
           
            
            dict_metrics_.update({'SVM_'+str(j[2])+'_metrics': [rmse_svm_1_test,rmse_percent_SVM_1_test, accuracy_svm_test_1, cm1_test, matrix_df_test]}) 
            
    
            
           
            confusion_matrix_L=[]
            confusionmatrix_L_prob=[]
            confusion_matrix_L.append(matrix_df)
            confusionmatrix_L_prob.append(cm_1_df)
            
            ###########################
            ################################################################
            ####
            matrix_SVM_train=confusion_matrix(Y_train_K1,y_hatk_train, )
            ###matrix_SVM_train_switch=confusion_matrix(Y_train_K1, y_hatk_train,)
            matrix_df=pd.DataFrame(matrix_SVM_train, columns=['0 W P', '1 W P', '3W P', '5W P', '7 W P', '9 W P'])
            ###matrix to dataframe
            ####
            sumpa=matrix_df.sum(axis=0)
            sumpa=sumpa.tolist()
            print(matrix_df)
            sumpa.append(1.0)
            print(len(sumpa))
            print(matrix_df.sum(axis=0))
            sumna=matrix_df.sum(axis=1)
            print(matrix_df.sum(axis=1))
            ###Hier in Confusion matrix einfügen
            matrix_df['Sum']=sumna
            
            matrix_df.loc[len(matrix_df.index)]=sumpa
            
            ###Useraccuracy
            UseraccL=[]
            Useracc1=(matrix_df.iat[0,0]/matrix_df.iat[0,6])*100
            print(Useracc1)
            print(matrix_df.iat[0,0])
            print(matrix_df.iat[0,6])
            UseraccL.append(Useracc1)
            Useracc2=(matrix_df.iat[1,1]/matrix_df.iat[1,6])*100
            UseraccL.append(Useracc2)
            Useracc3=(matrix_df.iat[2,2]/matrix_df.iat[2,6])*100
            UseraccL.append(Useracc3)
            Useracc4=(matrix_df.iat[3,3]/matrix_df.iat[3,6])*100
            UseraccL.append(Useracc4)
            Useracc5=(matrix_df.iat[4,4]/matrix_df.iat[4,6])*100
            UseraccL.append(Useracc5)
            Useracc6=(matrix_df.iat[5,5]/matrix_df.iat[5,6])*100
            UseraccL.append(Useracc6)
            UseraccL.append('-')
            matrix_df['UA']=UseraccL
            
            ###
            
            
            
            
            ####Producer Accuraccy
            
            ProduceraccL=[]
            PAacc1=(matrix_df.iat[0,0]/matrix_df.iat[6,0])*100
            print(PAacc1)
            ProduceraccL.append(PAacc1)
            PAacc2=(matrix_df.iat[1,1]/matrix_df.iat[6,1])*100
            print(matrix_df.iat[1,1])
            print(matrix_df.iat[6,1])
            ProduceraccL.append(PAacc2)
            PAacc3=(matrix_df.iat[2,2]/matrix_df.iat[6,2])*100
            ProduceraccL.append(PAacc3)
            PAacc4=(matrix_df.iat[3,3]/matrix_df.iat[6,3])*100
            ProduceraccL.append(PAacc4)
            PAacc5=(matrix_df.iat[4,4]/matrix_df.iat[6,4])*100
            ProduceraccL.append(PAacc5)
            PAacc6=(matrix_df.iat[5,5]/matrix_df.iat[6,5])*100
            ProduceraccL.append(PAacc6)
            ProduceraccL.append('-')
            ProduceraccL.append('-')    
            
            matrix_df.loc[len(matrix_df.index)]=ProduceraccL
            
            ### Overall accuracy add
            
            matrix_df.iat[6,7]='Overall Accuracy/F-1 '
            matrix_df.iat[7,7]=str(np.round(accuraccy_svm_train,2))+'/ '+str(np.round(f_1_score_macro_train,2))
            
            matrix_df.index=['0 W L','1 WL', '3 W L', '5W L', '7 WL', '9 WL', 'Sum', 'PA' ]
            
           
            matrix_SVM_train.diagonal()/matrix_SVM_train.sum(axis=1)
            cm_1=matrix_SVM_train.astype('float')/matrix_SVM_train.sum(axis=1)[:, np.newaxis]
            ##Confusion Matrix List
            CL_list=['0','1','3','5','7','9']
        
            cm_1_df=pd.DataFrame.from_records(cm_1)
            cm_1_df.rename(columns={0:'0_W', 1:'1_W',2:'3_W', 3:'W_5', 4:'W_7', 5:'W_9'}, inplace=True)
            cm_1_df.index=list(CL_list)
            ################
            ##Calculate User and Produser Accuraccy
            #############################################
            sumpa=cm_1.sum(axis=0)
            sumpa=sumpa.tolist()
            print(sumpa)
            sumpa.append(1.0)
            print(len(sumpa))
            print(cm_1.sum(axis=0))
            sumna=cm_1.sum(axis=1)
            print(cm_1.sum(axis=1))
            ###Hier in Confusion matrix einfügen
            cm_1_df['Sum']=sumna
            
            cm_1_df.loc[len(cm_1_df.index)]=sumpa
            
            #cm_1_df.at['6']=sumpa
            
            ###Useraccuracy
            UseraccL=[]
            Useracc1=(cm_1_df.iat[0,0]/cm_1_df.iat[0,6])*100
            print(Useracc1)
            print(cm_1_df.iat[0,0])
            print(cm_1_df.iat[0,6])
            UseraccL.append(Useracc1)
            Useracc2=(cm_1_df.iat[1,1]/cm_1_df.iat[1,6])*100
            UseraccL.append(Useracc2)
            Useracc3=(cm_1_df.iat[2,2]/cm_1_df.iat[2,6])*100
            UseraccL.append(Useracc3)
            Useracc4=(cm_1_df.iat[3,3]/cm_1_df.iat[3,6])*100
            UseraccL.append(Useracc4)
            Useracc5=(cm_1_df.iat[4,4]/cm_1_df.iat[4,6])*100
            UseraccL.append(Useracc5)
            Useracc6=(cm_1_df.iat[5,5]/cm_1_df.iat[5,6])*100
            UseraccL.append(Useracc6)
            UseraccL.append('-')
            cm_1_df['UA']=UseraccL
            
            ###
            
            ####Producer Accuraccy
            
            ProduceraccL=[]
            PAacc1=(cm_1_df.iat[0,0]/cm_1_df.iat[6,0])*100
            print(PAacc1)
            ProduceraccL.append(PAacc1)
            PAacc2=(cm_1_df.iat[1,1]/cm_1_df.iat[6,1])*100
            print(cm_1_df.iat[1,1])
            print(cm_1_df.iat[6,1])
            ProduceraccL.append(PAacc2)
            PAacc3=(cm_1_df.iat[2,2]/cm_1_df.iat[6,2])*100
            ProduceraccL.append(PAacc3)
            PAacc4=(cm_1_df.iat[3,3]/cm_1_df.iat[6,3])*100
            ProduceraccL.append(PAacc4)
            PAacc5=(cm_1_df.iat[4,4]/cm_1_df.iat[6,4])*100
            ProduceraccL.append(PAacc5)
            PAacc6=(cm_1_df.iat[5,5]/cm_1_df.iat[6,5])*100
            ProduceraccL.append(PAacc6)
            ProduceraccL.append('-')
            ProduceraccL.append('-')    
            
            cm_1_df.loc[len(cm_1_df.index)]=ProduceraccL
            
            ### Overall accuracy add
            
            cm_1_df.iat[6,7]='Overall Accuracy/F-1 '
            cm_1_df.iat[7,7]=str(np.round(accuracy_svm_train,2))+'/ '+str(np.round(f_1_score_macro_train,2))
            
            cm_1_df.index=['0 W L','1 WL', '3 W L', '5W L', '7 WL', '9 WL', 'Sum', 'PA' ]
            
            
            
            confusion_matrix_L.append(matrix_df)
            confusionmatrix_L_prob.append(cm_1_df)
            
            
            dict_metrics_.update({'rmse_SVM_'+str(j[2])+'_train_accuracy': [cm_1_df]}) 
            dict_metrics_.update({'rmse_SVM_1_train_accuracy': [matrix_df]})   
            'rmse_SVM_test_'+'-'+str(j[2])
            
            #'+str(count_list_2[i])+'
            
            dict_metrics_.update({'SVM_'+str(j[2])+'_metrics': [cm_1_df]}) 
            dict_metrics_.update({'SVM_'+str(j[2])+'_metrics': [matrix_df]}) 
            
            # cm1_test=cm_1_df
            # matrix_df_test=matrix_df
            
            dict_metrics_.update({'SVM_'+str(j[2])+'_metrics': [rmse_svm_1_test,rmse_percent_SVM_1_test, accuracy_svm_test_1,cm1_test, cm_1_df,rmse_train_svm_1, rmse_percent_SVM_1_train,accuracy_SVM_1_train, matrix_df_test, matrix_df]})
            ##show the key value pairs of the dictionary
            
            print(dict_metrics_.items())
            for i in dict_metrics_.items():
                print(i)
            
            #######################################################################################
            
            
            ################################################################
            ##Gaussian Process Regressor
            kernel=RBF(length_scale=1e+12)+WhiteKernel(noise_level=1.0)
            
            gpr1=GaussianProcessRegressor(kernel=kernel, alpha=0.0)
            gpr1.fit(X_train_K1, Y_train_K1.ravel())
            
            
            print("GPR 1 Score ")
            print(gpr1.score(X_test_K1, Y_test_K1))
            print("GPR 1 Score Trainingsdaten ")
            print(gpr1.score(X_train_K1, Y_train_K1))
            
            y_hatt_gpr1=gpr1.predict(X_test_K1)
            y_hatt_gpr1_train=gpr1.predict(X_train_K1)
            
            print('GPR 1 MAE ')
            mae_gpr_1=mean_absolute_error(Y_test_K1, y_hatt_gpr1)
            print(mae_gpr_1)
            print('GPR 1 MAE Trainingsdaten ')
            mae_gpr_1_train=mean_absolute_error(Y_train_K1, y_hatt_gpr1_train)
            print(mae_gpr_1_train)
            print('Mean Squared Error GPR 1')
            print(mean_squared_error(Y_test_K1, y_hatt_gpr1))
            print('Mean Squared Error GPR 1 Traingsdaten')
            print(mean_squared_error(Y_train_K1, y_hatt_gpr1_train))
            
            print('R2  GPR 1')
            r2_gpr_1=r2_score(Y_test_K1, y_hatt_gpr1)
            print(r2_gpr_1)
            print('R2  GPR 1 Trainingsdaten')
            r2_gpr_1_train=r2_score(Y_train_K1, y_hatt_gpr1_train)
            print(r2_gpr_1_train)
            
        
            
            ##################################################################
            
            ########################################################
            ##### Random forest Classifier--- Hyper tune grid search parameter
            
            #########################################################
            
            
            param_grid={'n_estimators':[25, 50, 100, 150],
                        'max_features': ['sqrt', 'log2', None],
                        'max_depth': [3,6, 9],
                        'max_leaf_nodes': [3,6,9],
                
                }
            #################################################################
            grid_search=GridSearchCV(RandomForestClassifier(),
                                     param_grid=param_grid)
            
            grid_search2=GridSearchCV(RandomForestClassifier(),
                                     param_grid=param_grid)
            
            
            
            pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier())
        ])
        
        
            # X_train_K1=X_1_1_np[train_index]
            # Y_train_K1=Y_1_1[train_index]
            
            # X_test_K1=X_1_1_np[test_index]
            # Y_test_K1=Y_1_1[test_index]
        
            m1Rkv=grid_search.fit(X_train_K1, Y_train_K1.ravel())
            m1Rkv.best_params_
        
        
            m1Rkv.fit(X_test_K1, Y_test_K1.ravel())
            
            
            df_metrics_train_2_.iat[j[6],11]=' Max_depth: '+ str(m1Rkv.best_params_['max_depth'])+' max_features: '+str(m1Rkv.best_params_['max_features'])+\
            ' max_leaf_nodes : '+ str(m1Rkv.best_params_['max_leaf_nodes'])+' n_estimators: '+ str(m1Rkv.best_params_['max_leaf_nodes'])
            
            #df_metrics_test_2_.iat[6,11]='C '+str(m1Rkv.best_params_['svm__C'])+' , Gamma '+str(m1Rkv.best_params_['svm__gamma'])
            print('RF HT best params for X1 features')
            m1Rkv.best_index_
        

            y_hatR1kv=m1Rkv.predict(X_test_K1)
            y_hatR1kv_train=m1Rkv.predict(X_train_K1)

            
            print('RF 1 HT Classification report Testdata')
            print(classification_report(Y_test_K1, y_hatR1kv))
            df_RF_1_HT_Test_df=pd.DataFrame(classification_report(Y_test_K1, y_hatR1kv, output_dict=True))
            print('RF 1 HT Classification report Traindata')
            print(classification_report(Y_train_K1, y_hatR1kv_train))
            df_RF_1_HT_Train_df=pd.DataFrame(classification_report(Y_train_K1, y_hatR1kv_train, output_dict=True))
            print('RMSE RF Testdaten ')
            rf_rmse_test=root_mean_squared_error(Y_test_K1, y_hatR1kv)
            print(rf_rmse_test)
            df_metrics_test_2_.iat[j[6],2]=rf_rmse_test
            rf_rmse_train=root_mean_squared_error(Y_train_K1, y_hatR1kv_train)
            print(rf_rmse_test)
            
            print('RMSE % RF HT Testdata')
            print((root_mean_squared_error(Y_test_K1, y_hatR1kv)/mean_GC)*100)
            df_metrics_test_2_.iat[j[6],3]=(root_mean_squared_error(Y_test_K1, y_hatR1kv)/mean_GC)*100
            rmse_perc_test_RF_1=((root_mean_squared_error(Y_test_K1, y_hatR1kv))/mean_GC)*100
            
            print('RMSE RF Traindata')
            print(rf_rmse_train)
            df_metrics_train_2_.iat[j[6],2]=rf_rmse_train
            
            print('RMSE % RF HT Traindata')
            print((root_mean_squared_error(Y_train_K1, y_hatR1kv_train)/mean_GC)*100)
            rmse_perc_train_RF_1=(root_mean_squared_error(Y_train_K1, y_hatR1kv_train)/mean_GC)*100
            df_metrics_train_2_.iat[j[6],3]=(root_mean_squared_error(Y_train_K1, y_hatR1kv_train)/mean_GC)*100
            
            print('RF')
            print('RF Score:'+str(m1Rkv.score(X_test_K1, Y_test_K1)))
            print('RF')
            print('RF Score Trainingsdaten:'+str(m1Rkv.score(X_train_K1, Y_train_K1)))
            df_metrics_test_2_.iat[j[6],6]=m1Rkv.score(X_train_K1, Y_train_K1)
            
            
            print('RF Accuracy Score:'+str(accuracy_score(Y_test_K1, y_hatR1kv)))
            RF_accuracy_test=accuracy_score(Y_test_K1, y_hatR1kv)
            print('RF Accuracy Score Trainingsdaten:'+str(accuracy_score(Y_train_K1, y_hatR1kv_train)))
            #print('Average Precision score:'+str(average_precision_score(Y_test_K1,y_hatk)))
           # print('Balanced Accuracy Score:'+str(balanced_accuracy_score(Y_test_K1,y_hatk).reshape(1,-1)))
            RF_accuracy_score_train=  accuracy_score(Y_train_K1, y_hatR1kv_train)
            print('RF MAE:'+ str(mean_absolute_error(Y_test_K1, y_hatR1kv)))
            print('RF MAE Trainingsdaten:'+ str(mean_absolute_error(Y_train_K1, y_hatR1kv_train)))
            
            print('RF MSE:'+str(mean_squared_error(Y_test_K1,y_hatR1kv)))
            df_metrics_test_2_.iat[j[6],1]=mean_squared_error(Y_test_K1,y_hatR1kv)
            r2RFtest=r2_score(Y_test_K1,y_hatR1kv)#Y_train_K1, y_hatR1kv_train
            print('RF MSE Trainingsdaten:'+str(mean_squared_error(Y_train_K1, y_hatR1kv_train)))
            df_metrics_train_2_.iat[j[6],1]=mean_squared_error(Y_train_K1, y_hatR1kv_train)
            r2RF_train=r2_score(Y_test_K1,y_hatR1kv)
            r2RF_train=r2_score(Y_train_K1, y_hatR1kv_train)
            print('RF R2:'+str(r2RFtest))
            df_metrics_test_2_.iat[j[6],0]=r2RFtest
            
            ######################################################################
            
            print('RF Cohens Kappa score:'+str(cohen_kappa_score(Y_test_K1,y_hatR1kv)))
            df_metrics_test_2_.iat[j[6],5]=cohen_kappa_score(Y_test_K1,y_hatR1kv)
            print('RF R2 Trainingsdaten:'+str(r2RF_train))
            print('RF Cohens Kappa score Trainingsdaten:'+str(cohen_kappa_score(Y_train_K1, y_hatR1kv_train)))
            df_metrics_train_2_.iat[j[6],5]=cohen_kappa_score(Y_train_K1, y_hatR1kv_train)
            
            
            print('RF HT HT Accuracy Score Testdata')
            print(accuracy_score(Y_test_K1, y_hatR1kv))
            df_metrics_test_2_.iat[j[6],6]=accuracy_score(Y_test_K1, y_hatR1kv)
            print('RF HT Balanced Accuracy Score Testdata')
            print(balanced_accuracy_score(Y_test_K1, y_hatR1kv))
            df_metrics_test_2_.iat[j[6],7]=balanced_accuracy_score(Y_test_K1, y_hatR1kv)
            print('RF HT Accuracy Score Traindata')
            print(accuracy_score(Y_train_K1, y_hatR1kv_train))
            df_metrics_train_2_.iat[j[6],6]=accuracy_score(Y_train_K1, y_hatR1kv_train)
            print('RF HT Balanced Accuracy Score Traindata')
            print(balanced_accuracy_score(Y_train_K1, y_hatR1kv_train))
            df_metrics_train_2_.iat[j[6],7]=balanced_accuracy_score(Y_train_K1, y_hatR1kv_train)
            # print('ROC of Testdata')
            # print(roc_auc_score(Y_test_K1,y_hatk))
            # print('ROC of Traindata')
            # print(roc_auc_score(Y_train_K1, y_hatk_train))
            
            print('RF HT Precision Score Testdata')
            print(precision_score(Y_test_K1, y_hatR1kv, average='micro'))
            df_metrics_test_2_.iat[j[6],9]=precision_score(Y_test_K1, y_hatR1kv, average='micro')
            
            print('RF HT Precision Score Traindata')
            print(precision_score(Y_train_K1, y_hatR1kv_train, average='micro'))
            df_metrics_train_2_.iat[j[6],9]=precision_score(Y_train_K1, y_hatR1kv_train, average='micro')
            print('RF HT F1 Score macro Testdata')
            print(f1_score(Y_test_K1, y_hatR1kv, average='macro'))
            #############################################################################
            
            RF_f1_score_train=f1_score(Y_train_K1, y_hatR1kv_train, average='weighted')
            
            RF_f1_score_test=f1_score(Y_test_K1, y_hatR1kv, average='weighted')
            ##################################################################################
            
            #################################################################################
            print('SVM HT F1 Score- macro Traindata')
            print(f1_score(Y_train_K1, y_hatR1kv_train, average='macro'))
            
            print('RF HT F1 Score- micro Testdata')
            print(f1_score(Y_test_K1, y_hatR1kv, average='micro'))
        
            print('RF HT F1 Score micro Traindata')
            print(f1_score(Y_train_K1, y_hatR1kv_train, average='micro'))
            
            print('RF HT F1 Score- weighted Testdata')
            print(f1_score(Y_test_K1, y_hatR1kv, average='weighted'))
        
            print('RF HT F1 Score weighted Traindata')
            print(f1_score(Y_train_K1, y_hatR1kv_train, average='weighted'))
            
            
            
            print('RF HT F1 Score- samples Testdata')
            print(f1_score(Y_test_K1, y_hatR1kv, average='weighted'))
            
            ###################
            #############
            ###########################################################################
            RF_accuracy_test
            
            F1_macro_test_RF=f1_score(Y_test_K1, y_hatR1kv, average='macro')
            
            F1_weighted_test_RF=f1_score(Y_test_K1, y_hatR1kv, average='weighted')
            
            F1_micro_test_RF=f1_score(Y_test_K1, y_hatR1kv, average='micro')
            
            df_metrics_test_F_scores_.iat[j[6],0]=RF_accuracy_test
            df_metrics_test_F_scores_.iat[j[6],2]=F1_macro_test_RF
            df_metrics_test_F_scores_.iat[j[6],4]=F1_weighted_test_RF
            df_metrics_test_F_scores_.iat[j[6],6]=F1_micro_test_RF
            
            print('RF HT F1 Score samples Traindata')
            print(f1_score(Y_train_K1, y_hatR1kv_train, average='weighted'))
            RF_accuracy_score_train
            
            F1_macro_train_RF=f1_score(Y_train_K1, y_hatR1kv_train, average='macro')
            
            F1_weighted_train_RF=f1_score(Y_test_K1, y_hatR1kv, average='weighted')
            
            F1_micro_train_RF=f1_score(Y_test_K1, y_hatR1kv, average='micro')
            
            df_metrics_test_F_scores_.iat[j[6],1]=RF_accuracy_score_train
            df_metrics_test_F_scores_.iat[j[6],3]=F1_macro_train_RF
            df_metrics_test_F_scores_.iat[j[6],5]=F1_weighted_train_RF
            df_metrics_test_F_scores_.iat[j[6],7]=F1_micro_train_RF
            
            # print('RF 1 HT ROC_AUC_Test ')
            # SVM_roc_auc_test=roc_auc_score_multiclass(Y_test_K1, y_hatR1kv)
            # print('RF 1 HT ROC_AUC_Train ')
            # SVM_roc_auc_train=roc_auc_score_multiclass(Y_train_K1, y_hatR1kv_train)
            
            print('RF 1 HT Matthews Corrcoef Testdaten ')
            matthews_corrcoef(Y_test_K1, y_hatR1kv)
            
            print('RF 1 HT Matthews Corrcoef Traindaten ')
            matthews_corrcoef(Y_train_K1, y_hatR1kv_train)
            
            
            #########################################################################################
            
            
            
            dict_metrics_['rmse_RF_test_'+str(j[2])]=r2RFtest
            dict_metrics_['rmse_RF_'+str(j[2])+'_train_accuracy']=r2RF_train
            dict_metrics_['rmse_RF_'+str(j[2])+'_test_accuracy']=RF_accuracy_test
            dict_metrics_['rmse_RF_'+str(j[2])+'_train_accuracy']=RF_accuracy_score_train
            dict_metrics_['rmse_RF_'+str(j[2])+'_test_rmse_percent']=rmse_perc_test_RF_1
            dict_metrics_['rmse_RF_'+str(j[2])+'_train_rmse_percent']=rmse_perc_train_RF_1
            
            
            ####################Hier Test des Erzeugens der Confuksion- Matrix
            #################
            
            matrix_RF=confusion_matrix(Y_test_K1,y_hatR1kv)
            matrix_df=pd.DataFrame(matrix_RF, columns=['0 W P', '1 W P', '3W P', '5W P', '7 W P', '9 W P'])
            ###matrix to dataframe
            ####
            sumpa=matrix_df.sum(axis=0)
            sumpa=sumpa.tolist()
            print(matrix_df)
            sumpa.append(1.0)
            print(len(sumpa))
            print(matrix_df.sum(axis=0))
            sumna=matrix_df.sum(axis=1)
            print(matrix_df.sum(axis=1))
            ###Hier in Confusion matrix einfügen
            matrix_df['Sum']=sumna
            
            matrix_df.loc[len(matrix_df.index)]=sumpa
            
            ###Useraccuracy
            UseraccL=[]
            Useracc1=(matrix_df.iat[0,0]/matrix_df.iat[0,6])*100
            print(Useracc1)
            print(matrix_df.iat[0,0])
            print(matrix_df.iat[0,6])
            UseraccL.append(Useracc1)
            Useracc2=(matrix_df.iat[1,1]/matrix_df.iat[1,6])*100
            UseraccL.append(Useracc2)
            Useracc3=(matrix_df.iat[2,2]/matrix_df.iat[2,6])*100
            UseraccL.append(Useracc3)
            Useracc4=(matrix_df.iat[3,3]/matrix_df.iat[3,6])*100
            UseraccL.append(Useracc4)
            Useracc5=(matrix_df.iat[4,4]/matrix_df.iat[4,6])*100
            UseraccL.append(Useracc5)
            Useracc6=(matrix_df.iat[5,5]/matrix_df.iat[5,6])*100
            UseraccL.append(Useracc6)
            UseraccL.append('-')
            matrix_df['UA']=UseraccL
            
            ###
            
            
            ####Producer Accuraccy
            
            ProduceraccL=[]
            PAacc1=(matrix_df.iat[0,0]/matrix_df.iat[6,0])*100
            print(PAacc1)
            ProduceraccL.append(PAacc1)
            PAacc2=(matrix_df.iat[1,1]/matrix_df.iat[6,1])*100
            print(matrix_df.iat[1,1])
            print(matrix_df.iat[6,1])
            ProduceraccL.append(PAacc2)
            PAacc3=(matrix_df.iat[2,2]/matrix_df.iat[6,2])*100
            ProduceraccL.append(PAacc3)
            PAacc4=(matrix_df.iat[3,3]/matrix_df.iat[6,3])*100
            ProduceraccL.append(PAacc4)
            PAacc5=(matrix_df.iat[4,4]/matrix_df.iat[6,4])*100
            ProduceraccL.append(PAacc5)
            PAacc6=(matrix_df.iat[5,5]/matrix_df.iat[6,5])*100
            ProduceraccL.append(PAacc6)
            ProduceraccL.append('-')
            ProduceraccL.append('-')    
            
            matrix_df.loc[len(matrix_df.index)]=ProduceraccL
            
            ### Overall accuracy add
            
            matrix_df.iat[6,7]='Overall Accuracy/F-1 '
            matrix_df.iat[7,7]=str(np.round(RF_accuracy_test,2))+'/ '+str(np.round(RF_f1_score_train,2))
            
            matrix_df.iat[7,6]=str('RF 1- Test')
            
            matrix_df.index=['0 W L','1 WL', '3 W L', '5W L', '7 WL', '9 WL', 'Sum', 'PA' ]
            
        
           
            matrix_RF.diagonal()/matrix_RF.sum(axis=1)
            cm_1=matrix_RF.astype('float')/matrix_RF.sum(axis=1)[:, np.newaxis]
            #cm_1=matrix_SVM_train.astype('float')/matrix_SVM_train.sum(axis=1)[:, np.newaxis]
            ##Confusion Matrix List
            CL_list=['0','1','3','5','7','9']
        
            cm_1_df=pd.DataFrame.from_records(cm_1)
            cm_1_df.rename(columns={0:'0_W', 1:'1_W',2:'3_W', 3:'W_5', 4:'W_7', 5:'W_9'}, inplace=True)
            cm_1_df.index=list(CL_list)
            ################
            ##Calculate User and Produser Accuraccy
            #############################################
            sumpa=cm_1.sum(axis=0)
            sumpa=sumpa.tolist()
            print(sumpa)
            sumpa.append(1.0)
            print(len(sumpa))
            print(cm_1.sum(axis=0))
            sumna=cm_1.sum(axis=1)
            print(cm_1.sum(axis=1))
            ###Hier in Confusion matrix einfügen
            cm_1_df['Sum']=sumna
            
            cm_1_df.loc[len(cm_1_df.index)]=sumpa
            
            
            ###Useraccuracy
            UseraccL=[]
            Useracc1=(cm_1_df.iat[0,0]/cm_1_df.iat[0,6])*100
            print(Useracc1)
            print(cm_1_df.iat[0,0])
            print(cm_1_df.iat[0,6])
            UseraccL.append(Useracc1)
            Useracc2=(cm_1_df.iat[1,1]/cm_1_df.iat[1,6])*100
            UseraccL.append(Useracc2)
            Useracc3=(cm_1_df.iat[2,2]/cm_1_df.iat[2,6])*100
            UseraccL.append(Useracc3)
            Useracc4=(cm_1_df.iat[3,3]/cm_1_df.iat[3,6])*100
            UseraccL.append(Useracc4)
            Useracc5=(cm_1_df.iat[4,4]/cm_1_df.iat[4,6])*100
            UseraccL.append(Useracc5)
            Useracc6=(cm_1_df.iat[5,5]/cm_1_df.iat[5,6])*100
            UseraccL.append(Useracc6)
            UseraccL.append('-')
            cm_1_df['UA']=UseraccL
            
            ###
            
            ####Producer Accuraccy
            
            ProduceraccL=[]
            PAacc1=(cm_1_df.iat[0,0]/cm_1_df.iat[6,0])*100
            print(PAacc1)
            ProduceraccL.append(PAacc1)
            PAacc2=(cm_1_df.iat[1,1]/cm_1_df.iat[6,1])*100
            print(cm_1_df.iat[1,1])
            print(cm_1_df.iat[6,1])
            ProduceraccL.append(PAacc2)
            PAacc3=(cm_1_df.iat[2,2]/cm_1_df.iat[6,2])*100
            ProduceraccL.append(PAacc3)
            PAacc4=(cm_1_df.iat[3,3]/cm_1_df.iat[6,3])*100
            ProduceraccL.append(PAacc4)
            PAacc5=(cm_1_df.iat[4,4]/cm_1_df.iat[6,4])*100
            ProduceraccL.append(PAacc5)
            PAacc6=(cm_1_df.iat[5,5]/cm_1_df.iat[6,5])*100
            ProduceraccL.append(PAacc6)
            ProduceraccL.append('-')
            ProduceraccL.append('-')    
            
            cm_1_df.loc[len(cm_1_df.index)]=ProduceraccL
            
            ### Overall accuracy add
            
            cm_1_df.iat[6,7]='Overall Accuracy/F-1 '
            cm_1_df.iat[6,7]='SVM- Test 2'
            cm_1_df.iat[7,7]=str(np.round(RF_accuracy_test,2))+'/ '+str(np.round(f_1_score_macro_train,2))
            cm_1_df.iat[7,7]=str(np.round(RF_accuracy_test,2))+'/ '+str(np.round(f_1_score_macro_train,2))
            cm_1_df.index=['0 W L','1 WL', '3 W L', '5W L', '7 WL', '9 WL', 'Sum', 'PA' ]
            
            
            
            # cm_1_df.to_excel('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1_prob.xlsx')
            # cm_1_df.to_csv('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1_prob.csv')
           
            
           
            cm1_test=cm_1_df
            matrix_df_test=matrix_df
            #'+str(count_list_2[i])+'
            
            dict_metrics_.update({'RF_'+str(j[2])+'_metrics': [rmse_svm_1_test,rmse_percent_SVM_1_test, accuracy_svm_test_1, cm1_test, matrix_df_test]}) 
            
            
            # confusion_matrix_L=[]
            # confusionmatrix_L_prob=[]
            confusion_matrix_L.append(matrix_df)
            confusionmatrix_L_prob.append(cm_1_df)
            
            ###########################
            ################################################################
            ####
            matrix_SVM_train=confusion_matrix(Y_train_K1,y_hatk_train, )
            ###matrix_SVM_train_switch=confusion_matrix(Y_train_K1, y_hatk_train,)
            matrix_df=pd.DataFrame(matrix_SVM_train, columns=['0 W P', '1 W P', '3W P', '5W P', '7 W P', '9 W P'])
            ###matrix to dataframe
            ####
            sumpa=matrix_df.sum(axis=0)
            sumpa=sumpa.tolist()
            print(matrix_df)
            sumpa.append(1.0)
            print(len(sumpa))
            print(matrix_df.sum(axis=0))
            sumna=matrix_df.sum(axis=1)
            print(matrix_df.sum(axis=1))
            ###Hier in Confusion matrix einfügen
            matrix_df['Sum']=sumna
            
            matrix_df.loc[len(matrix_df.index)]=sumpa
            
            ###Useraccuracy
            UseraccL=[]
            Useracc1=(matrix_df.iat[0,0]/matrix_df.iat[0,6])*100
            print(Useracc1)
            print(matrix_df.iat[0,0])
            print(matrix_df.iat[0,6])
            UseraccL.append(Useracc1)
            Useracc2=(matrix_df.iat[1,1]/matrix_df.iat[1,6])*100
            UseraccL.append(Useracc2)
            Useracc3=(matrix_df.iat[2,2]/matrix_df.iat[2,6])*100
            UseraccL.append(Useracc3)
            Useracc4=(matrix_df.iat[3,3]/matrix_df.iat[3,6])*100
            UseraccL.append(Useracc4)
            Useracc5=(matrix_df.iat[4,4]/matrix_df.iat[4,6])*100
            UseraccL.append(Useracc5)
            Useracc6=(matrix_df.iat[5,5]/matrix_df.iat[5,6])*100
            UseraccL.append(Useracc6)
            UseraccL.append('-')
            matrix_df['UA']=UseraccL
                        
            
            ####Producer Accuraccy
            
            ProduceraccL=[]
            PAacc1=(matrix_df.iat[0,0]/matrix_df.iat[6,0])*100
            print(PAacc1)
            ProduceraccL.append(PAacc1)
            PAacc2=(matrix_df.iat[1,1]/matrix_df.iat[6,1])*100
            print(matrix_df.iat[1,1])
            print(matrix_df.iat[6,1])
            ProduceraccL.append(PAacc2)
            PAacc3=(matrix_df.iat[2,2]/matrix_df.iat[6,2])*100
            ProduceraccL.append(PAacc3)
            PAacc4=(matrix_df.iat[3,3]/matrix_df.iat[6,3])*100
            ProduceraccL.append(PAacc4)
            PAacc5=(matrix_df.iat[4,4]/matrix_df.iat[6,4])*100
            ProduceraccL.append(PAacc5)
            PAacc6=(matrix_df.iat[5,5]/matrix_df.iat[6,5])*100
            ProduceraccL.append(PAacc6)
            ProduceraccL.append('-')
            ProduceraccL.append('-')    
            
            matrix_df.loc[len(matrix_df.index)]=ProduceraccL
            
            ### Overall accuracy add
            
            matrix_df.iat[6,7]='Overall Accuracy/F-1 '
            matrix_df.iat[7,7]=str(np.round(accuraccy_svm_train,2))+'/ '+str(np.round(f_1_score_macro_train,2))
            matrix_df.iat[7,6]=str('RF-1- Train')
            matrix_df.index=['0 W L','1 WL', '3 W L', '5W L', '7 WL', '9 WL', 'Sum', 'PA' ]
            
            
            
            # matrix_df.to_excel('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1_train.xlsx')
            # matrix_df.to_csv('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1_train.csv')
           
            
           
            matrix_SVM_train.diagonal()/matrix_SVM_train.sum(axis=1)
            cm_1=matrix_SVM_train.astype('float')/matrix_SVM_train.sum(axis=1)[:, np.newaxis]
            
                
            ##Confusion Matrix List
            CL_list=['0','1','3','5','7','9']
        
            cm_1_df=pd.DataFrame.from_records(cm_1)
            cm_1_df.rename(columns={0:'0_W', 1:'1_W',2:'3_W', 3:'W_5', 4:'W_7', 5:'W_9'}, inplace=True)
            cm_1_df.index=list(CL_list)
            ################
            ##Calculate User and Produser Accuraccy
            #############################################
            sumpa=cm_1.sum(axis=0)
            sumpa=sumpa.tolist()
            print(sumpa)
            sumpa.append(1.0)
            print(len(sumpa))
            print(cm_1.sum(axis=0))
            sumna=cm_1.sum(axis=1)
            print(cm_1.sum(axis=1))
            ###Hier in Confusion matrix einfügen
            cm_1_df['Sum']=sumna
            
            cm_1_df.loc[len(cm_1_df.index)]=sumpa
            
            
            ###Useraccuracy
            UseraccL=[]
            Useracc1=(cm_1_df.iat[0,0]/cm_1_df.iat[0,6])*100
            print(Useracc1)
            print(cm_1_df.iat[0,0])
            print(cm_1_df.iat[0,6])
            UseraccL.append(Useracc1)
            Useracc2=(cm_1_df.iat[1,1]/cm_1_df.iat[1,6])*100
            UseraccL.append(Useracc2)
            Useracc3=(cm_1_df.iat[2,2]/cm_1_df.iat[2,6])*100
            UseraccL.append(Useracc3)
            Useracc4=(cm_1_df.iat[3,3]/cm_1_df.iat[3,6])*100
            UseraccL.append(Useracc4)
            Useracc5=(cm_1_df.iat[4,4]/cm_1_df.iat[4,6])*100
            UseraccL.append(Useracc5)
            Useracc6=(cm_1_df.iat[5,5]/cm_1_df.iat[5,6])*100
            UseraccL.append(Useracc6)
            UseraccL.append('-')
            cm_1_df['UA']=UseraccL
            
            
            ####Producer Accuraccy
            
            ProduceraccL=[]
            PAacc1=(cm_1_df.iat[0,0]/cm_1_df.iat[6,0])*100
            print(PAacc1)
            ProduceraccL.append(PAacc1)
            PAacc2=(cm_1_df.iat[1,1]/cm_1_df.iat[6,1])*100
            print(cm_1_df.iat[1,1])
            print(cm_1_df.iat[6,1])
            ProduceraccL.append(PAacc2)
            PAacc3=(cm_1_df.iat[2,2]/cm_1_df.iat[6,2])*100
            ProduceraccL.append(PAacc3)
            PAacc4=(cm_1_df.iat[3,3]/cm_1_df.iat[6,3])*100
            ProduceraccL.append(PAacc4)
            PAacc5=(cm_1_df.iat[4,4]/cm_1_df.iat[6,4])*100
            ProduceraccL.append(PAacc5)
            PAacc6=(cm_1_df.iat[5,5]/cm_1_df.iat[6,5])*100
            ProduceraccL.append(PAacc6)
            ProduceraccL.append('-')
            ProduceraccL.append('-')    
            
            cm_1_df.loc[len(cm_1_df.index)]=ProduceraccL
            
            ### add Overall accuracy
            
            cm_1_df.iat[6,7]='Overall Accuracy/F-1 '
            cm_1_df.iat[7,7]=str(np.round(accuracy_svm,2))+'/ '+str(np.round(f_1_score_macro_train,2))
            
            cm_1_df.index=['0 W L','1 WL', '3 W L', '5W L', '7 WL', '9 WL', 'Sum', 'PA' ]
            
            
            
            # cm_1_df.to_excel('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1_prob_train.xlsx')
            # cm_1_df.to_csv('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1_prob_train.csv')
           
            
            # confusion_matrix_L=[]
            # confusionmatrix_L_prob=[]
            confusion_matrix_L.append(matrix_df)
            confusionmatrix_L_prob.append(cm_1_df)
            
            
            dict_metrics_['rmse_RF_test_'+str(j[2])]=r2RFtest
            dict_metrics_['rmse_RF_'+str(j[2])+'_train_accuracy']=r2RF_train
            dict_metrics_['rmse_RF_'+str(j[2])+'_test_accuracy']=RF_accuracy_test
            dict_metrics_['rmse_RF_'+str(j[2])+'_train_accuracy']=RF_accuracy_score_train
            dict_metrics_['rmse_RF_'+str(j[2])+'_test_rmse_percent']=rmse_perc_test_RF_1
            dict_metrics_['rmse_RF_'+str(j[2])+'_train_rmse_percent']=rmse_perc_train_RF_1
            
            
            dict_metrics_.update({'rmse-RF_'+str(j[2])+'_test_1': [cm_1_df]})   
            dict_metrics_.update({'rmse-RF_'+str(j[2])+'_test_1': [matrix_df]}) 
            
            dict_metrics_['RF_'+str(j[2])+'_metrics']=[rmse_svm_1_test]
            dict_metrics_.update({'RF_'+str(j[2])+'_metrics':[rmse_percent_SVM_1_test]}) 
            #dict_metrics.update({'RF_1_metrics': [r2RFtest,rmse_perc_test_RF_1, RF_accuracy_test, cm_1_df,matrix_df]}) 
            
            #dict_metrics.update({'SVM_1_metrics': [rmse_svm_1_test,rmse_percent_SVM_1_test, accuracy_svm_test_1,cm1_test, cm_1_df,rmse_train_svm_1, rmse_percent_SVM_1_train,accuracy_SVM_1_train, matrix_df_test, matrix_df]})
            
            
            dict_metrics_.update({'RF_'+str(j[2])+'_metrics': [rmse_svm_1_test,rmse_perc_test_RF_1, RF_accuracy_test,cm1_test, cm_1_df,rmse_train_svm_1, rmse_perc_train_RF_1,RF_accuracy_score_train, matrix_df_test, matrix_df]})
            
            
            
            cfkvrf=confusion_matrix(Y_test_K1,y_hatR1kv)
            
            # matrix_SVM=confusion_matrix(y_pred_test_SVM_1, Y_1_1)
            cfkvrf.diagonal()/cfkvrf.sum(axis=1)
            cm_1=cfkvrf.astype('float')/cfkvrf.sum(axis=1)[:, np.newaxis]
            
  
            dispkvrf=ConfusionMatrixDisplay(confusion_matrix=cfkvrf, display_labels=m1Rkv.classes_)
            #disp2=ConfusionMatrixDisplay(confusion_matrix=cf2, display_labels=model2.classes_)
            dispkvrf.plot()
            
            #############################
            
            matrix_RF=confusion_matrix(Y_test_K1,y_hatR1kv)
            matrix_df=pd.DataFrame(matrix_RF, columns=['0 W P', '1 W P', '3W P', '5W P', '7 W P', '9 W P'])
            ###matrix to dataframe
            ####
            sumpa=matrix_df.sum(axis=0)
            sumpa=sumpa.tolist()
            print(matrix_df)
            sumpa.append(1.0)
            print(len(sumpa))
            print(matrix_df.sum(axis=0))
            sumna=matrix_df.sum(axis=1)
            print(matrix_df.sum(axis=1))
            ###Hier in Confusion matrix einfügen
            matrix_df['Sum']=sumna
            
            matrix_df.loc[len(matrix_df.index)]=sumpa
            
            ###Useraccuracy
            UseraccL=[]
            Useracc1=(matrix_df.iat[0,0]/matrix_df.iat[0,6])*100
            print(Useracc1)
            print(matrix_df.iat[0,0])
            print(matrix_df.iat[0,6])
            UseraccL.append(Useracc1)
            Useracc2=(matrix_df.iat[1,1]/matrix_df.iat[1,6])*100
            UseraccL.append(Useracc2)
            Useracc3=(matrix_df.iat[2,2]/matrix_df.iat[2,6])*100
            UseraccL.append(Useracc3)
            Useracc4=(matrix_df.iat[3,3]/matrix_df.iat[3,6])*100
            UseraccL.append(Useracc4)
            Useracc5=(matrix_df.iat[4,4]/matrix_df.iat[4,6])*100
            UseraccL.append(Useracc5)
            Useracc6=(matrix_df.iat[5,5]/matrix_df.iat[5,6])*100
            UseraccL.append(Useracc6)
            UseraccL.append('-')
            matrix_df['UA']=UseraccL
            

            ####Producer Accuraccy
            
            ProduceraccL=[]
            PAacc1=(matrix_df.iat[0,0]/matrix_df.iat[6,0])*100
            print(PAacc1)
            ProduceraccL.append(PAacc1)
            PAacc2=(matrix_df.iat[1,1]/matrix_df.iat[6,1])*100
            print(matrix_df.iat[1,1])
            print(matrix_df.iat[6,1])
            ProduceraccL.append(PAacc2)
            PAacc3=(matrix_df.iat[2,2]/matrix_df.iat[6,2])*100
            ProduceraccL.append(PAacc3)
            PAacc4=(matrix_df.iat[3,3]/matrix_df.iat[6,3])*100
            ProduceraccL.append(PAacc4)
            PAacc5=(matrix_df.iat[4,4]/matrix_df.iat[6,4])*100
            ProduceraccL.append(PAacc5)
            PAacc6=(matrix_df.iat[5,5]/matrix_df.iat[6,5])*100
            ProduceraccL.append(PAacc6)
            ProduceraccL.append('-')
            ProduceraccL.append('-')    
            
            matrix_df.loc[len(matrix_df.index)]=ProduceraccL
            
            ### Overall accuracy add
            
            matrix_df.iat[6,7]='Overall Accuracy/F-1 '
            matrix_df.iat[7,7]=str(np.round(accuracy_svm,2))+'/ '+str(np.round(f_1_score_macro_train,2))
            
            matrix_df.index=['0 W L','1 WL', '3 W L', '5W L', '7 WL', '9 WL', 'Sum', 'PA' ]
            
            
            
            # matrix_df.to_excel('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1.xlsx')
            # matrix_df.to_csv('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1.csv')
           
           
            matrix_SVM.diagonal()/matrix_SVM.sum(axis=1)
            cm_1=matrix_SVM.astype('float')/matrix_SVM.sum(axis=1)[:, np.newaxis]
            ##Confusion Matrix List
            CL_list=['0','1','3','5','7','9']
        
            cm_1_df=pd.DataFrame.from_records(cm_1)
            cm_1_df.rename(columns={0:'0_W', 1:'1_W',2:'3_W', 3:'W_5', 4:'W_7', 5:'W_9'}, inplace=True)
            cm_1_df.index=list(CL_list)
            ################
            ##Calculate User and Produser Accuraccy
            #############################################
            sumpa=cm_1.sum(axis=0)
            sumpa=sumpa.tolist()
            print(sumpa)
            sumpa.append(1.0)
            print(len(sumpa))
            print(cm_1.sum(axis=0))
            sumna=cm_1.sum(axis=1)
            print(cm_1.sum(axis=1))
            ###Hier in Confusion matrix einfügen
            cm_1_df['Sum']=sumna
            
            cm_1_df.loc[len(cm_1_df.index)]=sumpa
            
            #cm_1_df.at['6']=sumpa
            
            ###Useraccuracy
            UseraccL=[]
            Useracc1=(cm_1_df.iat[0,0]/cm_1_df.iat[0,6])*100
            print(Useracc1)
            print(cm_1_df.iat[0,0])
            print(cm_1_df.iat[0,6])
            UseraccL.append(Useracc1)
            Useracc2=(cm_1_df.iat[1,1]/cm_1_df.iat[1,6])*100
            UseraccL.append(Useracc2)
            Useracc3=(cm_1_df.iat[2,2]/cm_1_df.iat[2,6])*100
            UseraccL.append(Useracc3)
            Useracc4=(cm_1_df.iat[3,3]/cm_1_df.iat[3,6])*100
            UseraccL.append(Useracc4)
            Useracc5=(cm_1_df.iat[4,4]/cm_1_df.iat[4,6])*100
            UseraccL.append(Useracc5)
            Useracc6=(cm_1_df.iat[5,5]/cm_1_df.iat[5,6])*100
            UseraccL.append(Useracc6)
            UseraccL.append('-')
            cm_1_df['UA']=UseraccL
            
            ####Producer Accuraccy
            
            ProduceraccL=[]
            PAacc1=(cm_1_df.iat[0,0]/cm_1_df.iat[6,0])*100
            print(PAacc1)
            ProduceraccL.append(PAacc1)
            PAacc2=(cm_1_df.iat[1,1]/cm_1_df.iat[6,1])*100
            print(cm_1_df.iat[1,1])
            print(cm_1_df.iat[6,1])
            ProduceraccL.append(PAacc2)
            PAacc3=(cm_1_df.iat[2,2]/cm_1_df.iat[6,2])*100
            ProduceraccL.append(PAacc3)
            PAacc4=(cm_1_df.iat[3,3]/cm_1_df.iat[6,3])*100
            ProduceraccL.append(PAacc4)
            PAacc5=(cm_1_df.iat[4,4]/cm_1_df.iat[6,4])*100
            ProduceraccL.append(PAacc5)
            PAacc6=(cm_1_df.iat[5,5]/cm_1_df.iat[6,5])*100
            ProduceraccL.append(PAacc6)
            ProduceraccL.append('-')
            ProduceraccL.append('-')    
            
            cm_1_df.loc[len(cm_1_df.index)]=ProduceraccL
            
            ### Overall accuracy add
            
            cm_1_df.iat[6,7]='Overall Accuracy/F-1 '
            cm_1_df.iat[7,7]=str(np.round(accuracy_svm,2))+'/ '+str(np.round(f_1_score_macro_train,2))
            
            cm_1_df.index=['0 W L','1 WL', '3 W L', '5W L', '7 WL', '9 WL', 'Sum', 'PA' ]
            

            # cm_1_df.to_excel('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1_prob.xlsx')
            # cm_1_df.to_csv('C:/Users/ronal/OneDrive/Dokumente/confusion_matrix_output_NEW/Confusion_Matrix_SVM_1_prob.csv')
           
            
            # confusion_matrix_L=[]
            # confusionmatrix_L_prob=[]
            confusion_matrix_L.append(matrix_df)
            confusionmatrix_L_prob.append(cm_1_df)
            
            
            dict_metrics_.update({'rmse-RF_'+str(j[2])+'_test': [cm_1_df]})   
            dict_metrics_.update({'rmse-RF_'+str(j[2])+'_test': [matrix_df]}) 
       
    
    
            dic_List_3.append(dict_metrics_)


###graphische Darstellung der Ergebnisse / hat geklappt mit for loop


matplotlib.rc('figure', figsize=(8,5))

### include metrics from list for visualiazation of the data
####################################
####
##Here list the keys of the dictionary
###

metric_keys=list(dict_metrics_.keys())
##metrics
metric_keys

##metrics Reihenfolge im dictionary
#RMSE test, RMSE % Test, Accuracy Test, #RMSE train, RMSE % Train, Accuracy Train


####Train Test F 1 score comparison metrics
df_metrics_test_F_scores_.columns.tolist()

#############################################
dict_metrics_.get('RF_3_metrics')[0]
dict_metrics_.get('RF_3_metrics')[1]
dict_metrics_.get('RF_3_metrics')[2]

dict_metrics_.get('SVM_4_metrics')[0]





color=['coral','blue', 'orange', 'green', 'black','yellow','brown', 'lime', 'cyan', 'grey']


######################################################################
##############visulization of the metrics with bar chart diagrams


compare_metrics_F3=pd.DataFrame({'SVM_1':[dict_metrics_.get('SVM_1_metrics')[0],dict_metrics_.get('SVM_1_metrics')[2],dict_metrics_.get('SVM_1_metrics')[5],dict_metrics_.get('SVM_1_metrics')[7]],
                               'SVM_2':[dict_metrics_.get('SVM_2_metrics')[0],dict_metrics_.get('SVM_2_metrics')[2],dict_metrics_.get('SVM_2_metrics')[5],dict_metrics_.get('SVM_2_metrics')[7]],
                               'SVM_3':[dict_metrics_.get('SVM_3_metrics')[0],dict_metrics_.get('SVM_3_metrics')[2],dict_metrics_.get('SVM_3_metrics')[5],dict_metrics_.get('SVM_3_metrics')[7]],
                               'SVM_4':[dict_metrics_.get('SVM_4_metrics')[0],dict_metrics_.get('SVM_4_metrics')[2],dict_metrics_.get('SVM_4_metrics')[5],dict_metrics_.get('SVM_4_metrics')[7]],
                               'SVM_5':[dict_metrics_.get('SVM_5_metrics')[0],dict_metrics_.get('SVM_5_metrics')[2],dict_metrics_.get('SVM_5_metrics')[5],dict_metrics_.get('SVM_5_metrics')[7]],
                               'SVM_6':[dict_metrics_.get('SVM_6_metrics')[0],dict_metrics_.get('SVM_6_metrics')[2],dict_metrics_.get('SVM_6_metrics')[5],dict_metrics_.get('SVM_6_metrics')[7]],
                               'SVM_7':[dict_metrics_.get('SVM_7_metrics')[0],dict_metrics_.get('SVM_7_metrics')[2],dict_metrics_.get('SVM_7_metrics')[5],dict_metrics_.get('SVM_7_metrics')[7]],
                               'RF_1':[dict_metrics_.get('RF_1_metrics')[0], dict_metrics_.get('RF_1_metrics')[2], dict_metrics_.get('RF_1_metrics')[5], dict_metrics_.get('RF_1_metrics')[7]],
                               'RF_2':[dict_metrics_.get('RF_2_metrics')[0], dict_metrics_.get('RF_2_metrics')[2], dict_metrics_.get('RF_2_metrics')[5], dict_metrics_.get('RF_2_metrics')[7]],
                               'RF_3':[dict_metrics_.get('RF_3_metrics')[0], dict_metrics_.get('RF_3_metrics')[2], dict_metrics_.get('RF_3_metrics')[5], dict_metrics_.get('RF_3_metrics')[7]],
                               'RF_4':[dict_metrics_.get('RF_4_metrics')[0], dict_metrics_.get('RF_4_metrics')[2], dict_metrics_.get('RF_4_metrics')[5], dict_metrics_.get('RF_4_metrics')[7]],
                               'RF_5':[dict_metrics_.get('RF_5_metrics')[0], dict_metrics_.get('RF_5_metrics')[2], dict_metrics_.get('RF_5_metrics')[0], dict_metrics_.get('RF_5_metrics')[2]],
                               'RF_6':[dict_metrics_.get('RF_6_metrics')[0], dict_metrics_.get('RF_6_metrics')[2], dict_metrics_.get('RF_6_metrics')[0], dict_metrics_.get('RF_6_metrics')[2]],
                               'RF_7':[dict_metrics_.get('RF_7_metrics')[0], dict_metrics_.get('RF_7_metrics')[2], dict_metrics_.get('RF_7_metrics')[0], dict_metrics_.get('RF_7_metrics')[2]]
                               }, index=['RMSE Test','Accuracy Test', 'RMSE Train', 'Train Accuracy' ])


ax=compare_metrics_F3.plot.bar(colormap='Paired', width=0.7)

rmse_svm_1_test,rmse_train_svm_1

ax.set_xlabel('Model Classifier Input', fontsize=14, labelpad=14)
ax.set_ylabel('Repeated Cross Fold Validation Metrics', fontsize=14,labelpad=10)
ax.title.set_size(15)
ax.set_title('')
ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')
plt.show()


color=['coral','blue', 'orange', 'green', 'black','yellow','brown', 'lime', 'cyan', 'grey']

color=['black']*14

compare_metrics_F_x=pd.DataFrame({'SVM_1':[dict_metrics_.get('SVM_1_metrics')[0], dict_metrics_.get('SVM_1_metrics')[0]],
                                'SVM_2':[dict_metrics_.get('SVM_2_metrics')[0], dict_metrics_.get('SVM_2_metrics')[0]],
                                'SVM_3':[dict_metrics_.get('SVM_3_metrics')[0], dict_metrics_.get('SVM_3_metrics')[0]],
                                'SVM_4':[dict_metrics_.get('SVM_4_metrics')[1],dict_metrics_.get('SVM_4_metrics')[6]],
                                'SVM_5':[dict_metrics_.get('SVM_5_metrics')[1],dict_metrics_.get('SVM_5_metrics')[6]],
                                'SVM_6':[dict_metrics_.get('SVM_6_metrics')[1],dict_metrics_.get('SVM_6_metrics')[6]],
                                'SVM_7':[dict_metrics_.get('SVM_7_metrics')[1],dict_metrics_.get('SVM_7_metrics')[6]],
                                'RF_1':[dict_metrics_.get('RF_1_metrics')[1],dict_metrics_.get('RF_1_metrics')[6]],
                                'RF_2':[dict_metrics_.get('RF_2_metrics')[1],dict_metrics_.get('RF_2_metrics')[6]],
                                'RF_3':[dict_metrics_.get('RF_1_metrics')[1],dict_metrics_.get('RF_1_metrics')[6]],
                                'RF_4':[dict_metrics_.get('RF_4_metrics')[1], dict_metrics_.get('RF_4_metrics')[6]],
                                'RF_5':[dict_metrics_.get('RF_5_metrics')[1], dict_metrics_.get('RF_5_metrics')[6]],
                                'RF_6':[dict_metrics_.get('RF_6_metrics')[1], dict_metrics_.get('RF_6_metrics')[6]],
                                'RF_7':[dict_metrics_.get('RF_7_metrics')[1], dict_metrics_.get('RF_7_metrics')[6]]
                                }, index=['RMSE % percent Test','RMSE % Train'])
 
 
ax=compare_metrics_F_x.plot.bar(colormap='Paired', edgecolor='red')
 


ax.set_xlabel('Model Classifier Input', fontsize=14)
ax.set_ylabel('RMSE % Train and -Testdata fit', fontsize=14)
ax.title.set_size(15)
ax.set_title('')
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12, fontweight='bold', color='black')
ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')
plt.show()


#########################################################################
#define color palette with n times colors
colors=[]

compare_metrics_F_y=pd.DataFrame({'SVM_1':[dict_metrics_.get('SVM_1_metrics')[2]*100,dict_metrics_.get('SVM_1_metrics')[7]*100],
                           'SVM_2':[dict_metrics_.get('SVM_2_metrics')[2]*100,dict_metrics_.get('SVM_2_metrics')[7]*100],
                           'SVM_3':[dict_metrics_.get('SVM_3_metrics')[2]*100,dict_metrics_.get('SVM_3_metrics')[7]*100],
                           'SVM_4':[dict_metrics_.get('SVM_4_metrics')[2]*100,dict_metrics_.get('SVM_4_metrics')[7]*100],
                           'SVM_5':[dict_metrics_.get('SVM_5_metrics')[2]*100,dict_metrics_.get('SVM_5_metrics')[7]*100],
                           'SVM_6':[dict_metrics_.get('SVM_6_metrics')[2]*100,dict_metrics_.get('SVM_6_metrics')[7]*100],
                           'SVM_7':[dict_metrics_.get('SVM_7_metrics')[2]*100,dict_metrics_.get('SVM_7_metrics')[7]*100],
                           'RF_1':[dict_metrics_.get('RF_1_metrics')[2]*100,dict_metrics_.get('RF_1_metrics')[7]*100],
                           'RF_2':[dict_metrics_.get('RF_2_metrics')[2]*100,dict_metrics_.get('RF_2_metrics')[7]*100],
                           'RF_3':[dict_metrics_.get('RF_3_metrics')[2]*100,dict_metrics_.get('RF_3_metrics')[7]*100],
                           'RF_4':[dict_metrics_.get('RF_4_metrics')[2]*100,dict_metrics_.get('RF_4_metrics')[7]*100],
                           'RF_5':[dict_metrics_.get('RF_5_metrics')[2]*100,dict_metrics_.get('RF_5_metrics')[7]*100],
                           'RF_6':[dict_metrics_.get('RF_6_metrics')[2]*100,dict_metrics_.get('RF_6_metrics')[7]*100],
                           'RF_7':[dict_metrics_.get('RF_7_metrics')[2]*100,dict_metrics_.get('RF_7_metrics')[7]*100]
                           }, index=['Accuracy Test','Train Accuracy' ])

#[rf_3_rmse_test,rf_3_test_accuracy,rf_3_rmse_train,rf_3_train_accuracy]
ax=compare_metrics_F_y.plot.bar(colormap='Paired', width=0.7)

rmse_svm_1_test,rmse_train_svm_1

ax.set_xlabel('Model Classifier Input', fontsize=16, labelpad=14)
ax.set_ylabel('Overall Accuracy %', fontsize=16,labelpad=10)
ax.title.set_size(15)
ax.set_title('')
ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.legend(loc=2)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')
plt.show()




   
compare_metrics_F_z=pd.DataFrame({'SVM_1':[dict_metrics_.get('SVM_1_metrics')[0],dict_metrics_.get('SVM_1_metrics')[5]],
                           'SVM_2':[dict_metrics_.get('SVM_2_metrics')[0],dict_metrics_.get('SVM_2_metrics')[5]],
                           'SVM_3':[dict_metrics_.get('SVM_3_metrics')[0],dict_metrics_.get('SVM_3_metrics')[5]],
                           'SVM_4':[dict_metrics_.get('SVM_4_metrics')[0],dict_metrics_.get('SVM_4_metrics')[5]],
                           'SVM_5':[dict_metrics_.get('SVM_5_metrics')[0],dict_metrics_.get('SVM_5_metrics')[5]],
                           'RF_1':[dict_metrics_.get('RF_1_metrics')[0],dict_metrics_.get('RF_1_metrics')[5]],
                           'RF_2':[dict_metrics_.get('RF_2_metrics')[0],dict_metrics_.get('RF_2_metrics')[5]],
                           'RF_3':[dict_metrics_.get('RF_3_metrics')[0],dict_metrics_.get('RF_3_metrics')[5]],
                           'RF_4':[dict_metrics_.get('RF_4_metrics')[0],dict_metrics_.get('RF_4_metrics')[5]],
                           'RF_5':[dict_metrics_.get('RF_5_metrics')[0],dict_metrics_.get('RF_5_metrics')[5]]
                           }, index=['RMSE Test', 'RMSE Train'])


ax=compare_metrics_F_z.plot.bar(colormap='Paired', width=0.8)

rmse_svm_1_test,rmse_train_svm_1

ax.set_xlabel('Model Classifier Input', fontsize=14, labelpad=14)
ax.set_ylabel('RMSE (Mean)', fontsize=14,labelpad=10)
ax.title.set_size(15)
ax.set_title('')
ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')

ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.legend(loc=2)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')
plt.show()


color2=['lightskyblue', 'lightskyblue', 'royalblue', 'royalblue','palegreen', 'palegreen',
       'forestgreen', 'forestgreen', 'lightsalmon', 'lightsalmon', 'orangered', 'orangered',
       'bisque', 'bisque', 'orange', 'orange','slateblue', 'slateblue', 'blue', 'blue',
       'yellow', 'yellow', 'peru', 'peru','violet', 'violet', 'lightgray', 'lightgray']
print(len(color))
edgecolors2=['red', 'yellow']*14
print(edgecolors2)
 
compare_metrics_F_z_H=pd.DataFrame({'SVM 1 test':[dict_metrics_.get('SVM_1_metrics')[0]],
                                 'SVM 1 train':[ dict_metrics_.get('SVM_1_metrics')[5]],
                           'SVM 2 test':[dict_metrics_.get('SVM_2_metrics')[0]],
                           'SVM 2 train':[dict_metrics_.get('SVM_2_metrics')[5]],
                           'SVM 3 test':[dict_metrics_.get('SVM_3_metrics')[0]],
                           'SVM 3 train':[dict_metrics_.get('SVM_3_metrics')[5]],
                           'SVM 4 test':[dict_metrics_.get('SVM_4_metrics')[0]],
                           'SVM 4 train':[dict_metrics_.get('SVM_4_metrics')[5]],
                           'SVM 5 test':[dict_metrics_.get('SVM_5_metrics')[0]],
                           'SVM 5 train':[dict_metrics_.get('SVM_5_metrics')[5]],
                           'SVM 6 test':[dict_metrics_.get('SVM_6_metrics')[0]],
                           'SVM 6 train':[dict_metrics_.get('SVM_6_metrics')[5]],
                           'SVM 7 test':[dict_metrics_.get('SVM_7_metrics')[0]],
                           'SVM 7 train':[dict_metrics_.get('SVM_7_metrics')[5]],
                           'RF 1 test':[dict_metrics_.get('RF_1_metrics')[0]],
                           'RF 1 train':[dict_metrics_.get('RF_1_metrics')[5]],
                           'RF 2 test':[dict_metrics_.get('RF_2_metrics')[0]],
                           'RF 2 train':[dict_metrics_.get('RF_2_metrics')[5]],
                           'RF 3 test':[dict_metrics_.get('RF_3_metrics')[0]],
                           'RF 3 train':[dict_metrics_.get('RF_3_metrics')[5]],
                           'RF 4 test':[dict_metrics_.get('RF_4_metrics')[0]],
                           'RF 4 train':dict_metrics_.get('RF_4_metrics')[5],
                           'RF 5 test':[dict_metrics_.get('RF_5_metrics')[0]],
                           'RF 5 train':[dict_metrics_.get('RF_5_metrics')[5]],
                           'RF 6 test':[dict_metrics_.get('RF_6_metrics')[0]],
                           'RF 6 train':[dict_metrics_.get('RF_6_metrics')[5]],
                           'RF 7 test':[dict_metrics_.get('RF_7_metrics')[0]],
                           'RF 7 train':[dict_metrics_.get('RF_7_metrics')[5]]
                           }, index=['RMSE'])


ax=compare_metrics_F_z_H.plot.bar(color=color2, width=1.25, edgecolor=edgecolors2)

rmse_svm_1_test,rmse_train_svm_1

ax.set_xlabel('Model Classifier Input', fontsize=14, labelpad=14)
ax.set_ylabel('RMSE (Mean)', fontsize=14,labelpad=10)
ax.title.set_size(15)
ax.set_title('')
ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')

ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
# plt.legend(['SVM 1 test','SVM 1 train', 'SVM 2 test','SVM 2 train',
#            'SVM 3 test','SVM 3 train', 'SVM 4 test','SVM 4 train',
#            'SVM 5 test','SVM 5 train', 'SVM 6 test','SVM 6 train',
#            'SVM 7 test','SVM 7 train','RF 1 test','RF 1 train',
#            'RF 2 test','RF 2 train',
#            'RF 3 test','RF 3 train', 'RF 4 test','RF 4 train',
#            'RF 5 test','RF 5 train', 'RF 6 test','RF 6 train',
#            'RF 7 test','RF 7 train'])
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.legend(loc=1, prop={'size':8}, bbox_to_anchor=(1.15,1.00))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')
plt.show()
    
    
    
color2=['lightskyblue', 'lightskyblue', 'royalblue', 'royalblue','palegreen', 'palegreen',
       'forestgreen', 'forestgreen', 'lightsalmon', 'lightsalmon', 'orangered', 'orangered',
       'bisque', 'bisque', 'orange', 'orange','slateblue', 'slateblue', 'blue', 'blue',
       'yellow', 'yellow', 'peru', 'peru','violet', 'violet', 'lightgray', 'lightgray']
print(len(color))
edgecolors2=['red', 'yellow']*14
print(edgecolors2)
 
compare_metrics_F_z_H=pd.DataFrame({'SVM 1 test':[dict_metrics_.get('SVM_1_metrics')[1]],
                                 'SVM 1 train':[ dict_metrics_.get('SVM_1_metrics')[6]],
                           'SVM 2 test':[dict_metrics_.get('SVM_2_metrics')[1]],
                           'SVM 2 train':[dict_metrics_.get('SVM_2_metrics')[6]],
                           'SVM 3 test':[dict_metrics_.get('SVM_3_metrics')[1]],
                           'SVM 3 train':[dict_metrics_.get('SVM_3_metrics')[6]],
                           'SVM 4 test':[dict_metrics_.get('SVM_4_metrics')[1]],
                           'SVM 4 train':[dict_metrics_.get('SVM_4_metrics')[6]],
                           'SVM 5 test':[dict_metrics_.get('SVM_5_metrics')[1]],
                           'SVM 5 train':[dict_metrics_.get('SVM_5_metrics')[6]],
                           'SVM 6 test':[dict_metrics_.get('SVM_6_metrics')[1]],
                           'SVM 6 train':[dict_metrics_.get('SVM_6_metrics')[6]],
                           'SVM 7 test':[dict_metrics_.get('SVM_7_metrics')[1]],
                           'SVM 7 train':[dict_metrics_.get('SVM_7_metrics')[6]],
                           'RF 1 test':[dict_metrics_.get('RF_1_metrics')[1]],
                           'RF 1 train':[dict_metrics_.get('RF_1_metrics')[6]],
                           'RF 2 test':[dict_metrics_.get('RF_2_metrics')[1]],
                           'RF 2 train':[dict_metrics_.get('RF_2_metrics')[6]],
                           'RF 3 test':[dict_metrics_.get('RF_3_metrics')[1]],
                           'RF 3 train':[dict_metrics_.get('RF_3_metrics')[6]],
                           'RF 4 test':[dict_metrics_.get('RF_4_metrics')[1]],
                           'RF 4 train':dict_metrics_.get('RF_4_metrics')[6],
                           'RF 5 test':[dict_metrics_.get('RF_5_metrics')[1]],
                           'RF 5 train':[dict_metrics_.get('RF_5_metrics')[6]],
                           'RF 6 test':[dict_metrics_.get('RF_6_metrics')[1]],
                           'RF 6 train':[dict_metrics_.get('RF_6_metrics')[6]],
                           'RF 7 test':[dict_metrics_.get('RF_7_metrics')[1]],
                           'RF 7 train':[dict_metrics_.get('RF_7_metrics')[6]]
                           }, index=['RMSE %'])


ax=compare_metrics_F_z_H.plot.bar(color=color2, width=1.25, edgecolor=edgecolors2)

rmse_svm_1_test,rmse_train_svm_1

ax.set_xlabel('Model Classifier Input', fontsize=14, labelpad=14)
ax.set_ylabel('RMSE % (Mean)', fontsize=14,labelpad=10)
ax.title.set_size(15)
ax.set_title('')
ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')

ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
# plt.legend(['SVM 1 test','SVM 1 train', 'SVM 2 test','SVM 2 train',
#            'SVM 3 test','SVM 3 train', 'SVM 4 test','SVM 4 train',
#            'SVM 5 test','SVM 5 train', 'SVM 6 test','SVM 6 train',
#            'SVM 7 test','SVM 7 train','RF 1 test','RF 1 train',
#            'RF 2 test','RF 2 train',
#            'RF 3 test','RF 3 train', 'RF 4 test','RF 4 train',
#            'RF 5 test','RF 5 train', 'RF 6 test','RF 6 train',
#            'RF 7 test','RF 7 train'])
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.legend(loc=1, prop={'size':8}, bbox_to_anchor=(1.15,1.00))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')
plt.show()
        
    
    
color2=['lightskyblue', 'lightskyblue', 'royalblue', 'royalblue','palegreen', 'palegreen',
       'forestgreen', 'forestgreen', 'lightsalmon', 'lightsalmon', 'orangered', 'orangered',
       'bisque', 'bisque', 'orange', 'orange','slateblue', 'slateblue', 'aquamarine', 'aquamarine',
       'yellow', 'yellow', 'peru', 'peru','violet', 'violet', 'lightgray', 'lightgray']
print(len(color))
edgecolors2=['red', 'yellow']*14
print(edgecolors2)
 
compare_metrics_F_z_H=pd.DataFrame({'SVM 1 test':[dict_metrics_.get('SVM_1_metrics')[2]*100],
                                 'SVM 1 train':[ dict_metrics_.get('SVM_1_metrics')[7]*100],
                           'SVM 2 test':[dict_metrics_.get('SVM_2_metrics')[2]*100],
                           'SVM 2 train':[dict_metrics_.get('SVM_2_metrics')[7]*100],
                           'SVM 3 test':[dict_metrics_.get('SVM_3_metrics')[2]*100],
                           'SVM 3 train':[dict_metrics_.get('SVM_3_metrics')[7]*100],
                           'SVM 4 test':[dict_metrics_.get('SVM_4_metrics')[2]*100],
                           'SVM 4 train':[dict_metrics_.get('SVM_4_metrics')[7]*100],
                           'SVM 5 test':[dict_metrics_.get('SVM_5_metrics')[2]*100],
                           'SVM 5 train':[dict_metrics_.get('SVM_5_metrics')[7]*100],
                           'SVM 6 test':[dict_metrics_.get('SVM_6_metrics')[2]*100],
                           'SVM 6 train':[dict_metrics_.get('SVM_6_metrics')[7]*100],
                           'SVM 7 test':[dict_metrics_.get('SVM_7_metrics')[2]*100],
                           'SVM 7 train':[dict_metrics_.get('SVM_7_metrics')[7]*100],
                           'RF 1 test':[dict_metrics_.get('RF_1_metrics')[2]*100],
                           'RF 1 train':[dict_metrics_.get('RF_1_metrics')[7]*100],
                           'RF 2 test':[dict_metrics_.get('RF_2_metrics')[2]*100],
                           'RF 2 train':[dict_metrics_.get('RF_2_metrics')[7]*100],
                           'RF 3 test':[dict_metrics_.get('RF_3_metrics')[2]*100],
                           'RF 3 train':[dict_metrics_.get('RF_3_metrics')[7]*100],
                           'RF 4 test':[dict_metrics_.get('RF_4_metrics')[2]*100],
                           'RF 4 train':[dict_metrics_.get('RF_4_metrics')[7]*100],
                           'RF 5 test':[dict_metrics_.get('RF_5_metrics')[2]*100],
                           'RF 5 train':[dict_metrics_.get('RF_5_metrics')[7]*100],
                           'RF 6 test':[dict_metrics_.get('RF_6_metrics')[2]*100],
                           'RF 6 train':[dict_metrics_.get('RF_6_metrics')[7]*100],
                           'RF 7 test':[dict_metrics_.get('RF_7_metrics')[2]*100],
                           'RF 7 train':[dict_metrics_.get('RF_7_metrics')[7]*100]
                           }, index=['Accuracy %'])


ax=compare_metrics_F_z_H.plot.bar(color=color2, width=1.25, edgecolor=edgecolors2)

rmse_svm_1_test,rmse_train_svm_1

ax.set_xlabel('Model Classifier Input', fontsize=14, labelpad=14)
ax.set_ylabel('RMSE % (Mean)', fontsize=14,labelpad=10)
ax.title.set_size(15)
ax.set_title('')
ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')

ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
# plt.legend(['SVM 1 test','SVM 1 train', 'SVM 2 test','SVM 2 train',
#            'SVM 3 test','SVM 3 train', 'SVM 4 test','SVM 4 train',
#            'SVM 5 test','SVM 5 train', 'SVM 6 test','SVM 6 train',
#            'SVM 7 test','SVM 7 train','RF 1 test','RF 1 train',
#            'RF 2 test','RF 2 train',
#            'RF 3 test','RF 3 train', 'RF 4 test','RF 4 train',
#            'RF 5 test','RF 5 train', 'RF 6 test','RF 6 train',
#            'RF 7 test','RF 7 train'])
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.legend(loc=1, prop={'size':8}, bbox_to_anchor=(1.15,1.00))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')
plt.show()    
    
    
    
    
###HEre graphics for F1 scores plotting!!


    
color2=['lightskyblue', 'lightskyblue', 'royalblue', 'royalblue','palegreen', 'palegreen',
       'forestgreen', 'forestgreen', 'lightsalmon', 'lightsalmon', 'orangered', 'orangered',
       'bisque', 'bisque', 'orange', 'orange','slateblue', 'slateblue', 'aquamarine', 'aquamarine',
       'yellow', 'yellow', 'peru', 'peru','violet', 'violet', 'lightgray', 'lightgray']
print(len(color))
edgecolors2=['red', 'yellow']*14
print(edgecolors2)
 
compare_metrics_F_z_H=pd.DataFrame({'SVM 1 test':[dict_metrics_.get('SVM_1_metrics')[2]*100],
                                 'SVM 1 train':[ dict_metrics_.get('SVM_1_metrics')[7]*100],
                           'SVM 2 test':[dict_metrics_.get('SVM_2_metrics')[2]*100],
                           'SVM 2 train':[dict_metrics_.get('SVM_2_metrics')[7]*100],
                           'SVM 3 test':[dict_metrics_.get('SVM_3_metrics')[2]*100],
                           'SVM 3 train':[dict_metrics_.get('SVM_3_metrics')[7]*100],
                           'SVM 4 test':[dict_metrics_.get('SVM_4_metrics')[2]*100],
                           'SVM 4 train':[dict_metrics_.get('SVM_4_metrics')[7]*100],
                           'SVM 5 test':[dict_metrics_.get('SVM_5_metrics')[2]*100],
                           'SVM 5 train':[dict_metrics_.get('SVM_5_metrics')[7]*100],
                           'SVM 6 test':[dict_metrics_.get('SVM_6_metrics')[2]*100],
                           'SVM 6 train':[dict_metrics_.get('SVM_6_metrics')[7]*100],
                           'SVM 7 test':[dict_metrics_.get('SVM_7_metrics')[2]*100],
                           'SVM 7 train':[dict_metrics_.get('SVM_7_metrics')[7]*100],
                           'RF 1 test':[dict_metrics_.get('RF_1_metrics')[2]*100],
                           'RF 1 train':[dict_metrics_.get('RF_1_metrics')[7]*100],
                           'RF 2 test':[dict_metrics_.get('RF_2_metrics')[2]*100],
                           'RF 2 train':[dict_metrics_.get('RF_2_metrics')[7]*100],
                           'RF 3 test':[dict_metrics_.get('RF_3_metrics')[2]*100],
                           'RF 3 train':[dict_metrics_.get('RF_3_metrics')[7]*100],
                           'RF 4 test':[dict_metrics_.get('RF_4_metrics')[2]*100],
                           'RF 4 train':[dict_metrics_.get('RF_4_metrics')[7]*100],
                           'RF 5 test':[dict_metrics_.get('RF_5_metrics')[2]*100],
                           'RF 5 train':[dict_metrics_.get('RF_5_metrics')[7]*100],
                           'RF 6 test':[dict_metrics_.get('RF_6_metrics')[2]*100],
                           'RF 6 train':[dict_metrics_.get('RF_6_metrics')[7]*100],
                           'RF 7 test':[dict_metrics_.get('RF_7_metrics')[2]*100],
                           'RF 7 train':[dict_metrics_.get('RF_7_metrics')[7]*100]
                           }, index=['Accuracy %'])


ax=compare_metrics_F_z_H.plot.bar(color=color2, width=1.25, edgecolor=edgecolors2)

rmse_svm_1_test,rmse_train_svm_1

ax.set_xlabel('Model Classifier Input', fontsize=14, labelpad=14)
ax.set_ylabel('RMSE % (Mean)', fontsize=14,labelpad=10)
ax.title.set_size(15)
ax.set_title('')
ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')

ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
# plt.legend(['SVM 1 test','SVM 1 train', 'SVM 2 test','SVM 2 train',
#            'SVM 3 test','SVM 3 train', 'SVM 4 test','SVM 4 train',
#            'SVM 5 test','SVM 5 train', 'SVM 6 test','SVM 6 train',
#            'SVM 7 test','SVM 7 train','RF 1 test','RF 1 train',
#            'RF 2 test','RF 2 train',
#            'RF 3 test','RF 3 train', 'RF 4 test','RF 4 train',
#            'RF 5 test','RF 5 train', 'RF 6 test','RF 6 train',
#            'RF 7 test','RF 7 train'])
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.legend(loc=1, prop={'size':8}, bbox_to_anchor=(1.15,1.00))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')
plt.show()    
    
    
##barplot visualization of classification results   
###############################################################################
df_metrics_test_F_scores_.columns.tolist()
print(df_metrics_test_F_scores_.iloc[1:2,1:2])
a=df_metrics_test_F_scores_.iat[0,0]
print(a)
####################################################
#############################################

color3=['lightskyblue', 'royalblue', 'palegreen','lightgray']*4


df_metrics_test_F_scores_.iat[0,1]    
compare_metrics_F_z_H=pd.DataFrame({' 1 Model test Acc':[df_metrics_test_F_scores_.iat[0,0], df_metrics_test_F_scores_.iat[7,0]],
                                 '1 Model test f1 weighted':[ df_metrics_test_F_scores_.iat[0,4], df_metrics_test_F_scores_.iat[7,4]],
                           '1 Model test f1 Micro ':[df_metrics_test_F_scores_.iat[0,6], df_metrics_test_F_scores_.iat[7,6]],
                           '1 Model test f1 test Macro':[df_metrics_test_F_scores_.iat[0,2],df_metrics_test_F_scores_.iat[7,2] ],
                           '2 test Acc':[df_metrics_test_F_scores_.iat[1,0], df_metrics_test_F_scores_.iat[8,0]],
                           '2 Model test f1 weighted':[ df_metrics_test_F_scores_.iat[1,4],df_metrics_test_F_scores_.iat[8,4]],
                           '2 Model Micro ':[df_metrics_test_F_scores_.iat[1,6],df_metrics_test_F_scores_.iat[8,6]],
                           '2 Model Macro':[df_metrics_test_F_scores_.iat[1,2],df_metrics_test_F_scores_.iat[8,2]],
                           '3 Model test Acc':[df_metrics_test_F_scores_.iat[2,0], df_metrics_test_F_scores_.iat[9,0]],
                           '3 Model test f1 weighted':[ df_metrics_test_F_scores_.iat[2,4], df_metrics_test_F_scores_.iat[9,4]],
                           '3 Model Micro ':[df_metrics_test_F_scores_.iat[2,6],df_metrics_test_F_scores_.iat[9,6] ],
                           '3 Model Macro':[df_metrics_test_F_scores_.iat[2,2], df_metrics_test_F_scores_.iat[9,2]],
                           '7 Model test Acc':[df_metrics_test_F_scores_.iat[6,0], df_metrics_test_F_scores_.iat[13,0]],
                           '7 Model test f1 weighted':[ df_metrics_test_F_scores_.iat[6,4], df_metrics_test_F_scores_.iat[13,4]],
                           '7 Model Micro ':[df_metrics_test_F_scores_.iat[6,6], df_metrics_test_F_scores_.iat[13,6]],
                           '7 Macro':[df_metrics_test_F_scores_.iat[6,2],df_metrics_test_F_scores_.iat[13,2]]
                           }, index=['RF models', 'SVM models'])


ax=compare_metrics_F_z_H.plot.bar(color=color3, width=0.8, edgecolor=edgecolors2)

rmse_svm_1_test,rmse_train_svm_1

ax.set_xlabel('Model Classifier Input', fontsize=14, labelpad=14)
ax.set_ylabel('Accuracy % vs F1-scores % (Mean)', fontsize=14,labelpad=10)
ax.title.set_size(15)
ax.set_title('')
ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')

ax.tick_params(width=3, length=4)
#ax.tick_params(axis='both', which='major', )
#ax.set_title('Model Score Statistic (n=30 ')
#plt.title('äää')
# plt.legend(['SVM 1 test','SVM 1 train', 'SVM 2 test','SVM 2 train',
#            'SVM 3 test','SVM 3 train', 'SVM 4 test','SVM 4 train',
#            'SVM 5 test','SVM 5 train', 'SVM 6 test','SVM 6 train',
#            'SVM 7 test','SVM 7 train','RF 1 test','RF 1 train',
#            'RF 2 test','RF 2 train',
#            'RF 3 test','RF 3 train', 'RF 4 test','RF 4 train',
#            'RF 5 test','RF 5 train', 'RF 6 test','RF 6 train',
#            'RF 7 test','RF 7 train'])
ax.tick_params(width=3, length=4)
ax.grid(False)
plt.legend(loc=1, prop={'size':8}, bbox_to_anchor=(1.25,1.00))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold', color='black')
plt.show()    
         
#####################################################
         
#dict_metrics_.update({'SVM_5_metrics':[rmse_svm_3_test,rmse_svm_3_test_perc, accuracy_svm_3_test, cm1_test,matrix_df_test,rmse_svm_3_train,rmse_prec_svm_3_train,accuracy_svm_3_train, cm_1_df, matrix_df]})    
dict_metrics_.get('SVM_5_metrics')[0]####RMSE test
dict_metrics_.get('SVM_5_metrics')[1]#####RMSE % test 
dict_metrics_.get('SVM_5_metrics')[2]#####Accuracy test
dict_metrics_.get('SVM_5_metrics')[3]####Confusion 1 Matrix test
dict_metrics_.get('SVM_5_metrics')[4]####Confusion 2 Matrix test
dict_metrics_.get('SVM_5_metrics')[5]###RMSE train
dict_metrics_.get('SVM_5_metrics')[6]##RMSE % train
dict_metrics_.get('SVM_5_metrics')[7]##Accuracy train
dict_metrics_.get('SVM_5_metrics')[8]###Confusion 1 Matrix train
dict_metrics_.get('SVM_5_metrics')[9]###Confusion 2 Matrix train


####### for indexing
#dict_metrics_.update({'RF_5_metrics': [rf_3_rmse_test,rmse_rf_3_test_perc,rf_3_test_accuracy,cm1_test,matrix_df_test,rf_3_rmse_train, rf_3_rmse_perc_train,rf_3_train_accuracy,cm_1_df, matrix_df]})

dict_metrics_.get('RF_5_metrics')[0]

dict_metrics_.get('RF_5_metrics')[0]####RMSE test
dict_metrics_.get('RF_5_metrics')[1]#####RMSE % test 
dict_metrics_.get('RF_5_metrics')[2]#####Accuracy test
dict_metrics_.get('RF_5_metrics')[3]####Confusion 1 Matrix test
dict_metrics_.get('RF_5_metrics')[4]####Confusion 2 Matrix test
dict_metrics_.get('RF_5_metrics')[5]###RMSE train
dict_metrics_.get('RF_5_metrics')[6]##RMSE % train
dict_metrics_.get('RF_5_metrics')[7]##Accuracy train
dict_metrics_.get('RF_5_metrics')[8]###Confusion 1 Matrix train
dict_metrics_.get('RF_5_metrics')[9]###Confusion 2 Matrix train
##automatically select best model and confusion matkri
#########################################################################################
#########################################################################################
model_list=[]
accuraccy_test=[]
accuraccy_train=[]
model_list.append('SVM_1_metrics')
model_list.append('SVM_2_metrics')
model_list.append('SVM_3_metrics')
model_list.append('SVM_4_metrics')
model_list.append('SVM_5_metrics')
model_list.append('SVM_6_metrics')
model_list.append('SVM_7_metrics')
model_list.append('RF_1_metrics')
model_list.append('RF_2_metrics')
model_list.append('RF_3_metrics')
model_list.append('RF_4_metrics')
model_list.append('RF_5_metrics')
model_list.append('RF_6_metrics')
model_list.append('RF_7_metrics')
################################################################

###accuraccy test lists
accuraccy_test.append(dict_metrics_.get('SVM_1_metrics')[2])
accuraccy_test.append(dict_metrics_.get('SVM_2_metrics')[2])
accuraccy_test.append(dict_metrics_.get('SVM_3_metrics')[2])
accuraccy_test.append(dict_metrics_.get('SVM_4_metrics')[2])
accuraccy_test.append(dict_metrics_.get('SVM_5_metrics')[2])
accuraccy_test.append(dict_metrics_.get('SVM_6_metrics')[2])
accuraccy_test.append(dict_metrics_.get('SVM_7_metrics')[2])
accuraccy_test.append(dict_metrics_.get('RF_1_metrics')[2])
accuraccy_test.append(dict_metrics_.get('RF_2_metrics')[2])
accuraccy_test.append(dict_metrics_.get('RF_3_metrics')[2])
accuraccy_test.append(dict_metrics_.get('RF_4_metrics')[2])
accuraccy_test.append(dict_metrics_.get('RF_5_metrics')[2])
accuraccy_test.append(dict_metrics_.get('RF_6_metrics')[2])
accuraccy_test.append(dict_metrics_.get('RF_7_metrics')[2])

###accuraccy train list
accuraccy_train.append(dict_metrics_.get('SVM_1_metrics')[7])
accuraccy_train.append(dict_metrics_.get('SVM_2_metrics')[7])
accuraccy_train.append(dict_metrics_.get('SVM_3_metrics')[7])
accuraccy_train.append(dict_metrics_.get('SVM_4_metrics')[7])
accuraccy_train.append(dict_metrics_.get('SVM_5_metrics')[7])
accuraccy_train.append(dict_metrics_.get('SVM_6_metrics')[7])
accuraccy_train.append(dict_metrics_.get('SVM_7_metrics')[7])
accuraccy_train.append(dict_metrics_.get('RF_1_metrics')[7])
accuraccy_train.append(dict_metrics_.get('RF_2_metrics')[7])
accuraccy_train.append(dict_metrics_.get('RF_3_metrics')[7])
accuraccy_train.append(dict_metrics_.get('RF_4_metrics')[7])
accuraccy_train.append(dict_metrics_.get('RF_5_metrics')[7])
accuraccy_train.append(dict_metrics_.get('RF_6_metrics')[7])
accuraccy_train.append(dict_metrics_.get('RF_7_metrics')[7])


#################################################################
###Calculate the best model with the best overall accuracy
zipped_metrics=list(zip(model_list, accuraccy_test, accuraccy_train))

zipped_metrics_sorted_test=sorted(zipped_metrics, key=lambda x: x[1])
print(zipped_metrics_sorted_test[-1:])
print(zipped_metrics_sorted_test[-1:])
print('The model with the highest Overall accuracy of the Testdata is: ')
best_model_test=[]
for i in zipped_metrics_sorted_test[-1:]:
    print(i[0],'Testaccuracy:',i[1],'Trainaccuracy:', i[2])
    model_best=i[0]
    best_model_test.append(model_best)
##### get the best model from the metric dictionary    
    model_metrics_best_test=dict_metrics_.get(best_model_test[0])

































