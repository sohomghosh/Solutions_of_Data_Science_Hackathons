import pandas as pd
import numpy as np
import random
import xgboost as xgb
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, auc,roc_curve
import pandas_profiling 
from scipy import stats
from datetime import datetime

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")



profile_train = pandas_profiling.ProfileReport(train)
rejected_variables = profile_train.get_rejected_variables(threshold=0.9)
profile_train.to_file(outputfile="pandas_profiling_output_train.html")

profile_test = pandas_profiling.ProfileReport(test)
rejected_variables = profile_test.get_rejected_variables(threshold=0.9)
profile_test.to_file(outputfile="pandas_profiling_output_test.html")

'''
/#Gender : spelling correction - one hot make
///#DOB: Age calculate
///#Lead_Creation_Date: Date difference of lead creation from today
//#City_Code: LabelEncoder
/#City_Category: One hot encode
//#Employer_Code: LabelEncoder
/#Employer_Category1: One hot encode
/#Employer_Category2: One hot encode
/#Customer_Existing_Primary_Bank_Code: One hot encode
/#Primary_Bank_Type: One hot encode
/#Contacted: One hot encode
/#Source: One hot encode
/#Source_Category: One hot encode
/#Loan_Amount / income
/#Loan_Amount / EMI
/#income - existing -emi
/#existing emi > or < emi
/#EMI is available
/#Var1: One hot enpcde

*AFTER*#Interest = P*T*R/100 or compound interest
#MISSING VALUE TREATMENT
'''

#train['Approved'].value_counts()
#0    68693
#1     1020

#The Evaluation Criteria for this problem is AUC_ROC .

train_test = train.append(test)

one_hot_encoded_train_test = pd.get_dummies(train_test[['Gender','City_Category','Employer_Category1','Employer_Category2','Customer_Existing_Primary_Bank_Code','Primary_Bank_Type', 'Contacted', 'Source', 'Source_Category', 'Var1']])

lbl = LabelEncoder()
train_test['city_encoded'] = lbl.fit_transform(list(train_test['City_Code'].values))
train_test['employer_encoded'] = lbl.fit_transform(list(train_test['Employer_Code'].values))
a = datetime.today() - pd.to_datetime(train_test['DOB'], format='%d/%m/%y',errors='coerce')
train_test['age'] = a.dt.days
ld = datetime.today() - pd.to_datetime(train_test['Lead_Creation_Date'], format='%d/%m/%y',errors='coerce')
train_test['lead_age'] = ld.dt.days
train_test['loan_emi_ratio'] = train_test['Loan_Amount']/train_test['EMI']
train_test['loan_income_ratio'] = train_test['Loan_Amount']/train_test['Monthly_Income']
train_test['savings'] = train_test['Monthly_Income'] - train_test['Existing_EMI'] - train_test['EMI']
train_test['is_new_emi_more'] = train_test['EMI']>train_test['Existing_EMI']
train_test['is_new_emi_more'] = train_test['is_new_emi_more'].replace(to_replace = {True:1,False:0})
train_test['is_emi_available'] = train_test['EMI'].isnull().replace(to_replace = {True:1,False:0})


train_test_use = pd.concat([train_test[['Monthly_Income','Existing_EMI','Loan_Amount','Loan_Period','Interest_Rate','EMI','city_encoded','employer_encoded','age','lead_age','loan_emi_ratio','loan_income_ratio','savings','is_new_emi_more','is_emi_available','Approved']],one_hot_encoded_train_test], axis=1)

train_test_use.to_csv('train_test_use.csv',index=False)

X_train_all=train_test_use[0:len(train.index)]
X_test=train_test_use[len(train.index):len(train_test_use.index)]
features = ['Monthly_Income','Existing_EMI','Loan_Amount','Loan_Period','Interest_Rate','EMI','city_encoded','employer_encoded','age','lead_age','loan_emi_ratio','loan_income_ratio','savings','is_new_emi_more','is_emi_available'] + list(one_hot_encoded_train_test.columns)

X_train=X_train_all.sample(frac=0.80, replace=False)
X_valid=pd.concat([X_train_all, X_train]).drop_duplicates(keep=False)



dtrain = xgb.DMatrix(X_train[features],X_train['Approved'] , missing=np.nan)
dvalid = xgb.DMatrix(X_valid[features],missing=np.nan)
dtest = xgb.DMatrix(X_test[features], missing=np.nan)

'''
nrounds = 820
watchlist = [(dtrain, 'train')]
params = {"objective": "binary:logistic", "booster" : "gbtree", "nthread": 4,"silent": 1,"eta": 0.04, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,"min_child_weight": 1,"seed": 2016, "tree_method": "exact"}
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
valid_preds = bst.predict(dvalid)
test_preds = bst.predict(dtest)

roc_auc_score(X_valid['Approved'], valid_preds)

ans = [1 if i>.2 else 0 for i in test_preds]

sub = pd.DataFrame({'ID':test['ID'],'Approved':ans})
sub[['ID','Approved']].to_csv('sub_1.csv', index = False)
'''



nrounds = 1820
watchlist = [(dtrain, 'train')]
params = {"objective": "multi:softmax", "num_class":2 , "booster" : "gbtree", "nthread": 4,"silent": 1,"eta": 0.04, "max_depth": 7, "subsample": 0.9, "colsample_bytree": 0.7,"min_child_weight": 1,"seed": 2016, "tree_method": "exact"}

params = {"objective": "multi:softmax", "num_class":2 , "booster" : "gbtree", "nthread": 4,"silent": 1,"eta": 0.001, "max_depth": 3, "subsample": 0.9, "colsample_bytree": 0.7,"min_child_weight": 50,"seed": 2016, "tree_method": "exact", "scale_pos_weight":0.6,"colsample_bylevel":0.8, "n_estimators":100}

bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
valid_preds = bst.predict(dvalid)
test_preds = bst.predict(dtest)

roc_auc_score(X_valid['Approved'], valid_preds)


sub = pd.DataFrame({'ID':test['ID'],'Approved':ans})
sub[['ID','Approved']].to_csv('sub_'+str(i)+'.csv', index = False)
i = i+1

'''
#TO-DO
smote - class balance
TSNE of whole in 2D; plot 2 classes with different colors
'''
