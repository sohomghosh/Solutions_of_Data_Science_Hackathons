
# coding: utf-8

# In[1]:

import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')


# In[2]:

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pandas_profiling
import seaborn as sns
import mpld3
mpld3.enable_notebook()


# In[3]:

train1 = pd.read_csv("train1.csv")
train9 = pd.read_csv("train9.csv")
hero_data = pd.read_csv("hero_data.csv")
test1 = pd.read_csv("test1.csv")
test9 = pd.read_csv("test9.csv")


# In[4]:

train_set =  pd.concat([train1,train9])
validation_set = test9
test_set = test1
train_set = pd.merge(train_set, hero_data, on = 'hero_id')
validation_set = pd.merge(validation_set, hero_data, on = 'hero_id')
test_set = pd.merge(test_set, hero_data, on = 'hero_id')


# In[6]:

pandas_profiling.ProfileReport(train_set)


# In[7]:

pandas_profiling.ProfileReport(validation_set)


# In[8]:

pandas_profiling.ProfileReport(test_set)


# In[5]:

train_one_hot = pd.get_dummies(train_set[['primary_attr','attack_type']])
validation_one_hot = pd.get_dummies(validation_set[['primary_attr','attack_type']])
test_one_hot = pd.get_dummies(test_set[['primary_attr','attack_type']])


# In[6]:

for st in ['Escape', 'Carry', 'Disabler', 'Initiator', 'Durable', 'Nuker', 'Pusher', 'Jungler', 'Support']:
	train_set[st] = train_set['roles'].apply(lambda x : 1 if st in str(x) else 0)
	validation_set[st] = validation_set['roles'].apply(lambda x : 1 if st in str(x) else 0)
	test_set[st] = test_set['roles'].apply(lambda x : 1 if st in str(x) else 0)


# In[7]:

for cl in ['base_health', 'base_mana', 'base_mana_regen']:
	del train_set[cl]
	del validation_set[cl]
	del test_set[cl]


train_all = pd.concat([train_set,train_one_hot], axis = 1)
validation_all = pd.concat([validation_set, validation_one_hot], axis = 1)
test_all = pd.concat([test_set, test_one_hot], axis = 1)


# In[8]:

features = ['num_games', 'base_health_regen', 'base_armor', 'base_magic_resistance', 'base_attack_min',
       'base_attack_max', 'base_strength', 'base_agility', 'base_intelligence',
       'strength_gain', 'agility_gain', 'intelligence_gain', 'attack_range',
       'projectile_speed', 'attack_rate', 'move_speed', 'turn_rate', 'Escape',
       'Carry', 'Disabler', 'Initiator', 'Durable', 'Nuker', 'Pusher',
       'Jungler', 'Support', 'primary_attr_agi', 'primary_attr_int',
       'primary_attr_str', 'attack_type_Melee', 'attack_type_Ranged']


target = ['kda_ratio']


# In[20]:

sns.heatmap(train_all[features + target].corr(),annot=True)


# In[21]:

train_all[features + target].corr()


# # SelectKBest

# In[386]:

from sklearn.feature_selection import SelectKBest, chi2, f_classif
ch2 = SelectKBest(chi2, k=4)
selector = SelectKBest(f_classif, k=5)
selector.fit_transform(train_all[features], train_all['kda_ratio'])


# In[387]:

scores = selector.scores_
pd.DataFrame({'feature_name':features,'importance_scores':scores}).sort_values('importance_scores', ascending = False)


# In[388]:

selector.pvalues_


# In[379]:

'''
#chi2 for categorical features
selector = SelectKBest(chi2, k=5)
selector.fit_transform(train_all[[categorical_features]], train_all['kda_ratio'])
scores = selector.scores_

# Get the raw p-values for each feature, and transform from p-values into scores #Explore more
raw_p_values = -np.log10(selector.pvalues_)
'''


# # Recursive Feature Elimination

# In[396]:

#http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
from sklearn.feature_selection import RFE
#from sklearn.svm import SVC #for classsification
#svc = SVC(kernel="linear", C=1) #for classsification
#rfe = RFE(estimator=svc, n_features_to_select=5, step=1) #for classsification
from sklearn.svm import SVR
svr = SVR(kernel="linear")
rfe = RFE(estimator=svr, n_features_to_select=5, step=1)
rfe.fit(train_all[features], train_all['kda_ratio'])
ranking = rfe.ranking_.reshape(digits.images[0].shape)
rfe.ranking


# In[ ]:

from sklearn.feature_selection import RFECV
#svc = SVC(kernel="linear") #For classification
#    The "accuracy" scoring is proportional to the number of correct classifications
#rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy') #For classification
from sklearn.svm import SVR
svr = SVR(kernel="linear")
rfecv = RFECV(estimator=svr, step=1, cv=StratifiedKFold(2), scoring='RMSE')
rfecv.fit(train_all[features], train_all['kda_ratio'])
print("Optimal number of features : %d" % rfecv.n_features_)
rfecv.ranking_
rfecv.grid_scores_


# In[393]:

features
#


# In[59]:

dtrain = xgb.DMatrix(train_all[features], train_all['kda_ratio'], missing=np.nan)
dvalid = xgb.DMatrix(validation_all[features], missing=np.nan)
dtest = xgb.DMatrix(test_all[features], missing=np.nan)

watchlist = [(dtrain, 'train')]


# In[60]:

nrounds = 850
params = {"objective": "reg:linear","booster": "gbtree", "nthread": 4,"silent": 1,"eta": 0.01, "max_depth": 8, "subsample": 0.01, "colsample_bytree": 0.3,"min_child_weight": 3,"seed": 2016,"tree_method": "exact"}
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=400)
valid_preds = bst.predict(dvalid)
print(mean_squared_error(validation_set['kda_ratio'], valid_preds))


# In[64]:

test_preds = bst.predict(dtest)
sub = pd.DataFrame({'id':test_set['id'],'kda_ratio':test_preds})
i = i + 1
sub.to_csv('sub_' + str(i) + ".csv", index = False)


# In[164]:

import lightgbm as lgb


# In[165]:

dtrain = lgb.Dataset(train_all[features], train_all['kda_ratio'], free_raw_data=False)
dvalid = lgb.Dataset(validation_all[features], reference = dtrain, free_raw_data=False)


# In[352]:

#http://lightgbm.readthedocs.io/en/latest/Parameters.html
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'rmse'},
    'num_leaves': 25,
    'learning_rate': 0.01,
    'feature_fraction': 0.2,
    'bagging_fraction': 0.09,
    'tree_learner':'voting',
    'bagging_freq': 2,
    'max_depth':15,
    'min_data_in_leaf':22,
    'min_sum_hessian_in_leaf':1e-3,
    'feature_fraction':.6,
    'feature_fraction_seed':2,
    'bagging_fraction':.09,
    'bagging_freq':2,
    'bagging_seed':3,
    'early_stopping_round':0,
    #'lambda_l1':.4,
    #'lambda_l2':.1,
    'min_split_gain':0,
    'drop_rate':0.01,
    'skip_drop':0.5,
    'max_drop':1,
    'uniform_drop':'false',
    'xgboost_dart_mode':'false',
    'drop_seed':4,
    'top_rate':0.2,
    'other_rate':0.1,
    #'max_cat_threshold':32,
    #'cat_smooth':10,
    #'cat_l2':10,
    #'max_cat_to_onehot':4,
    'top_k':20,
    #'max_bin':50,
    #min_data_in_bin
    #data_random_seed
    #output_model
    #input_model
    #output_result
    #pre_partition
    #is_sparse
    #two_round
    #save_binary
    #verbosity
    #header
    #label
    #weight
    #query
    #ignore_column
    #categorical_feature
    #predict_raw_score
    #predict_leaf_index
    #predict_contrib
    #bin_construct_sample_cnt
    #num_iteration_predict
    #pred_early_stop
    #pred_early_stop_freq
    #pred_early_stop_margin
    #use_missing
    #zero_as_missing
    #init_score_file
    #valid_init_score_file
    #sigmoid
    #alpha
    #fair_c
    #poisson_max_delta_step
    #scale_pos_weight
    #boost_from_average
    #is_unbalance
    #max_position
    #label_gain
    #num_class
    #reg_sqrt
    #tweedie_variance_power

    'verbose': 0
}
evals_result = {}


# In[353]:

print('Start training...')
# train
lgbm = lgb.train(params, dtrain, num_boost_round=950, valid_sets=dvalid,  evals_result=evals_result, verbose_eval=400)#early_stopping_rounds=500,

print('Save model...')
# save model to file
#lgbm.save_model('model.txt')

print('Start predicting...')
# predict
valid_preds = lgbm.predict(validation_all[features], num_iteration=lgbm.best_iteration)#num_iteration = num_trees
# eval
print('The rmse of prediction using validation set is:', mean_squared_error(validation_set['kda_ratio'], valid_preds) ** 0.5)
print('The mse of prediction using validation set is:', mean_squared_error(validation_set['kda_ratio'], valid_preds))


# In[16]:

print('Feature names:', lgbm.feature_name())
print('Feature importances:', list(lgbm.feature_importance()))


# In[214]:

test_preds = lgbm.predict(test_all[features])
sub = pd.DataFrame({'id':test_set['id'],'kda_ratio':test_preds})
i = i + 1
print(i)
sub.to_csv('lgbm_sub_' + str(i) + ".csv", index = False)


# In[37]:

print('Plot metrics during training...')
ax = lgb.plot_metric(evals_result, metric='l2')


# In[74]:

print('Plot feature importances...')
ax = lgb.plot_importance(lgbm, max_num_features=10)


# In[73]:

print('Plot 4th tree...')  # one tree use categorical feature to split
ax = lgb.plot_tree(lgbm, tree_index=3, figsize=(20, 8), show_info=['split_gain'])


# In[70]:

import graphviz
print('Plot 4th tree with graphviz...')
graph = lgb.create_tree_digraph(lgbm, tree_index=, name='Tree4')
graph.render(view=True)


# In[60]:

from sklearn.model_selection import GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
estimator = lgb.LGBMRegressor(num_boost_round=20, early_stopping_rounds=5)
grid_lgbm = GridSearchCV(estimator, param_grid)
grid_lgbm.fit(train_all[features], train_all['kda_ratio'])
print('Best parameters found by grid search are:', grid_lgbm.best_params_)


# In[59]:

train_all[features].head()


# In[61]:

'''
# dump model with pickle
with open('model.pkl', 'wb') as fout:
    pickle.dump(gbm, fout)
# load model with pickle to predict
with open('model.pkl', 'rb') as fin:
    pkl_bst = pickle.load(fin)
# can predict with any iteration when loaded in pickle way
y_pred = pkl_bst.predict(X_test, num_iteration=7)
# eval with loaded model
'''


# In[75]:

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence


# In[78]:

#Try lgbm with sklearn wrapper
my_plots = plot_partial_dependence(sklearn_model,       
                                   features=[0,1,4], # column numbers of plots we want to show; index of columns we want to see
                                   X=train_all[features],            # raw predictors data.
                                   feature_names=['feature_1', 'feature_2', 'feature_3'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis


# In[57]:

from catboost import CatBoostRegressor
#https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboostregressor-docpage/


# In[163]:

#Description of each parameters: https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/
model = CatBoostRegressor(depth=14, iterations=88, learning_rate=0.1,loss_function='RMSE', eval_metric='RMSE', random_seed=1,use_best_model=True,
                        l2_leaf_reg=6,
                        model_size_reg=None,#Didn't have any effect
                        rsm=0.5,#To enable random subspace method for feature bagging use 
                        border_count=20,#The number of splits for numerical features. Allowed values are integers from 1 to 255 inclusively.
                        feature_border_type=None,#string : The binarization mode (the possible values of objects are divided into disjoint ranges (buckets) delimited by the threshold values (splits)) for numerical features.Possible values:-> Median Uniform UniformAndQuantiles MaxLogSum MinEntropy GreedyLogSum 
                        fold_permutation_block_size=14,#int : Objects in the dataset are grouped in blocks before the random permutations. This parameter defines the size of the blocks. The smaller is the value, the slower is the training. Large values may result in quality degradation.
                        od_pval=None,#The threshold for the IncToDec overfitting detector type. The training is stopped when the specified value is reached. Requires that a test dataset was input.For best results, it is recommended to set a value in the range. The larger the value, the earlier overfitting is detected.
                        od_wait=1,
                        od_type=None,
                        nan_mode=None,
                        counter_calc_method="Universal",#'Universal', 'Static', 'Basic'
                        leaf_estimation_iterations=None,
                        leaf_estimation_method=None,
                        thread_count=None,
                        verbose=None,
                        logging_level=None,
                        metric_period=None,
                        ctr_leaf_count_limit=None,
                        store_all_simple_ctr=None,
                        max_ctr_complexity=None,
                        has_time=None,
                        one_hot_max_size=None,
                        random_strength=1,
                        name=None,
                        ignored_features=None,
                        train_dir=None,
                        custom_metric=None,
                        bagging_temperature=1.6,
                        save_snapshot=None,
                        snapshot_file=None,
                        fold_len_multiplier=2,
                        used_ram_limit=None,
                        gpu_ram_part=None,
                        allow_writing_files=None,
                        approx_on_full_history=None,
                        boosting_type="Dynamic",
                        simple_ctr=None,
                        combinations_ctr=None,
                        per_feature_ctr=None,
                        ctr_description=None,
                        task_type=None,
                        device_config=None,
                        devices=None)#, verbose=True
model.fit(train_all[features],train_all['kda_ratio'],cat_features=[],eval_set = (validation_all[features], validation_all['kda_ratio']),use_best_model = True)
valid_preds = model.predict(validation_all[features])
print('The rmse of prediction using validation set is:', mean_squared_error(validation_set['kda_ratio'], valid_preds) ** 0.5)
print('The mse of prediction using validation set is:', mean_squared_error(validation_set['kda_ratio'], valid_preds))


# In[156]:

test_preds = lgbm.predict(test_all[features])
sub = pd.DataFrame({'id':test_set['id'],'kda_ratio':test_preds})
i = i + 1
print(i)
sub.to_csv('catboost_sub_' + str(i) + ".csv", index = False)


# # Ensemble Models

# In[354]:

best_xgboost = pd.read_csv("sub_9.csv")
best_catboost = pd.read_csv("catboost_sub_12.csv")
best_lgbm = pd.read_csv("lgbm_sub_17.csv")


# In[361]:

ans = (best_xgboost['kda_ratio'] + best_catboost['kda_ratio'] + best_lgbm['kda_ratio'])/3



# In[363]:

ensembled_sub = pd.DataFrame({'id':test_set['id'],'kda_ratio':ans.values})


# In[364]:

ensembled_sub.head()


# In[365]:

ensembled_sub.to_csv("esembled_submission_1.csv",index=False)


# # Stacking

# In[366]:

#stack_train = 
#stack_test = pd.DataFrame({'best_xgboost_kda_ratio' : best_xgboost['kda_ratio'], 'best_catboost_kda_ratio' : best_catboost['kda_ratio'], 'best_lgbm_kda_ratio' : best_lgbm['kda_ratio']})


# In[9]:
#https://www.analyticsvidhya.com/blog/2018/02/introductory-guide-regularized-greedy-forests-rgf-python/

###############Classifier#####################
from rgf.sklearn import RGFRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold, cross_val_score

rgf = RGFRegressor(max_leaf=400,
                    algorithm="RGF_Sib",
                    test_interval=100,
                    verbose=True)

rgf.fit(train_all[features], train_all['kda_ratio'])
valid_preds = list(rgf.predict(validation_all[features]))
test_preds = list(rgf.predict(test_all[features]))

valid_preds = model.predict(validation_all[features])
print('The rmse of prediction using validation set is:', mean_squared_error(validation_set['kda_ratio'], valid_preds) ** 0.5)

test_preds = list(rgf.predict(test_all[features]))


##Using grid serach
parameters = {'max_leaf':[1000,1200,1300,1400,1500,1600,1700,1800,1900,2000],
              'l2':[0.1,0.2,0.3],
              'min_samples_leaf':[5,10]}

model = GridSearchCV(estimator=rgf,
                   param_grid=parameters,
                   scoring='neg_mean_squared_error',
                   n_jobs = -1,
                   cv = 3)

model.fit(train_all[features], train_all['kda_ratio'])
model.best_params_
#OUTPUT# {'l2': 0.3, 'min_samples_leaf': 10, 'max_leaf': 1200}

np.sqrt(-model.best_score_) #Think why minus (-)
#NOW train an rgf model using these parameters




###############Classifier#####################
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold, cross_val_score
from rgf.sklearn import RGFClassifier
rgf = RGFClassifier(max_leaf=400,
                    algorithm="RGF_Sib",
                    test_interval=100,
                    verbose=True)

n_folds = 3

rgf_scores = cross_val_score(rgf,
                             train_all[features],
                             train_all['class'],
                             cv=StratifiedKFold(n_folds))

rgf_score = sum(rgf_scores)/n_folds
print('RGF Classfier score: {0:.5f}'.format(rgf_score))


