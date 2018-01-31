import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb

train1 = pd.read_csv("train1.csv")
train9 = pd.read_csv("train9.csv")
hero_data = pd.read_csv("hero_data.csv")

test1 = pd.read_csv("test1.csv")
test9 = pd.read_csv("test9.csv")

train_set =  pd.concat([train1,train9])
validation_set = test9
test_set = test1

train_set = pd.merge(train_set, hero_data, on = 'hero_id')
validation_set = pd.merge(validation_set, hero_data, on = 'hero_id')
test_set = pd.merge(test_set, hero_data, on = 'hero_id')

#new feature
#Total Number of games played by each user
#Total Types of games played by each user
#Number of times hero appeared
#Types of game hero appeared
#May need to add validation set for training too

#Target kda_ratio
#kda_ratio (target)
#((Kills + Assists)*1000/Deaths) 
#Ratio: where kill, assists and deaths are average values per match for that hero

#Test columns not present: num_wins


#One hot
train_one_hot = pd.get_dummies(train_set[['primary_attr','attack_type']])
validation_one_hot = pd.get_dummies(validation_set[['primary_attr','attack_type']])
test_one_hot = pd.get_dummies(test_set[['primary_attr','attack_type']])


#Make columns from text and one hot
train_set['roles']
set([j for i in train_set['roles'].apply(lambda x : str(x).split(':')) for j in i])
#{'Escape', 'Carry', 'Disabler', 'Initiator', 'Durable', 'Nuker', 'Pusher', 'Jungler', 'Support'}

for st in ['Escape', 'Carry', 'Disabler', 'Initiator', 'Durable', 'Nuker', 'Pusher', 'Jungler', 'Support']:
	train_set[st] = train_set['roles'].apply(lambda x : 1 if st in str(x) else 0)
	validation_set[st] = validation_set['roles'].apply(lambda x : 1 if st in str(x) else 0)
	test_set[st] = test_set['roles'].apply(lambda x : 1 if st in str(x) else 0)


#Remove : as all have fixed values
for cl in ['base_health', 'base_mana', 'base_mana_regen']:
	del train_set[cl]
	del validation_set[cl]
	del test_set[cl]


train_all = pd.concat([train_set,train_one_hot], axis = 1)
validation_all = pd.concat([validation_set, validation_one_hot], axis = 1)
test_all = pd.concat([test_set, test_one_hot], axis = 1)

features = ['num_games', 'base_health_regen', 'base_armor', 'base_magic_resistance', 'base_attack_min',
       'base_attack_max', 'base_strength', 'base_agility', 'base_intelligence',
       'strength_gain', 'agility_gain', 'intelligence_gain', 'attack_range',
       'projectile_speed', 'attack_rate', 'move_speed', 'turn_rate', 'Escape',
       'Carry', 'Disabler', 'Initiator', 'Durable', 'Nuker', 'Pusher',
       'Jungler', 'Support', 'primary_attr_agi', 'primary_attr_int',
       'primary_attr_str', 'attack_type_Melee', 'attack_type_Ranged']


target = ['num_wins', 'kda_ratio']
###############Data_visualization
###############Missing value treatment

dtrain = xgb.DMatrix(train_all[features], train_all['kda_ratio'], missing=np.nan)
dvalid = xgb.DMatrix(validation_all[features], missing=np.nan)
dtest = xgb.DMatrix(test_all[features], missing=np.nan)

watchlist = [(dtrain, 'train')]

nrounds = 1000
params = {"objective": "reg:linear","booster": "gbtree", "nthread": 4,"silent": 1,"eta": 0.01, "max_depth": 8, "subsample": 0.1, "colsample_bytree": 0.7,"min_child_weight": 1,"seed": 2016,"tree_method": "exact"}
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=400)
valid_preds = bst.predict(dvalid)
print(mean_squared_error(validation_set['kda_ratio'], valid_preds))

test_preds = bst.predict(dtest)
sub = pd.DataFrame({'id':train_set['id'],'kda_ratio':test_preds})
i = i + 1
sub.to_csv('sub_' + str(i) + ".csv", index = False)

#*** Strategy - 1
#*** Predict kda_ratio directly
#*** Validation, train mix
#use sklearn xgboost, lightgbm, catboost

base_agility, primary_attr_agi, support most correlated with target: kda_ratio
base_strength, strength_gain, Nuker more correlated with target: kda_ratio

#############TO EXPLORE
#LightGBM links: 
#https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
#http://lightgbm.readthedocs.io/en/latest/Python-API.html
#http://lightgbm.readthedocs.io/en/latest/Parameters.html
#http://lightgbm.readthedocs.io/en/latest/Quick-Start.html
#http://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
#callback#https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
import lightgbm as lgb
dtrain = lgb.Dataset(train_all[features], train_all['kda_ratio'], free_raw_data=False)
dvalid = lgb.Dataset(validation_all[features], reference = dtrain, free_raw_data=False)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'rmse'},
    'num_leaves': 25,
    'learning_rate': 0.01,
    'feature_fraction': 0.2,
    'bagging_fraction': 0.09,
    'bagging_freq': 2,
    'verbose': 0
}
evals_result = {}
print('Start training...')
# train
lgbm = lgb.train(params, dtrain, num_boost_round=950, valid_sets=dvalid,  evals_result=evals_result, verbose_eval=400)#early_stopping_rounds=500,

print('Save model...')
# save model to file
#lgbm.save_model('model.txt')

print('Start predicting...')
# predict
valid_preds = lgbm.predict(validation_all[features], num_iteration=lgbm.best_iteration)
# eval
print('The rmse of prediction using validation set is:', mean_squared_error(validation_set['kda_ratio'], valid_preds) ** 0.5)
print('The mse of prediction using validation set is:', mean_squared_error(validation_set['kda_ratio'], valid_preds))
test_preds = lgbm.predict(test_all[features])
sub = pd.DataFrame({'id':test_set['id'],'kda_ratio':test_preds})
i = i + 1
print(i)
sub.to_csv('lgbm_sub_' + str(i) + ".csv", index = False)


#Catbost links:
#https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboostregressor-docpage/
#https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/#regression
#Catboost Visualization# https://tech.yandex.com/catboost/doc/dg/features/visualization_catboost-viewer-docpage/#visualization_catboost-viewer
from catboost import CatBoostRegressor
model = CatBoostRegressor(depth=14, iterations=88, learning_rate=0.1, eval_metric='RMSE', random_seed=1,use_best_model=True)#, verbose=True
model.fit(train_all[features],train_all['kda_ratio'],cat_features=[],eval_set = (validation_all[features], validation_all['kda_ratio']),use_best_model = True)
valid_preds = model.predict(validation_all[features])
print('The rmse of prediction using validation set is:', mean_squared_error(validation_set['kda_ratio'], valid_preds) ** 0.5)
print('The mse of prediction using validation set is:', mean_squared_error(validation_set['kda_ratio'], valid_preds))
test_preds = lgbm.predict(test_all[features])
sub = pd.DataFrame({'id':test_set['id'],'kda_ratio':test_preds})
i = i + 1
print(i)
sub.to_csv('catboost_sub_' + str(i) + ".csv", index = False)


