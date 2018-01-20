import pandas as pd
import numpy as np
import xgboost as xgb
import pandas_profiling
import random
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

i = 1

train = pd.read_csv("/home/sohom/Desktop/AV_NMIMS_Jan18/train_jDb5RBj.csv")
test = pd.read_csv("/home/sohom/Desktop/AV_NMIMS_Jan18/test_dan2xFI.csv")

profile = pandas_profiling.ProfileReport(train)
rejected_variables_train = profile.get_rejected_variables(threshold=0.9)
profile.to_file(outputfile="pandas_profiling_output_train.html")

profile = pandas_profiling.ProfileReport(test)
rejected_variables_test = profile.get_rejected_variables(threshold=0.9)
profile.to_file(outputfile="pandas_profiling_output_test.html")

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": 2,
                "seed": 2016, "tree_method": "exact"}

params1 = {"objective": "binary:logistic","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1,
                "seed": 2016, "tree_method": "exact"}


features = np.setdiff1d(train.columns, ['ID', 'Purchase', 'AAANHANG', 'ABESAUT', 'ABROM', 'ABYSTAND', 'AFIETS', 'AGEZONG', 'APERSAUT', 'APLEZIER', 'ATRACTOR', 'AVRAAUT', 'AWALAND', 'AWAOREG', 'AWAPART', 'AWERKT', 'MOSHOOFD'])

train_ids = random.sample(list(train.index),int(.8*len(train.index)))

train_new = train[train.index.isin(train_ids)]
valid = train[~train.index.isin(train_ids)]

dtrain = xgb.DMatrix(train_new[features], train_new['Purchase'], missing=np.nan)
dvalid = xgb.DMatrix(valid[features], missing=np.nan)
dtest = xgb.DMatrix(test[features], missing=np.nan)

nrounds = 300
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)

valid_preds = bst.predict(dvalid)
fpr, tpr, thresholds = metrics.roc_curve(valid['Purchase'], valid_preds, pos_label=2)
metrics.auc(fpr, tpr)
print(thresholds)

test_preds = bst.predict(dtest)
submit = pd.DataFrame({'ID': test['ID'], 'Purchase': test_preds})
submit.to_csv("XGB"+str(i)+".csv", index=False)
i = i +1


#use catboost
#Categorical: 1. MOSTYPE
#

#Make different models for different age groups #see L1

import seaborn as sns
import matplotlib.pyplot as plt
import operator
sns.boxplot(data=train, x = '')

###########################################################################################################################
#Plot variable importance using SRK code
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i,feat))
    outfile.close()


create_feature_map(features)
bst.dump_model('xgbmodel.txt', 'xgb.fmap', with_stats=True)
importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()
imp_df.to_csv("imp_feat.txt", index=False)


# create a function for labeling #
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height, '%f' % float(height), ha='center', va='bottom')


#imp_df = pd.read_csv('imp_feat.txt')
labels = np.array(imp_df.feature.values)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,6))
rects = ax.bar(ind, np.array(imp_df.fscore.values), width=width, color='y')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Importance score")
ax.set_title("Variable importance")
autolabel(rects)
plt.savefig('dummy_feature_imp_diagram.png',dpi=1000)
plt.show()
###########################################################################################################################
from sklearn.feature_extraction.text import CountVectorizer

#New Features
1) Income : Low, High, NA 
2) Age : young, juniors, seniors
3) Replace: MOSTYPE code by 'text' and make tfidf out of it

di = {}
for line in open("MOSTYPE_Customer subtype.txt"):
	line = line[:-1].split(': ')
	di[int(line[0])]= line[1]

train['new_mostype'] = train['MOSTYPE'].replace(to_replace=di)
trest['new_mostype'] = test['MOSTYPE'].replace(to_replace=di)

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)

4) MOSTYPE : Treat as categorical [May do one hot encoding]
5) MOSHOOFD Customer main type : L2 : Treat as categorical [May do one hot encoding]
6) Model make for only those train set which are present in test set
