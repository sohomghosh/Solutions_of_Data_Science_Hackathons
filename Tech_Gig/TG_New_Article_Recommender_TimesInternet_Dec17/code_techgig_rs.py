import numpy as np
import pandas as pd
import gc
import random
import re
import difflib
import pickle
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import fasttext
from dateutil import parser
from sklearn.preprocessing import LabelEncoder



test = pd.read_csv("test-data.csv")
imp_ids = list(set(test['impression_id']))
item_ids = list(set(test['item_id']))

'''
train1 = pd.read_csv("/index/sohom_experiment/tg_recosys/Train/Train/Train_1.csv")
train = train1[(train1['impression_id'].isin(imp_ids))|(train1['item_id'].isin(item_ids))]
del train1
gc.collect()

train2 = pd.read_csv("/index/sohom_experiment/tg_recosys/Train/Train/Train_2.csv")
train = train.append(train2[(train2['impression_id'].isin(imp_ids))|(train2['item_id'].isin(item_ids))])
del train2
gc.collect()

for i in range(3,11):
	train_new = pd.read_csv("/index/sohom_experiment/tg_recosys/Train/Train/Train_"+str(i)+".csv")
	train = train.append(train_new[(train_new['impression_id'].isin(imp_ids))|(train_new['item_id'].isin(item_ids))])
	del train_new
	gc.collect()


train.to_csv("train_use.csv", index=False)
'''
train_use = pd.read_csv("train_use.csv")



'''
imp_attr = pd.DataFrame({})
j = 1
for i in range(1,8):
	for k in range(1,4):
		print("/index/sohom_experiment/tg_recosys/Impression-Id-Attributes/Impression_Id_Attributes/Pub_"+str(i)+"/impression_attr_"+str(j)+".csv")
		file = "/index/sohom_experiment/tg_recosys/Impression-Id-Attributes/Impression_Id_Attributes/Pub_"+str(i)+"/impression_attr_"+str(j)+".csv"
		imp_attr_new = pd.read_csv(file, low_memory = False)
		print(imp_attr_new.columns)
		j = j + 1
		imp_attr_new['impression_id'] = imp_attr_new['impression_id'].astype(str)
		imp_attr = imp_attr.append(imp_attr_new[imp_attr_new['impression_id'].isin(imp_ids)])
		print(imp_attr_new.shape)
		print("\n")
		del imp_attr_new
		gc.collect()
'''


#imp_attr = pd.read_csv("/index/sohom_experiment/tg_recosys/Impression-Id-Attributes/Impression_Id_Attributes/imp_id_attr.csv")
#imp_attr['impression_id'] = imp_attr['impression_id'].astype(str)
#imp_attr[imp_attr['impression_id'].isin(imp_ids)].to_csv('/index/sohom_experiment/tg_recosys/imp_id_attr_use.csv', index = False)

imp_attr_use = pd.read_csv("imp_id_attr_use.csv")



#itm_cat_map = pd.read_excel("/index/sohom_experiment/tg_recosys/Item_Category_Map/Item_Category_Map_1.xlsx", sheet =0)
#itm_cat_map = itm_cat_map.append(pd.read_excel("/index/sohom_experiment/tg_recosys/Item_Category_Map/Item_Category_Map_2.xlsx", sheet =0))
#itm_cat_map = itm_cat_map[itm_cat_map['item_id'].isin(item_ids)]
#itm_cat_map.to_csv("imp_cat_map_use.csv", index = False)

itm_cat_map_use = pd.read_csv("imp_cat_map_use.csv")




'''
##Item_Attributes_all.csv
#Recovering if first line is not item_id
fp = open("/index/sohom_experiment/tg_recosys/Item_Attributes_all.csv")
lines_recovered = fp.readlines()[1]
line_prev = 
for line in fp.readlines()[2:]:
	if 
'''


'''
#Eliminating commas from last of each lines of the file Item_Attributes_all.csv
item_attr_list_of_lists = []
fp = open("/index/sohom_experiment/tg_recosys/Item_Attributes_all.csv")
for line in fp.readlines()[1:]:
	line = re.sub('[^0-9a-zA-Z| +:.-]+', '', line).replace('\r','')
	tk = line.split('|')
	try:
		tk1 = int(tk[0])
		tk2 = tk[1]
		tk3 = ' '.join(tk[2:len(tk)-1])
		tk4 = tk[len(tk)-1].replace(',','').replace('\n','')
		if tk1 in item_ids:
			item_attr_list_of_lists.append([tk1,tk2,tk3,tk4])
	except ValueError:
		pass


item_attr = pd.DataFrame(item_attr_list_of_lists, columns = ['item_id', 'title', 'description', 'timestamp_creation'])
fp.close()


item_attr.to_csv("item_attr_use.csv",index = False)
'''

item_attr_use = pd.read_csv("/index/sohom_experiment/tg_recosys/item_attr_use.csv")


'''
train_use.columns
'impression_id', 'item_id', 'click'

test.columns
'impression_id', 'item_id'

imp_attr_use.columns
'impression_id', 'geo', 'site_id', 'uuid', 'adunit_id', 'refrenceUrl', 'timestamp_impression', 'uvh'

itm_cat_map_use.columns
'item_id', 'Keywords', 'Keywords_Score', 'Concepts', 'Concepts_Score'

item_attr_use.columns
'item_id', 'title', 'description', 'timestamp_creation' 
'''

train1 = pd.merge(train_use, imp_attr_use, on = 'impression_id', how = 'inner')
train2 = pd.merge(train1, itm_cat_map_use, on = 'item_id', how = 'inner')
train3 = pd.merge(train2, item_attr_use, on ='item_id', how = 'inner')

train3.shape
#(89785, 17)

train3.to_csv("train_final.csv", index = False)

train3 = pd.read_csv("train_final.csv")

test1 = pd.merge(test, imp_attr_use, on = 'impression_id', how = 'left')
test2 = pd.merge(test1, itm_cat_map_use, on = 'item_id', how = 'left')
test3 = pd.merge(test2, item_attr_use, on ='item_id', how = 'left')

test3.shape
#(491294, 16)
test.shape
#(188798, 2)

##############Need to groupby later

test3.to_csv("test_final.csv", index = False)


train_test = train3.append(test3)
train_test.to_csv("train_test.csv",index = False)


#######################################################################################################################################
#######################################################################################################################################

test = pd.read_csv("test-data.csv")
train_test = pd.read_csv("train_test.csv")

#train_test.columns
#['Concepts', 'Concepts_Score', 'Keywords', 'Keywords_Score', 'adunit_id','click', 'description', 'geo', 'impression_id', 'item_id', 'refrenceUrl', 'site_id', 'timestamp_creation', 'timestamp_impression', 'title', 'uuid', 'uvh']


'''
features: 
tfidf, word2vec, fasttext on [title, description]
1) avg. vector scores from keywords
1.5) avg. vector scores with weightage scores from keywords
2) avg. vector scores with weightage scores from concepts

//2.5) number of keywords
//2.7) number of concepts
[After] 3) bag of words based on keywords, concepts
// 4) avg. of keyword_score
// 5) avg. of concept_score

//[as it is] 5) geo
// [encode usig label encoder] 6) adunit_id
//[as it is] 7) train_test['site_id']
//8) Hour, min, sec, day from timestamp_creation
//9) Hour, min, sec, day from imestamp_impression
//10) Difference between 8 and 9

AFTER
**11) uvh	User view history, last 5 items that the user clicked. :: TRY TO INCLUDE properties of these items
*12) uuid	Unique id given to a user.



########click, non-click distribution see; may do smote
#########


label :: 2 for clicked, 1 for not clicked
1) click
'''

def date_convert(x):
	try:
		return parser.parse(str(x).replace('+5:30','GMT'))
	except:
		return pd.NaT	


train_test['timestamp_creation_formatted'] = train_test['timestamp_creation'].apply(lambda x : date_convert(x))

train_test['timestamp_impression_formatted'] =  train_test['timestamp_impression'].apply(lambda x: date_convert(x))

train_test['weekday_created'] = train_test['timestamp_creation_formatted'].dt.weekday
train_test['day_of_month_created'] = train_test['timestamp_creation_formatted'].dt.day
train_test['month_created'] = train_test['timestamp_creation_formatted'].dt.month
train_test['hour_created'] = train_test['timestamp_creation_formatted'].dt.hour
train_test['minutes_created'] = train_test['timestamp_creation_formatted'].dt.minute
train_test['seconds_created'] = train_test['timestamp_creation_formatted'].dt.second

train_test['weekday_impression'] = train_test['timestamp_impression_formatted'].dt.weekday
train_test['day_of_month_impression'] = train_test['timestamp_impression_formatted'].dt.day
train_test['hour_impression'] = train_test['timestamp_impression_formatted'].dt.hour
train_test['minutes_impression'] = train_test['timestamp_impression_formatted'].dt.minute
train_test['seconds_impression'] = train_test['timestamp_impression_formatted'].dt.second

train_test['difference_impression_creation'] = train_test['timestamp_impression_formatted'] - train_test['timestamp_creation_formatted']
train_test['difference_impression_creation_in_seconds'] = train_test['difference_impression_creation'].astype('timedelta64[s]')

lbl = LabelEncoder()
train_test['adunit_encoded'] = lbl.fit_transform(train_test['adunit_id'].values)

train_test['number_of_keywords'] = train_test['Keywords_Score'].apply(lambda x : len(str(x).split(',')))
train_test['number_of_concepts'] = train_test['Concepts_Score'].apply(lambda x : len(str(x).split(',')))

train_test['avg_keywords_score'] = train_test['Keywords_Score'].apply(lambda x : np.nanmean([float(i) for i in  str(x).split(',') if i!='null']))
train_test['avg_concepts_score'] = train_test['Concepts_Score'].apply(lambda x : np.nanmean([float(i) for i in  str(x).split(',') if i!='null']))


train_test['num_words_title'] = train_test['title'].apply(lambda x : len(x.split(' ')))

for i in range(3):
	train_test['S'+str(i)] = train_test['uvh'].apply(lambda x : len(str(x).split('S')[i].split(',')) if len(str(x).split('S')) -1 >= i else 0)


train_test.to_csv("train_test_v2.csv",index=False)


##############################################################################################################

train_test = pd.read_csv("train_test_v2.csv")
test = pd.read_csv("test-data.csv")

itm_cat_map_use = pd.read_csv("imp_cat_map_use.csv")
keywords = list(set([j.strip() for i in itm_cat_map_use['Keywords'].values for j in str(i).split(',')]))
concepts = list(set([j.strip() for i in itm_cat_map_use['Concepts'].values for j in str(i).split(',')]))

pickle.dump(keywords, open('keywords.txt', 'wb'))
pickle.dump(concepts, open('concepts.txt', 'wb'))

kc = list(set(keywords + concepts))

#sentences_split=[[j for j in re.split('\W', str(i)) if j!=''] for i in train_test['description']]
sentences_split=[[j for j in  kc if j in str(i) and j!=''] for i in train_test['description']]

pickle.dump(sentences_split, open('sentences_split', 'wb'))
#sentences_split = pickle.load(open('sentences_split', 'rb'))

keyword_score_dict = {}
concepts_score_dict = {}

for i1,i2 in zip(itm_cat_map_use['Keywords'].values,itm_cat_map_use['Keywords_Score'].values):
	for j1,j2 in zip(str(i1).split(','),str(i2).split(',')):
		if j2!='' and j2 !='null':
			keyword_score_dict[j1] = float(j2)


for i1,i2 in zip(itm_cat_map_use['Concepts'].values,itm_cat_map_use['Concepts_Score'].values):
	for j1,j2 in zip(str(i1).split(','),str(i2).split(',')):
		if j2!='' and j2 != 'null':
			concepts_score_dict[j1] = float(j2)


pickle.dump(keyword_score_dict, open('keyword_score_dict', 'wb'))
pickle.dump(concepts_score_dict, open('concepts_score_dict', 'wb'))

#wordvec
model_w2v = word2vec.Word2Vec(sentences_split, size=40,min_count =1, window=3, workers =-1,sample=1e-5)
features_sent_concepts = np.zeros(shape=(0,40))
features_sent_keywords = np.zeros(shape=(0,40))
for i in sentences_split:
	su=np.zeros(shape=(40))
	su_n=np.zeros(shape=(40))
	num_cps = 0
	num_keys = 0
	for j in i:
		if j in concepts:
			try:
				jj = difflib.get_close_matches(j,concepts)[1]
				k=np.array(model_w2v.wv[jj])
				su=su+k
				num_cps = num_cps + 1
			except (IndexError, ValueError, KeyError):
				pass
		if j in keywords:	
			try:
				jj = difflib.get_close_matches(j,keywords)[1]
				k=np.array(model_w2v.wv[jj])
				su_n=su_n+k
				num_keys = num_keys + 1
			except (IndexError, ValueError, KeyError):
				pass
	features_sent_concepts=np.vstack([features_sent_concepts, su/num_cps])
	features_sent_keywords=np.vstack([features_sent_keywords, su_n/num_keys])


pickle.dump(features_sent_concepts, open("features_sent_concepts.txt", 'wb'))
pickle.dump(features_sent_keywords, open("features_sent_keywords.txt", 'wb'))

#tfidf
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=.01, use_idf=True, smooth_idf=False, sublinear_tf=True)
sklearn_representation = sklearn_tfidf.fit([str(i) for i in train_test['description']])
train_test_tfidf=pd.DataFrame(sklearn_tfidf.transform([str(i) for i in train_test['description']]).todense())


train_test_tfidf.to_csv("train_test_tfidf.csv")

#fasttext
train_test['description'].to_csv('train_test_description.csv',index=False)
model_sk = fasttext.skipgram('train_test_description.csv', 'model_sk',dim=40)
features_sent_ft_concepts = np.zeros(shape=(0,40))
features_sent_ft_keywords = np.zeros(shape=(0,40))
for i in sentences_split:
	su=np.zeros(shape=(40))
	su_n=np.zeros(shape=(40))
	num_cps = 0
	num_keys = 0
	for j in i:
		if j in concepts:
			try:
				jj = difflib.get_close_matches(j,concepts)[1]
				k=np.array(model_sk[jj])
				su=su+k
				num_cps = num_cps + 1
			except (IndexError, ValueError, KeyError):
				pass
		if j in keywords:	
			try:
				jj = difflib.get_close_matches(j,keywords)[1]
				k=np.array(model_sk[jj])
				su_n=su_n+k
				num_keys = num_keys + 1
			except (IndexError, ValueError, KeyError):
				pass
	features_sent_ft_concepts=np.vstack([features_sent_ft_concepts, su/num_cps])
	features_sent_ft_keywords=np.vstack([features_sent_ft_keywords, su_n/num_keys])


pickle.dump(features_sent_ft_concepts, open("features_sent_ft_concepts.txt", 'wb'))
pickle.dump(features_sent_ft_keywords, open("features_sent_ft_keywords.txt", 'wb'))

#Keeping w2v, fasttext and tfidf side by side
train_test=pd.concat([train_test,pd.DataFrame(features_sent_concepts),pd.DataFrame(features_sent_keywords),pd.DataFrame(features_sent_ft_concepts),pd.DataFrame(features_sent_ft_keywords),train_test_tfidf],axis=1)

train_test.to_csv("train_test_v3.csv", index = False)


#####################################################################################################################################
features_sent_concepts = np.zeros(shape=(0,40))
features_sent_keywords = np.zeros(shape=(0,40))
for i in sentences_split:
	su=np.zeros(shape=(40))
	su_n=np.zeros(shape=(40))
	num_cps = 0
	num_keys = 0	
	for j in i:
		if j in concepts:
			try:
				jj = difflib.get_close_matches(j,concepts)[1]
				k=np.array(model_w2v.wv[jj])
				su=su+k*concepts_score_dict[j]
				num_cps = num_cps + 1
			except (IndexError, ValueError, KeyError):
				pass
		if j in keywords:	
			try:
				jj = difflib.get_close_matches(j,keywords)[1]
				k=np.array(model_w2v.wv[jj])
				su_n=su_n+k*keyword_score_dict[jj]
				num_keys = num_keys + 1
			except (IndexError, ValueError, KeyError):
				pass
	features_sent_concepts=np.vstack([features_sent_concepts, su/num_cps])
	features_sent_keywords=np.vstack([features_sent_keywords, su_n/num_keys])


pickle.dump(features_sent_ft_concepts, open("features_sent_ft_concepts.txt", 'wb'))
pickle.dump(features_sent_ft_keywords, open("features_sent_ft_keywords.txt", 'wb'))



features_sent_ft_concepts = np.zeros(shape=(0,40))
features_sent_ft_keywords = np.zeros(shape=(0,40))
for i in sentences_split:
	su=np.zeros(shape=(40))
	su_n=np.zeros(shape=(40))
	num_cps = 0
	num_keys = 0
	for j in i:
		if j in concepts:
			try:
				jj = difflib.get_close_matches(j,concepts)[1]
				k=np.array(model_sk[jj])
				su=su+k*concepts_score_dict[jj]
				num_cps = num_cps + 1
			except (IndexError, ValueError, KeyError):
				pass
		if j in keywords:
			try:
				jj = difflib.get_close_matches(j,keywords)[1]
				k=np.array(model_sk[jj])
				su_n=su_n+k*keyword_score_dict[jj]
				num_keys = num_keys + 1
			except (IndexError, ValueError, KeyError):
				pass
	features_sent_ft_concepts=np.vstack([features_sent_ft_concepts, su/num_cps])
	features_sent_ft_keywords=np.vstack([features_sent_ft_keywords, su_n/num_keys])


pickle.dump(features_sent_ft_concepts, open("features_sent_ft_concepts_weight.txt", 'wb'))
pickle.dump(features_sent_ft_keywords, open("features_sent_ft_keywords_weight.txt", 'wb'))

#np.savetxt("features_sent_ft_concepts_weight.txt", features_sent_ft_concepts)
#np.savetxt("features_sent_ft_keywords_weight.txt", features_sent_ft_keywords)


train_test = pd.concat([train_test, pd.DataFrame(features_sent_concepts), pd.DataFrame(features_sent_keywords), pd.DataFrame(features_sent_ft_concepts), pd.DataFrame(features_sent_ft_keywords)], axis=1)
train_test.to_csv("train_test_v4.csv", index = False)

train_test.columns=[str(i) for i in train_test.columns]

features = list(np.setdiff1d(train_test.columns, ['Concepts', 'Concepts_Score', 'Keywords', 'Keywords_Score', 'adunit_id', 'description', 'timestamp_creation', 'timestamp_impression', 'difference_impression_creation', 'refrenceUrl', 'title', 'uuid', 'uvh', 'timestamp_creation_formatted', 'timestamp_impression_formatted','impression_id', 'item_id', 'click']))



data = train_test[0:len(train3.index)]

#################################################################################################################################

train_ids = random.sample(list(data.index),int(.8*len(data.index)))
train = data[data.index.isin(train_ids)]
valid = data[~data.index.isin(train_ids)]

X_train=train[features]
y_train=train['click']
X_valid=valid[features]
y_valid=valid['click']

X_test=train_test[features][len(train3.index):len(train_test.index)]

dtrain = xgb.DMatrix(X_train, y_train - 1, missing=np.nan)
dvalid = xgb.DMatrix(X_valid, missing=np.nan) 
dtest = xgb.DMatrix(X_test, missing=np.nan)

nrounds = 1000
watchlist = [(dtrain, 'train')]
params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4,"num_class": 2, "silent": 1,"eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,"min_child_weight": 1,"seed": 2016, "tree_method": "exact"}

bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)

valid_preds = bst.predict(dvalid)
valid_preds = [i+1 for i in valid_preds]

np.sqrt(mean_squared_error(y_valid, valid_preds))

test_preds = bst.predict(dtest)
test_preds = [i+1 for i in test_preds]

###impression_id,item_id,click
pre_submit = pd.DataFrame({'impression_id': train_test['impression_id'][len(train3.index):len(train_test.index)], 'item_id':train_test['item_id'][len(train3.index):len(train_test.index)], 'click': test_preds})

submit = pre_submit.groupby(['impression_id', 'item_id'],as_index=False)['click'].agg(lambda x: x.value_counts().index[0])

submit[['impression_id', 'item_id', 'click']].to_csv("xgb4.csv", index=False)


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
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%f' % float(height),
                ha='center', va='bottom')


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


##FUTURE WORK
##**10) feature exploration, univariate analysis etc. do
##**11) uvh	User view history, last 5 items that the user clicked. :: TRY TO INCLUDE properties of these items
##**12) while joining check multiple records not form
