
# coding: utf-8

# In[15]:


import pandas as pd
from sklearn.metrics import roc_auc_score, auc,roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
import numpy as np
import xgboost as xgb


# In[2]:


train = pd.read_csv("train.csv")
campaign_data = pd.read_csv("campaign_data.csv")
test = pd.read_csv("test.csv")


# In[3]:


train_full = train.merge(campaign_data, on = 'campaign_id', how = 'inner')
test_full = test.merge(campaign_data, on = 'campaign_id', how = 'inner')
train_test = train_full.append(test_full)


# In[4]:


one_hot_encoded_train_test = pd.get_dummies(train_test[['communication_type']])

train_test['day_of_month'] =  train_test['send_date'].apply(lambda x: int(str(x).split('-')[0]))
train_test['hour_of_day'] = train_test['send_date'].apply(lambda x: int(str(x)[str(x).find(' ')+1:str(x).find(':')]))
train_test['min_of_hour'] = train_test['send_date'].apply(lambda x: int(str(x)[str(x).find(':')+1:len(str(x))]))


# In[9]:


one_hot_encoded_train_test.head()


# In[5]:


sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=.01, use_idf=True, smooth_idf=False, sublinear_tf=True)
train_test_tfidf_sub = pd.DataFrame(sklearn_tfidf.fit_transform(train_test['subject']).todense())
train_test_tfidf_email = pd.DataFrame(sklearn_tfidf.fit_transform(train_test['email_body']).todense())


# In[9]:


np.savetxt('train_test_tfidf_sub.txt',train_test_tfidf_sub)


# In[5]:


train_test_tfidf_sub = np.loadtxt("train_test_tfidf_sub.txt")
train_test_tfidf_sub


# In[6]:


train_test_tfidf_sub = pd.DataFrame(train_test_tfidf_sub)


# In[10]:


np.savetxt('train_test_tfidf_email.txt',train_test_tfidf_email)


# In[ ]:


sentences_split=[word_tokenize(i) for i in train_test['email_body']]

#wordvec
model_w2v = word2vec.Word2Vec(sentences_split, size=40,min_count =1, window=3, workers =-1,sample=1e-5)
features_sent = np.zeros(shape=(0,40))
for i in sentences_split:
	su=np.zeros(shape=(40))
	num_words = 0
	for j in i:
		k=np.array(model_w2v.wv[j])
		su=su+k
		#print(su)
		num_words = num_words + 1
	features_sent=np.vstack([features_sent, su/num_words])


# In[ ]:


sentences_split=[word_tokenize(i) for i in train_test['subject']]

#wordvec
model_w2v = word2vec.Word2Vec(sentences_split, size=40,min_count =1, window=3, workers =-1,sample=1e-5)
features_sent_sub = np.zeros(shape=(0,40))
for i in sentences_split:
	su=np.zeros(shape=(40))
	num_words = 0
	for j in i:
		k=np.array(model_w2v.wv[j])
		su=su+k
		#print(su)
		num_words = num_words + 1
	features_sent_sub=np.vstack([features_sent_sub, su/num_words])


# In[14]:


#train_test_use = pd.concat([train_test[['no_of_images', 'no_of_internal_links', 'no_of_sections', 'total_links','day_of_month','hour_of_day','min_of_hour','is_click']],one_hot_encoded_train_test,train_test_tfidf_sub,train_test_tfidf_email,pd.DataFrame(features_sent),pd.DataFrame(features_sent_sub)], axis=1)
train_test_use = pd.concat([train_test[['no_of_images', 'no_of_internal_links', 'no_of_sections', 'total_links','day_of_month','hour_of_day','min_of_hour','is_click']].reset_index(),one_hot_encoded_train_test.reset_index(),train_test_tfidf_sub.reset_index()], axis=1)
train_test_use.to_csv("train_test_use.csv", index = False)




# In[29]:


X_train_all=train_test_use[0:len(train.index)]
X_test=train_test_use[len(train.index):len(train_test_use.index)]


# In[34]:


train_full.shape


# In[16]:


X_train_all.columns


# In[47]:


X_train_all.columns = [str(i) for i in X_train_all.columns]
X_test.columns = [str(i) for i in X_test.columns]
del X_train_all['index']
del X_test['index']
features=list(X_train_all.columns)
features.remove('is_click')
X_train=X_train_all.sample(frac=0.80, replace=False)
X_valid=pd.concat([X_train_all, X_train]).drop_duplicates(keep=False)

dtrain = xgb.DMatrix(X_train[features],X_train['is_click'] , missing=np.nan)
dvalid = xgb.DMatrix(X_valid[features],missing=np.nan)
dtest = xgb.DMatrix(X_test[features], missing=np.nan)


# In[42]:


X_train[features].head()


# In[41]:


X_train['is_click'].shape


# In[22]:


X_train.head()


# In[23]:


i=1


# In[28]:


X_train['is_click'].shape


# In[43]:


dtrain


# In[44]:


nrounds = 820
watchlist = [(dtrain, 'train')]
params = {"objective": "binary:logistic","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1,
                "seed": 2016, "tree_method": "exact"}
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)




# In[51]:


valid_preds = bst.predict(dvalid)
test_preds = bst.predict(dtest)
roc_auc_score(X_valid['is_click'], valid_preds)


# In[52]:


test_preds


# In[49]:


features


# In[53]:


sub = pd.DataFrame({'id':test['id'],'is_click':test_preds})
sub[['id','is_click']].to_csv('sub_'+str(i)+'.csv', index = False)
i = i+1


# In[54]:


sub.head()

