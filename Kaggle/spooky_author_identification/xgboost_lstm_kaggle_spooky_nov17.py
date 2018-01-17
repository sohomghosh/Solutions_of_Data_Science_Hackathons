import pandas as pd
import string
import numpy as np
import re
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import fasttext
import nltk
import enchant
from sklearn.metrics import log_loss
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tag import pos_tag
from collections import Counter
from sklearn.preprocessing import StandardScaler
import keras
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential

from keras.layers import Dense, Activation

from numpy.random import random, normal


train=pd.read_csv('/home/sohom/Desktop/Kaggle_Spooky Author Identification/train.csv')
test=pd.read_csv('/home/sohom/Desktop/Kaggle_Spooky Author Identification/test.csv')

test['author']=np.nan
train_test=train.append(test)
num_classes = len(set(train['author']))
sentences_split=[re.split('\W', i) for i in train_test['text']]

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


#tfidf
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=.01, use_idf=True, smooth_idf=False, sublinear_tf=True)
sklearn_representation = sklearn_tfidf.fit(train_test['text'])
train_test_tfidf=pd.DataFrame(sklearn_tfidf.transform(train_test['text']).todense())


#fasttext
train_test['text'].to_csv('train_test_text.csv',index=False)
model_sk = fasttext.skipgram('train_test_text.csv', 'model_sk',dim=40)
features_sent_ft = np.zeros(shape=(0,40))
for i in sentences_split:
	su=np.zeros(shape=(40))
	num_words = 0
	for j in i:
		k=np.array(model_sk[j])
		su=su+k
		#print(su)
		num_words = num_words + 1
	features_sent_ft=np.vstack([features_sent_ft, su/num_words])



#Keeping w2v, fasttext and tfidf side by side
train_test_features=pd.concat([pd.DataFrame(features_sent),pd.DataFrame(features_sent_ft),train_test_tfidf],axis=1)


#Number of each pos

#number of non english words
#Average Number of words each sentences
#number of stop words
#number of punctations
#nuber of us spelling words
#number of british spellings words
#number of new english words

train_test['num_of_unique_punctuations'] = train_test['text'].apply(lambda x : len(set(x).intersection(set(string.punctuation))))
d_us = enchant.Dict("en_US")
train_test['num_of_american_spelling_words'] = train_test['text'].apply(lambda x: sum([1 for wd in word_tokenize(str(x)) if d_us.check(wd)]))
d_gb = enchant.Dict("en_GB")
train_test['num_of_greatbritain_spelling_words'] = train_test['text'].apply(lambda x: sum([1 for wd in word_tokenize(str(x)) if d_gb.check(wd)]))
stp_wds = set(stopwords.words('english'))
train_test['num_stopwords'] = train_test['text'].apply(lambda x: len(stp_wds.intersection(word_tokenize(str(x)))))
punct_list = list(set(string.punctuation))
train_test['avg_no_words'] = train_test['text'].apply(lambda x: np.nanmean([len([kk for kk in word_tokenize(stn) if kk not in punct_list]) for stn in sent_tokenize(str(x))]))
eng_wds = words.words() + list(set(string.punctuation))
train_test['num_of_non_english_words'] = train_test['text'].apply(lambda x: len([i for i in word_tokenize(str(x)) if i not in eng_wds]))
for snt in train_test['text']:
	di = Counter([j for i,j in pos_tag(word_tokenize(snt))])
	for a in di.keys():
		train_test[str(a)] = di[a]


for i in ['num_of_unique_punctuations', 'num_of_american_spelling_words', 'num_of_greatbritain_spelling_words', 'num_stopwords', 'avg_no_words', 'num_of_non_english_words']:
	train_test_features[i] = train_test[i].values


#train_test_features = pd.concat([train_test_features, train_test[['num_of_unique_punctuations', 'num_of_american_spelling_words', 'num_of_greatbritain_spelling_words', 'num_stopwords', 'avg_no_words', 'num_of_non_english_words']]], axis = 1)

train_test_features.to_csv("train_test_features_v2.csv", index = False)


X_train_all=train_test_features[0:len(train.index)]
X_test=train_test_features[len(train.index):len(train_test_features.index)]
###X_train_all.columns=["feature_"+str(i) for i in range(0,X_train_all.shape[1])]
features=X_train_all.columns
X_train_all['author']=train['author'].replace(to_replace={'EAP':0,'HPL':1,'MWS':2})
###X_test.columns=["feature_"+str(i) for i in range(0,X_test.shape[1])]

X_train=X_train_all.sample(frac=0.80, replace=False)
X_valid=pd.concat([X_train_all, X_train]).drop_duplicates(keep=False)



dtrain = xgb.DMatrix(X_train[features],X_train['author'] , missing=np.nan)
dvalid = xgb.DMatrix(X_valid[features],missing=np.nan)
dtest = xgb.DMatrix(X_test, missing=np.nan)

#eta: 0.08
nrounds = 820
watchlist = [(dtrain, 'train')]
params = {"objective": "multi:softprob","booster": "gbtree", "nthread": 4,"silent": 1,"eta": 0.04, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,"min_child_weight": 1,"seed": 2016, "num_class":3,"tree_method": "exact"}
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
valid_preds = bst.predict(dvalid)
test_preds = bst.predict(dtest)
print(log_loss(X_valid['author'], valid_preds))

#0.5219

#0.510

#'yes' if v == 1 else 'no' if v == 2 else 'idle' 
#submit = pd.DataFrame({'id': test['id'], 'author': ['EAP' if i==0 else 'HPL' if i==1 else 'MWS'  for i in test_preds]})
submit=pd.DataFrame(test_preds,columns=["EAP","HPL","MWS"])
submit['id'] = test['id']
submit[["id","EAP","HPL","MWS"]].to_csv("xgb6.csv", index=False)


######################################################################
########## Using Simple sequential Neural Networks ############
######################################################################
#Reference: https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle

# scale the data before any neural net:
scl = StandardScaler()
X_train_scaled = scl.fit_transform(X_train[features]) #Ideally should have been done on train_test
y_train = keras.utils.to_categorical(X_train['author'], num_classes=3)
X_valid_scaled = scl.transform(X_valid[features])
y_valid = keras.utils.to_categorical(X_valid['author'], num_classes=3)
X_test_scaled = scl.transform(X_test[features])

# create a simple 3 layer sequential neural net
model = Sequential()

#first parameter i.e. 50 in this case is number of neurons in hidden layer
#input_layer_neurons = input_dim = number of columns
model.add(Dense(50, input_dim=X_train[features].shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_train_scaled, y=y_train, batch_size=64, 
          epochs=5, verbose=1, 
          validation_data=(X_valid_scaled, y_valid))
valid_preds = model.predict(np.array(X_valid_scaled))
pred = model.predict(np.array(X_test_scaled))
print(log_loss(y_valid, valid_preds))
#0.49158055325

submission  = pd.DataFrame(pred)
submission ['id'] = test['id']
submission.columns = ['EAP', 'HPL', 'MWS', 'id']
submission[['id', 'EAP', 'HPL', 'MWS']].to_csv('sub_w2v_deep_learning_sequential_nn.csv', index=False)


########################################################################
############## Think from here LSTM MODEL ####
#########################################################################

###Working with only tfidf may be because all values are positive
model = Sequential()
model.add(Embedding(X_train.shape[0], 64,input_length=len(features)))
model.add(LSTM(32, dropout=0.4, recurrent_dropout=0.1))
model.add(Dense(3, activation='sigmoid'))
#model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())

dummy_train_y = np_utils.to_categorical(X_train['author'])

model.fit(np.array(X_train[features]), dummy_train_y, validation_split=0.2 , epochs=4, batch_size=16, verbose=2)
results = model.predict(np.array(X_test[features]))

results = pd.DataFrame(results, columns=['EAP', 'HPL','MWS'])
results.insert(0, "id", list(test['id']))
results.to_csv('lstm1_tfidf_only.csv',index=False)

########################################################################
############## Using LSTM MODEL ####
#########################################################################
###Reference: https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from tqdm import tqdm
from sklearn.model_selection import train_test_split

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author.values)
xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)

xtest = test.text.values
# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 70

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)
xtest_seq = token.texts_to_sequences(xtest)

# we need to binarize the labels for the neural net
ytrain_enc = np_utils.to_categorical(ytrain)
yvalid_enc = np_utils.to_categorical(yvalid)

# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)
xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)

word_index = token.word_index

train_test = train.append(test)
sentences_split = []
for doc_text in train_test['text']:
	wd_lt = []
	for word in word_tokenize(doc_text):
		if word not in set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
			wd_lt.append(word.lower().strip())
	sentences_split.append(wd_lt)		


# create an embedding matrix for the words we have in the dataset
vocab_size = len(word_index) + 1
word_vec_dim = 300
embedding_matrix = np.zeros((vocab_size, word_vec_dim))
word2vec = word2vec.Word2Vec(sentences_split, size=word_vec_dim, min_count =1, window=3, workers =-1,sample=1e-5)

'''
using glove
# load the GloVe vectors in a dictionary:

embeddings_index = {}
f = open('glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
'''


for word, i in tqdm(word_index.items()):
	try:
		embedding_matrix[i] = word2vec.wv[word] #FOR GLOVE# embeddings_index.get(word)
	except KeyError:
		pass


# A simple LSTM with two dense layers
model = Sequential()
model.add(Embedding(vocab_size,
                     word_vec_dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback : Early stoping prevents overfiting
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
#checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks = [earlystop]) #callbacks = [earlystop, checkpoint]

##Without early stopping
##model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, verbose=1, validation_data=(xvalid_pad, yvalid_enc))


valid_preds = model.predict(np.array(xvalid_pad))
pred = model.predict(np.array(xtest_pad))
print(log_loss(yvalid_enc, valid_preds))

submission  = pd.DataFrame(pred)
submission ['id'] = test['id']
submission.columns = ['EAP', 'HPL', 'MWS', 'id']
submission[['id', 'EAP', 'HPL', 'MWS']].to_csv('sub_w2v_lstm_with_early_stoping.csv', index=False)



##########################################################################################
######################### Bi-directional LSTM ############################################
##########################################################################################
# A simple bidirectional LSTM with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(vocab_size,
                     word_vec_dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(350, dropout=0.3, recurrent_dropout=0.3)))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])

valid_preds = model.predict(np.array(xvalid_pad))
pred = model.predict(np.array(xtest_pad))
print(log_loss(yvalid_enc, valid_preds))

submission  = pd.DataFrame(pred)
submission ['id'] = test['id']
submission.columns = ['EAP', 'HPL', 'MWS', 'id']
submission[['id', 'EAP', 'HPL', 'MWS']].to_csv('sub_w2v_bilstm_with_early_stoping.csv', index=False)


########################################################################
###################### GRU #############################################
########################################################################
# GRU and two dense layers
model = Sequential()
model.add(Embedding(vocab_size,
                     word_vec_dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])

valid_preds = model.predict(np.array(xvalid_pad))
pred = model.predict(np.array(xtest_pad))
print(log_loss(yvalid_enc, valid_preds))

submission  = pd.DataFrame(pred)
submission ['id'] = test['id']
submission.columns = ['EAP', 'HPL', 'MWS', 'id']
submission[['id', 'EAP', 'HPL', 'MWS']].to_csv('sub_w2v_gru_with_early_stoping.csv', index=False)


#######################################################################
############# Adding other features to Embeddings #####################
#######################################################################
#Reference: https://www.kaggle.com/knowledgegrappler/embeddings-features-tdf-idf-let-s-party
'''
#KERAS MODEL DEFINITION
from keras.layers import Dense, Dropout, Embedding
from keras.layers import Flatten, Input, SpatialDropout1D, Concatenate
from keras.models import Model
from keras.optimizers import Adam 
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def get_model():
    embed_dim = 30
    dropout_rate = 0.9
    emb_dropout_rate = 0.9
   
    input_text = Input(shape=[maxlen], name="stem_input")
    
    input_num = Input(shape=[X_train["num_input"].shape[1]], name="num_input") #Consists of numeric features like no. of stop words, length of sentences etc.
    
    input_svd = Input(shape=[X_train["svd_vect"].shape[1]], name="svd_vect") #Consists of 
    
    emb_lstm = SpatialDropout1D(emb_dropout_rate) (Embedding(n_stem_seq, embed_dim
                                                ,input_length = maxlen
                                                               ) (input_text)) #Like other lstm models creating embeddings
    concatenate = Concatenate()([(Flatten() (emb_lstm)), input_num, input_svd])
    dense = Dropout(dropout_rate) (Dense(256) (concatenate))
    
    output = Dense(3, activation="softmax")(dense)

    model = Model([input_text, input_num, input_svd], output)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

model = get_model()
model.summary()

#TRAIN KERAS MODEL
file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)

model = get_model()
model.fit(X_train, y_train, epochs=150
          , validation_data=[X_valid, y_valid]
         , batch_size=512
         , callbacks = callbacks)

#MODEL EVALUATION
from sklearn.metrics import log_loss

model = get_model()
model.load_weights(file_path)

preds_train = model.predict(X_train)
preds_valid = model.predict(X_valid)

print(log_loss(y_train, preds_train))
print(log_loss(y_valid, preds_valid))

#PREDICTION
preds = pd.DataFrame(model.predict(X_test), columns=target_vars)
submission = pd.concat([test["id"],preds], 1)
submission.to_csv("./submission.csv", index=False)
submission.head()

'''

########################################################################
###Other features : 
#Reference: https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author
#Number of words in the text
#Number of unique words in the text
#Number of characters in the text
# Number of stopwords
# Number of punctuations
# Number of upper case words
# Number of title case words
# Average length of the words
#Character wise tfidf
#Other features as described in: https://www.kaggle.com/c/quora-question-pairs/discussion/34355       Structural features - by creating graphs from texts We built density features from the graph built from the edges between pairs of questions inside train and test datasets concatenated. We had counts of neighbors of question 1, question 2, the min, the max, intersections, unions, shortest path length when main edge cut....


#SVD on tfidf
'''
n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
'''


#sent2vec : from word2vec
'''
#M is the array made per word of a sentence
#V is the array representing the sentence
#Returning normalized vector per sentence
 M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())
'''


#########################################################################
############## Think from here CNN MODEL ###############################
#########################################################################

#Reference: stackoverfow.com
'''
Embedding layer is a simple matrix multiplication that transforms words into their corresponding word embeddings.

The weights of the Embedding layer are of the shape (vocabulary_size, embedding_dimension). For each training sample, its input are integers, which represent certain words. The integers are in the range of the vocabulary size. The Embedding layer transforms each integer i into the ith line of the embedding weights matrix.

In order to quickly do this as a matrix multiplication, the input integers are not stored as a list of integers but as a one-hot matrix. Therefore the input shape is (nb_words, vocabulary_size) with one non-zero value per line. If you multiply this by the embedding weights, you get the output in the shape

(nb_words, vocab_size) x (vocab_size, embedding_dim) = (nb_words, embedding_dim)
So with a simple matrix multiplication you transform all the words in a sample into the corresponding word embeddings.
'''
#Reference: https://www.kaggle.com/kamalkishor1991/basic-python-cnn-with-embeddings-42/notebook
#Reference: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

from nltk.tokenize import word_tokenize
import keras
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras import regularizers
from sklearn.metrics import log_loss
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
import pandas as pd
from numpy.random import random, normal
from gensim.models import word2vec


train=pd.read_csv('/home/sohom/Desktop/Kaggle_Spooky Author Identification/train.csv')
test=pd.read_csv('/home/sohom/Desktop/Kaggle_Spooky Author Identification/test.csv')

train_test=train.append(test)

all_words = list(set(word.lower().strip() for doc_text in train_test['text'] for word in word_tokenize(doc_text) if word not in set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')))

#maximum number of words in a text i.e. in a row
feature_len = max([len(word_tokenize(doc_text)) for doc_text in train_test['text']])

#creating dictonary of words with their encoded values
#x_train_test = [one_hot(doc_text, feature_len) for doc_text in train_test['text']]



vocab_size = len(all_words) + 1
word2ids = {}
for index, word in enumerate(all_words):
    word2ids[word] = index + 1


y_train_all = keras.utils.to_categorical(train['author'].replace(to_replace={'EAP':0,'HPL':1,'MWS':2}), num_classes=3)

x_train_test = []

for doc_text in train_test['text']:
	x_train_test.append([word2ids[word.lower().strip()] for word in word_tokenize(doc_text) if word not in set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')])


x_train_test_padded = pad_sequences(x_train_test, maxlen=feature_len, padding='post')

x_train_all = x_train_test_padded[0:len(train)]
x_test = x_train_test_padded[len(train):]


#validation set
msk = np.random.rand(len(x_train_all)) < 0.8
x_train_all = np.array(x_train_all)
y_train_all = np.array(y_train_all)
x_valid = x_train_all[~msk]
y_valid = y_train_all[~msk]
x_train = x_train_all[msk]
y_train = y_train_all[msk]

word_vec_dim = 40

sentences_split = []
for doc_text in train_test['text']:
	wd_lt = []
	for word in word_tokenize(doc_text):
		if word not in set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
			wd_lt.append(word.lower().strip())
	sentences_split.append(wd_lt)		


############################ word2vec ########################################################################
#create embedding_matrix if dimension vocab_size X word_vec_dim having each unique word in each row
embedding_matrix = np.zeros((vocab_size, word_vec_dim))

word2vec = word2vec.Word2Vec(sentences_split, size=word_vec_dim, min_count =1, window=3, workers =-1,sample=1e-5)

for word,idx in word2ids.items():
    if word in word2vec.wv:
        embedding_matrix[idx] = word2vec.wv[word]
    else:
        embedding_matrix[idx] = normal(scale=0.6, size=(word_vec_dim,))


model = Sequential([
    Embedding(vocab_size, word_vec_dim, input_length=feature_len, weights=[embedding_matrix], trainable=True),
    Dropout(0.5),
    Conv1D(word_vec_dim, 7, border_mode='same', activation='relu'),
    Dropout(0.5),
    Flatten(),
    BatchNormalization(),
    Dense(40, activation='relu'),
    Dropout(0.6),
    Dense(3, activation='softmax')])


model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()
model.fit(np.array(x_train), y_train, validation_data=(np.array(x_valid),y_valid), nb_epoch=3, batch_size=64)
valid_preds = model.predict(np.array(x_valid))
pred = model.predict(np.array(x_test))
print(log_loss(y_valid, valid_preds))
#0.49158055325

submission  = pd.DataFrame(pred)
submission ['id'] = test['id']
submission.columns = ['EAP', 'HPL', 'MWS', 'id']
submission[['id', 'EAP', 'HPL', 'MWS']].to_csv('sub_w2v.csv', index=False)


############################ fasttext ########################################################################
model_sk = fasttext.skipgram('train_test_text.csv', 'model_sk',dim=word_vec_dim)
#model_sk = fasttext.load_model('model_sk.bin')

model_sk_words_cleaned = list(set([wd.strip().lower().replace('"','').replace('.','').replace(',','').replace(':','').replace(';','') for wd in list(model_sk.words)]))


for word,idx in word2ids.items():
    if word in model_sk_words_cleaned:
        embedding_matrix[idx] = model_sk[word]
    else:
        embedding_matrix[idx] = normal(scale=0.6, size=(word_vec_dim,))


model = Sequential([
    Embedding(vocab_size, word_vec_dim, input_length=feature_len, weights=[embedding_matrix], trainable=True),
    Dropout(0.5),
    Conv1D(word_vec_dim, 9, border_mode='same', activation='relu'),
    Dropout(0.6),
    Flatten(),
    BatchNormalization(),
    Dense(50, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')])


model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()
model.fit(np.array(x_train), y_train, validation_data=(np.array(x_valid),y_valid), nb_epoch=3, batch_size=64)
valid_preds = model.predict(np.array(x_valid))
pred = model.predict(np.array(x_test))
print(log_loss(y_valid, valid_preds))

submission  = pd.DataFrame(pred)
submission ['id'] = test['id']
submission.columns = ['EAP', 'HPL', 'MWS', 'id']
submission[['id', 'EAP', 'HPL', 'MWS']].to_csv('sub_ft.csv', index=False)


#################### Think how additional features like #no. of stop words etc. can be given to this model
'''
train_test_features = pd.read_csv("train_test_features_v2.csv")
features = train_test_features[['num_of_unique_punctuations', 'num_of_american_spelling_words', 'num_of_greatbritain_spelling_words', 'num_stopwords', 'avg_no_words', 'num_of_non_english_words']]
other_features = Input(shape=(features.shape[0],))
#Refer : https://www.kaggle.com/knowledgegrappler/embeddings-features-tdf-idf-let-s-party

'''

###################### GRU ###################
#Rhttps://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggleeference: 
# GRU with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')



################################################################################

#ENSEMBLE
#Make 3 classes for 3 authors; predict probab of each author using different algo #Max probab assign
#Use doc2vec
#SEE Approach of others



####################### Others ###############
#Quora duplicate question pair detection: Link: https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
'''
########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
#model.summary()
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train], labels_train, \
        validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
        epochs=200, batch_size=2048, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
'''

'''
#Quora duplicate question SRK's approach
Reference: https://www.kaggle.com/sudalairajkumar/keras-starter-script-with-word-embeddings
# Model Architecture #
sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = Conv1D(128, 3, activation='relu')(embedded_sequences_1)
x1 = MaxPooling1D(10)(x1)
x1 = Flatten()(x1)
x1 = Dense(64, activation='relu')(x1)
x1 = Dropout(0.2)(x1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = Conv1D(128, 3, activation='relu')(embedded_sequences_2)
y1 = MaxPooling1D(10)(y1)
y1 = Flatten()(y1)
y1 = Dense(64, activation='relu')(y1)
y1 = Dropout(0.2)(y1)

merged = merge([x1,y1], mode='concat')
merged = BatchNormalization()(merged)
merged = Dense(64, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = BatchNormalization()(merged)
preds = Dense(1, activation='sigmoid')(merged)
model = Model(input=[sequence_1_input,sequence_2_input], output=preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])



'''

'''
####### opening files ith codecs
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(values[3])
        texts_2.append(values[4])
        labels.append(int(values[5]))
'''


