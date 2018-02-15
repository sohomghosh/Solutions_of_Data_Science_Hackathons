# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb
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
from nltk.tokenize import word_tokenize
from gensim.models import word2vec

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train=pd.read_csv("../input/train.csv",sep=',',header=0)
test=pd.read_csv("../input/test.csv",sep=',',header=0)

#train = train.iloc[0:100,:]
#test = test.iloc[0:100,:]

y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].as_matrix()
xtrain, xvalid, ytrain, yvalid = train_test_split(train['comment_text'].values, y, random_state=42, test_size=0.1, shuffle=True)
xtest = test['comment_text'].values
num_classes = 6
token = text.Tokenizer(num_words=None)
max_len = 150

token.fit_on_texts(list(xtrain) + list(xvalid) + [str(i) for i in list(xtest)])
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)
xtest_seq = token.texts_to_sequences([str(i) for i in list(xtest)])

ytrain_enc = ytrain
yvalid_enc = yvalid

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)
xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)

word_index = token.word_index

train_test = train.append(test)
sentences_split = []
train_test['comment_text'] = [str(i) for i in train_test['comment_text']]
for doc_text in train_test['comment_text']:
	wd_lt = []
	for word in word_tokenize(doc_text):
		if word not in set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
			wd_lt.append(word.lower().strip())
	sentences_split.append(wd_lt)


# create an embedding matrix for the words we have in the dataset
vocab_size = len(word_index) + 1
word_vec_dim = 50
embedding_matrix = np.zeros((vocab_size, word_vec_dim))
word2vec = word2vec.Word2Vec(sentences_split, size=word_vec_dim, min_count =1, window=3, workers =-1,sample=1e-5)

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

model.add(Dense(824, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(724, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback : Early stoping prevents overfiting
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
#checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=8, verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks = [earlystop]) #callbacks = [earlystop, checkpoint]

valid_preds = model.predict(np.array(xvalid_pad))
pred = model.predict(np.array(xtest_pad))
print(log_loss(yvalid_enc, valid_preds))

submission  = pd.DataFrame(pred)
submission ['id'] = test['id']
submission.columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'id']
submission[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].to_csv('sub_w2v_lstm_with_early_stoping.csv', index=False)

# Any results you write to the current directory are saved as output.
