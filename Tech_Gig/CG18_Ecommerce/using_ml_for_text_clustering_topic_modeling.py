import pandas as pd
import numpy as np
import re
from gensim.models import word2vec

df = pd.read_csv("tg_ecomm_df.csv")
df.columns = ['new_link', 'id', 'item_group_id', 'title', 'link', 'description', 'google_product_category', 'l2_category', 'product_type', 'image_link', 'condition', 'size', 'color', 'availability', 'price', 'brand', 'gender', 'shipping', 'sale_price', 'totaldiscount', 'pattern', 'adult', 'custom_label_3', 'gtin', 'custom_label_2', 'custom_label_4', 'material']

del df['link']
df = df.iloc[6:,]
gc.collect()
df.head(2)

df['sale_price'] = df['sale_price'].apply(lambda x : int(str(x)[:-4]))
df['totaldiscount'] = df['totaldiscount'].apply(lambda x : int(str(x)[:-4]))
df['price'] = df['price'].apply(lambda x : int(str(x)[:-4]))

df = df.fillna('')
df['all'] = df['title'] + ' ' + df['description'] + ' ' + df['product_type'] + ' ' + df['size'] + ' ' + df['color'] + ' ' + df['brand'] + ' ' + df['gender'] + ' ' + df['pattern'] + ' ' + df['material']

sentences_split=[re.split('\W', i) for i in df['all']]
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
    
np.savetxt('features_sent.txt', features_sent) 


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=30)
kmeans.fit(features_sent)#Remember X : NEEDS TO BE SCALED
y_kmeans = kmeans.predict(features_sent)#Remember X : NEEDS TO BE SCALED


open("clusters.txt",'w').write(str(list(y_kmeans)))


#index.
dictionary = corpora.Dictionary(sentences_split)
# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in sentences_split]


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=50, id2word = dictionary, passes=50)

topics = []
for dd in sentences_split:
	bow_vector = dictionary.doc2bow(dd)
	lis=ldamodel.get_document_topics(bow_vector, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
	mx = max([b for (a,b) in lis])
	for (a,b) in lis:
		if b == mx:
			topics.append(a)
			break;

open('topics_from_lda.txt','w').write(str(topics))