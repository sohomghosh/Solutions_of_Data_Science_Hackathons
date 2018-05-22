#To run:
#python tg_cg18_ecomm_final.py
#In browser  type http://0.0.0.0:5000/
#Enter search query in the text field and press GO

# coding: utf-8

# In[85]:


from bs4 import BeautifulSoup
import pandas as pd
import re
import gc
import difflib
import warnings
#warnings.filterwarnings('ignore')
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from fuzzywuzzy import fuzz

app = Flask(__name__)
CORS(app, support_credentials=True)



# In[3]:


tags = ['id', 'g:item_group_id', 'title', 'link', 'description', 'g:google_product_category', 'g:l2_category', 'g:product_type', 'g:image_link', 'g:condition', 'g:size', 'g:color', 'g:availability', 'g:price', 'g:brand', 'g:gender', 'g:shipping', 'g:sale_price', 'g:totaldiscount', 'g:pattern', 'g:adult', 'g:custom_label_3', 'g:gtin', 'g:custom_label_2', 'g:custom_label_4', 'g:material']

## creating structured csv file from unstructured xml file Refer: converting_xml_to_csv.py

# # Using the structured file created

# In[48]:


df = pd.read_csv("tg_ecomm_df.csv")
df.columns = ['new_link', 'id', 'item_group_id', 'title', 'link', 'description', 'google_product_category', 'l2_category', 'product_type', 'image_link', 'condition', 'size', 'color', 'availability', 'price', 'brand', 'gender', 'shipping', 'sale_price', 'totaldiscount', 'pattern', 'adult', 'custom_label_3', 'gtin', 'custom_label_2', 'custom_label_4', 'material']


# In[49]:


del df['link']
df = df.iloc[6:,]
gc.collect()
df.head(2)


# In[50]:


df['sale_price'] = df['sale_price'].apply(lambda x : int(str(x)[:-4]))
df['totaldiscount'] = df['totaldiscount'].apply(lambda x : int(str(x)[:-4]))
df['price'] = df['price'].apply(lambda x : int(str(x)[:-4]))

# In[68]:


df = df.fillna('')


# In[69]:


df['all'] = df['title'] + ' ' + df['description'] + ' ' + df['product_type'] + ' ' + df['size'] + ' ' + df['color'] + ' ' + df['brand'] + ' ' + df['gender'] + ' ' + df['pattern'] + ' ' + df['material']

#Using the clsuters and topics file created by using_ml_for_text_clustering_topic_modeling.py
df['cluster'] = [str(i) for i in eval(open("clusters.txt",'r').read())]
df['topic'] = [str(i) for i in eval(open("topics_from_lda.txt",'r').read())]

# # Intent Detection

# In[51]:


def page_query_detect(text):
    op = ''
    if 'wish' in text.lower():
        op = 'Page Query redirect to webpage :- Wishlist'
    if 'deliver' in text.lower():
        op = 'Page Query redirect to webpage :- Delivery'
    if 'satus' in text.lower():
        op = 'Page Query redirect to webpage :- Delivery'
    if 'bring' in text.lower():
        op = 'Page Query redirect to webpage :- Delivery'
    if 'hand over' in text.lower():
        op = 'Page Query redirect to webpage :- Delivery'
    if 'exchange' in text.lower():
        op = 'Page Query redirect to webpage :- Exchange'
    if 'change' in text.lower():
        op = 'Page Query redirect to webpage :- Exchange'
    return op


# In[52]:


def search_by_id_detect(text):
    op = ''
    #p-<followed by 5 to 9 digits>
    if re.search(r'p-[0-9]{5,9}', text):
        op = text
        return op
    #5 to 9 digits such that lasttwo digits are not '00'; generally last 2 digits 00 signifies price
    if re.search(r'[0-9]{5,9}', text) and text[-2:]!='00':
        op = 'p-'+text
        return op
    return op


# In[53]:


def search_by_category(text):
    related_terms = []
    for tk in text.lower().strip().split():
        if tk in list(df['l2_category'].str.lower().unique()) and tk not in related_terms:
            related_terms.append(tk)
        else:    
            for i in difflib.get_close_matches(tk,[str(i).lower().strip() for i in list(df['l2_category'].unique())],5,.6):
                if i not in related_terms:
                    related_terms.append(i)
    return related_terms


# In[55]:


def search_by_product_type(text):
    related_terms = []
    for tk in text.lower().strip().split():
        if tk in list(df['product_type'].str.lower().unique()) and tk not in related_terms:
            related_terms.append(tk)
        else:    
            for i in difflib.get_close_matches(tk,[str(i).lower().strip() for i in list(df['product_type'].str.lower().unique())],5,.8):
                if i not in related_terms:
                    related_terms.append(i)
    return related_terms


# In[58]:


def search_by_condition(text):
    if 'new' in text:
        return 'new'
    return ''


# In[59]:


def search_by_gender(text):
    for tk in text.lower().strip().split():
        if tk in ['male', 'man', 'gents',  'he', 'him', 'mens', 'men', "men's"]:
            return 'male'
        if tk in ['female', 'woman', 'ladies', 'lady', 'she', 'her', 'women', 'womens', "women's"]:
            return 'female'
    return ''


# In[64]:


def search_by_amt(text):#price, sale_price, totaldiscount
    if 'cost' in text or'less' in text or 'within' or 'greater' in text or 'between' in text or 'under' in text or 'below' in text or 'above' in text or 'more' in text or 'lesser' or 'price' or 'inr' or 'rupees':
        return [int(s) for s in text.split() if s.isdigit()]
    else:
        return []

def search_by_gt(text):
    if 'greater' in text or 'more' in text or 'above':
        return 'more'
    return ''

def search_by_lt(text):
    if 'less' in text or 'under' in text or 'below' in text or 'lesser' in text or 'within':
        return 'less'
    return ''
    
def detect_price_discount_sale(text):
    if 'price' in text or 'cost' in text or 'fare' in text or 'rate' in text or 'value' in text:
        return 'price'
    elif 'discount' in text or 'rebate' in text or 'concession' in text:
        return 'totaldiscount'
    elif 'net' in text or 'sale price' in text or 'sale' in text:
        return 'sale_price'
    else:
        return ''
    return ''


# In[65]:


def search_by_attr(text,attr):
    terms = []
    for tk in text.lower().strip().split():
        if tk.lower().strip() in [str(i).lower().strip() for i in  list(set(df[attr]))]:
            terms.append(tk.lower().strip())
    return terms


# In[66]:


def input_text_to_intent(text, dtf):
    text = str(text).lower().strip()
    dt = dtf
    relaxed_ids = set([])
    ans = page_query_detect(text)
    if ans != '':
        return [pd.DataFrame({}), [ans]]
    else:
        typ = []
        
        #Exact match
        id_prod = search_by_id_detect(text)
        if id_prod != '':
            dtf = dtf[dtf['id'] == id_prod]
            typ.append('search by product id: ' + str(id_prod))
            relaxed_ids = relaxed_ids | set(dt[dt['id'] == id_prod]['id'])
        
        contd = search_by_condition(text)
        if contd != '':
            dtf = dtf[dtf['condition'] == contd]
            typ.append('search by condition: ' + str(contd))
            relaxed_ids = relaxed_ids | set(dt[dt['condition'] == contd]['id'])
        
        gender = search_by_gender(text)
        if gender != '':
            dtf = dtf[dtf['gender'] == gender]
            typ.append('search by gender: ' + str(gender))
            relaxed_ids = relaxed_ids | set(dt[dt['gender'] == gender]['id'])
        
        #Characteristics search
        size = search_by_attr(text,'size')
        if len(size) != 0:
            dtf = dtf[dtf['size'].str.lower().isin(size)]
            typ.append('search by size: ' + str(size))
            relaxed_ids = relaxed_ids | set(dt[dt['size'].str.lower().isin(size)]['id'])
            
        color = search_by_attr(text,'color')
        if len(color) != 0:
            dtf = dtf[dtf['color'].str.lower().isin(color)]
            typ.append('search by color: ' + str(color))
            relaxed_ids = relaxed_ids | set(dt[dt['color'].str.lower().isin(color)]['id'])
            
        avail = search_by_attr(text, 'availability')
        if len(avail) != 0:
            dtf = dtf[dtf['availability'].str.lower().isin(avail)]
            typ.append('search_by_availability: ' + str(avail))
            relaxed_ids = relaxed_ids | set(dt[dt['availability'].str.lower().isin(avail)]['id'])
            
        brand = search_by_attr(text, 'brand')
        if len(brand) != 0:
            dtf = dtf[dtf['brand'].str.lower().isin(brand)]
            typ.append('search_by_brand: ' + str(brand))
            relaxed_ids = relaxed_ids | set(dt[dt['brand'].str.lower().isin(brand)]['id'])
        
        pattern = search_by_attr(text,'pattern')
        if len(pattern) != 0:
            dtf = dtf[dtf['pattern'].str.lower().isin(pattern)]
            typ.append('search_by_pattern: ' + str(pattern))
            relaxed_ids = relaxed_ids | set(dt[dt['pattern'].str.lower().isin(pattern)]['id'])
        
        cus_label = search_by_attr(text, 'custom_label_2')
        if len(cus_label) != 0:
            dtf = dtf[dtf['custom_label_2'].str.lower().isin(cus_label)]
            typ.append('search_by_custom_label: ' + str(cus_label))
            relaxed_ids = relaxed_ids | set(dt[dt['custom_label_2'].str.lower().isin(cus_label)]['id'])
                               
        material = search_by_attr(text, 'material')
        if len(material) != 0:
            dtf = dtf[dtf['material'].str.lower().isin(material)]
            typ.append('search_by_material: ' + str(material))
            relaxed_ids = relaxed_ids | set(dt[dt['material'].str.lower().isin(material)]['id'])
        
        #Fuzzy match
        cat = search_by_category(text)
        if len(cat) != 0:
            dtf = dtf[dtf['l2_category'].str.lower().isin(cat)]
            typ.append('search_by_similar_categories: ' + str(cat))
            relaxed_ids = relaxed_ids | set(dt[dt['l2_category'].str.lower().isin(cat)]['id'])
            
        prod = search_by_product_type(text)
        if len(prod) != 0:
            dtf = dtf[dtf['product_type'].str.lower().isin(prod)]
            typ.append('search_by_similar_product_type: ' + str(prod))
            relaxed_ids = relaxed_ids | set(dt[dt['product_type'].str.lower().isin(prod)]['id'])
        
        #Price search
        param_to_search = detect_price_discount_sale(text)
        amt = search_by_amt(text)
        gt = search_by_gt(text)
        lt = search_by_lt(text)
        if len(amt) == 2 and gt != '' and lt != '':
            mini = min(amt)
            maxi = max(amt)
            dtf = dtf[(dtf[param_to_search]>=mini) & (dtf[param_to_search]<=maxi)]
            typ.append('search_by_amount with min max')
            relaxed_ids = relaxed_ids | set(dt[(dt[param_to_search]>=mini) & (dt[param_to_search]<=maxi)]['id'])
        elif gt != '' and lt == '':
            dtf = dtf[dtf[param_to_search]>=min(amt)]
            typ.append('search_by_amount with min')
            relaxed_ids = relaxed_ids | set(dt[dt[param_to_search]>=min(amt)]['id'])
        elif gt == '' and lt != '':
            dtf = dtf[dtf[param_to_search]<=max(amt)]
            typ.append('search_by_amount with max')
            relaxed_ids = relaxed_ids | set(dt[dt[param_to_search]<=max(amt)]['id'])
        elif len(amt)!=0:
            mini = min(amt) - .2*min(amt)
            maxi = max(amt)
            dtf = dtf[(dtf['price']>=mini) & (dtf['price']<=maxi)]
            typ.append('search_by_amount')
            relaxed_ids = relaxed_ids | set(df[(df['price']>=mini) & (df['price']<=maxi)]['id'])
        else:
            pass
        #ranking based on entire document
        #rank_df(dtf) and list(relaxed_ids) and text
    #print(list(relaxed_ids)[:10])
    relaxed_ids = relaxed_ids - set(list(dtf['id']))
    final_df = relevancy_matrix(text,dtf,relaxed_ids)
    
    return [final_df.drop_duplicates(),typ]


# # Recommendation - Relevancy Matrix construction for Ranking

# In[67]:




# In[70]:


def score_increase(row, li, param):
    new_score = int(row['score'])
    if row[param].lower() in li:
        new_score = new_score + 10
    return new_score


# In[107]:


###Convert all texts/documents to vectors, find the documents which is nearest to the query vector using euclidean distance or cosine similarity 
#given a text and a matrix generate rank
def relevancy_matrix(text,dtf,relaxed_ids):
    dtf = dtf.fillna('')
    dtf['all'] = dtf['title'].str.lower() + ' ' + dtf['description'].str.lower() + ' ' + dtf['product_type'].str.lower() + ' ' + dtf['size'].str.lower() + ' ' + dtf['color'].str.lower() + ' ' + dtf['brand'].str.lower() + ' ' + dtf['gender'].str.lower() + ' ' + dtf['pattern'].str.lower() + ' ' + dtf['material'].str.lower()
    dtf['score'] = dtf['all'].apply(lambda x: fuzz.ratio(str(x), text))
    cat = search_by_category(text)
    if len(cat) != 0 and dtf.shape[0] != 0:
        dtf['score'] = dtf.apply(lambda x:score_increase(x,cat,'l2_category'), axis=1) 
    
    cus_label = search_by_attr(text, 'custom_label_2')
    if len(cus_label)!= 0 and dtf.shape[0] != 0:
        dtf['score'] = dtf.apply(lambda x:score_increase(x,cus_label,'custom_label_2'), axis=1) 
    
    related_topics = list(dtf[dtf['score']==dtf['score'].max()]['topic'].unique())
    if len(related_topics) !=0 and dtf.shape[0] != 0:
        dtf['score'] = dtf.apply(lambda x:score_increase(x,related_topics,'topic'), axis=1)
    
    related_clusters = list(dtf[dtf['score']==dtf['score'].max()]['cluster'].unique())
    if len(related_clusters) !=0 and dtf.shape[0] != 0:
        dtf['score'] = dtf.apply(lambda x:score_increase(x,related_clusters,'cluster'), axis=1)
    
    dtf = dtf.sort_values(['score'], ascending = False)
    
    rlx_dtf = df[df['id'].isin(relaxed_ids)]
    rlx_dtf['all'] = rlx_dtf['title'] + ' ' + rlx_dtf['description'] + ' ' + rlx_dtf['product_type'] + ' ' + rlx_dtf['size'] + ' ' + rlx_dtf['color'] + ' ' + rlx_dtf['brand'] + ' ' + rlx_dtf['gender'] + ' ' + rlx_dtf['pattern'] + ' ' + rlx_dtf['material']
    rlx_dtf['score'] = rlx_dtf['all'].apply(lambda x: fuzz.ratio(str(x), text))
    if len(cat) != 0 and rlx_dtf.shape[0] != 0:
      rlx_dtf['score'] = rlx_dtf.apply(lambda x:score_increase(x,cat,'l2_category'), axis=1) 
    
    if len(cus_label) != 0 and rlx_dtf.shape[0] != 0:
        rlx_dtf['score'] = rlx_dtf.apply(lambda x:score_increase(x,cus_label,'custom_label_2'), axis=1) 
    
    if len(related_topics) !=0 and rlx_dtf.shape[0] != 0:
        rlx_dtf['score'] = rlx_dtf.apply(lambda x:score_increase(x,related_topics,'topic'), axis=1)

    if len(related_clusters) !=0 and rlx_dtf.shape[0] != 0:
        rlx_dtf['score'] = rlx_dtf.apply(lambda x:score_increase(x,related_clusters,'cluster'), axis=1)
    
    rlx_dtf = rlx_dtf.sort_values(['score'], ascending = False)
    
    return dtf.append(rlx_dtf)


# # Sample Input and Output
@cross_origin(supports_credentials=True)
@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    ans = input_text_to_intent(text.lower(), df)
    return str({'parsed intents' : str(ans[1]), 'dataframe' : str(ans[0].to_json(orient='table'))})
 

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0",debug=True)



# In[72]:


#input_text_to_intent('my wishlist', df)


# In[98]:


#input_text_to_intent('t-shirt within rs 500',df)[0][['id','title','sale_price']].head()


# In[99]:


#input_text_to_intent('t-shirt within rs 500',df)[1]


# In[100]:


#input_text_to_intent('Denim Shirts',df)[0][['id', 'title', 'product_type']].head()


# In[101]:


#input_text_to_intent('Denim Shirts',df)[1]


# In[108]:


#input_text_to_intent('Mens Leather Shoes',df)[0][['id','title','l2_category', 'material']].head()


# In[109]:


#input_text_to_intent('Mens Leather Shoes',df)[1]


# In[110]:


#input_text_to_intent('Shirts less than 500', df)[0][['id', 'title', 'price']].head()


# In[105]:


#input_text_to_intent('Shirts less than 500', df)[0].head(2)


# In[106]:


#input_text_to_intent('Shirts less than 500', df)[1]


# In[111]:


#input_text_to_intent('My delivery', df)


# In[112]:


#input_text_to_intent('Exchange', df)


# In[113]:


#input_text_to_intent('Elizabeth Arden Floral fragrance', df)[0][['new_link','id','title']].head(2)


# In[114]:


#input_text_to_intent('Elizabeth Arden Floral fragrance', df)[1]


# # Others

# In[ ]:


#########More Weightage#########
#Related items by text matching 'title', 'description', 'product_type', 'size', 'color', 'brand', 'gender', 'pattern', 'material
#Document clustering by description
#Items in same group l2_category
#google_product_category
#custom_label_2



#done#within same price range
#done#Synonyms
#done#difflib text matching
#after#Manually group : pattern, material ...


# In[ ]:


#####PRICE
#RANGE between 49 (30)  and 431625 (500000)
#less than number
#greater than number
#between lower number and higher number
#less than number and greater than number
#less than number or greater than number
#less than greater than using symbol
#under, below, 
#rupees, inr

#####By item id
#p-<7digits>
#<7digits>

####By l2_category
#if text in list(df['l2_category'].unique())

###By product_type
#if text in list(df['product_type'].unique())

###By other attributes
#'condition', 'size', 'color', 'availability', 'brand', 'gender', 'shipping', 'totaldiscount', 'pattern', 'adult',
#       'custom_label_3', 'custom_label_2', 'custom_label_4','material'

####Title Description
#all other conditions




#/'condition', /'size', /'color', /'availability', *'price', /'brand',
#       /'gender', /'shipping', *'sale_price', *'totaldiscount', /'pattern', /'adult',
#       -'custom_label_3', /'custom_label_2', -'custom_label_4',/'material'
