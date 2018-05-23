import pandas as pd
import json

fp = open('/data/tg_cg18_bigdata/rc_2018_02.csv', 'w')
fp.write('author,body,created_utc,controversiality,edited,author_flair_css_class,author_flair_text,link_id,parent_id,score\n')
for line in open("/data/tg_cg18_bigdata/RC_2018-02"):
    di =json.loads(line)
    fp.write(di['author'].replace(',','').replace('\n','') + "," + di['body'].replace(',','').replace('\n','') + "," + str(di['created_utc']) + "," + str(di['controversiality']) + "," + str(di['edited']) + "," + str(di['author_flair_css_class']).replace(',','').replace('\n','')+ "," + str(di['author_flair_text']).replace(',','').replace('\n','') + "," + di['link_id'].replace(',','').replace('\n','') + "," + str(di['parent_id']).replace(',','').replace('\n','') + "," + str(di['score']) + "\n")
  fp.close()
  
