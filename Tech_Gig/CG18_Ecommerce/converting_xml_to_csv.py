#Converting xml dataframe to pandas dataframe
import pandas as pd
from bs4 import BeautifulSoup

tags = ['id', 'g:item_group_id', 'title', 'link', 'description', 'g:google_product_category', 'g:l2_category', 'g:product_type', 'g:image_link', 'g:condition', 'g:size', 'g:color', 'g:availability', 'g:price', 'g:brand', 'g:gender', 'g:shipping', 'g:sale_price', 'g:totaldiscount', 'g:pattern', 'g:adult', 'g:custom_label_3', 'g:gtin', 'g:custom_label_2', 'g:custom_label_4', 'g:material']
i = 0
li_for_df = []
for line in open("pla_rpc_shoppersstop (1).xml", 'r'):
    link = line[line.find('<link>') + 6 : line.find('</link>')]
    rw= [link]
    if i <6:
        pass
    elif i>156930:
        break
    else:
        soup = BeautifulSoup(line)
        for tag in tags:
            try:
                element = soup.find(tag).string.strip()
                rw.append(element)
            except:
                rw.append("")
    li_for_df.append(rw)
    i = i + 1

pd.DataFrame(li_for_df).to_csv("tg_ecomm_df.csv", index = False)
