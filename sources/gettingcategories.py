# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\max\Documents\Dev\Prog2\Abgabe4\SALT_USWS\sources\allWithCategory.csv')
df.columns = ['latitude', 'longitude','rating','review_count','price','categories']
df = df[0:100]

categories_df['categories'] = 0

for x in range(len(df)):
    cat = df.at[x,'categories']
    if(cat in categories_df['categories']):
        categories_df=categories_df.append({'categories' : df.at[x,'categories']},ignore_index=True)
        
            if(cat in categories_df.at[y,'category']):
                categories_df.at[x,'count']=+1
    else:
        print('no')
        new_column = pd.DataFrame([cat,1],columns=['category','count'])
        categories_df = categories_df.append(new_column)
        yy = len(categories_df)
    
#categories_df = categories_df.sort_values(by=['count'])
#print('#Kategorien:' + str(len(categories_df)))
#      
#for x in range(len(categories_df)):
#    if(categories_df.at[x,'count'] > 10):
#        print(categories_df.at[x,'category'])

list = df['categories']