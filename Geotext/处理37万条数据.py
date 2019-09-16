# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:21:03 2019

@author: GIS
"""

import pandas as pd
import numpy as np
import re


tweet = pd.read_csv('E:\\deep_network\\Geotext dataset\\Data\\address_all_9236.csv',
                    encoding='ISO-8859-1', header=None)
tweet[13].value_counts()


tweet1 = pd.read_csv('E:\\deep_network\\Geotext dataset\\Data\\full_text_all_label.csv',
                    encoding='ISO-8859-1', header=None)
tweet1[6].value_counts()
tweet1.columns = ['uid','time','cor','lat','lng','content','state','fenci']



# =================================提取@duixiagn ==============================
def extra_aite(df):
    pattern = "[＠@]\\s?[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}:?"
    text = list(df['content'])
    aite = []
    for ele in text:
        ele = re.findall(pattern, str(ele))
        ele1 = []
        for e in ele:
            e = e.strip(':')
            ele1.append(e)
        aite.append(ele1)
    df['aite'] = aite
    return df

tweet2 = extra_aite(tweet1)


#=============================== 替换重复的州==================================
add = list(tweet2.state.value_counts().keys())
state = list(tweet2.state)
state1 = []
for ele in state:
    if ele == 'District de Columbia':
        ele = 'District of Columbia'
    if ele == 'Floride':
        ele = 'Florida'
    if ele == 'GÃ©orgie':
        ele = 'Georgia'
    if ele == 'GÃÂ©orgie':
        ele = 'Georgia'
    if ele == 'Louisiane':
        ele = 'Louisiana'
    if ele == 'Virginie':
        ele = 'Virginia'
    if ele == 'Washington, D.C.':
        ele = 'Washington'
    if ele == 'Caroline-du-Nord':
        ele = 'North Carolina'
    if ele == 'Caroline-du-Sud':
        ele = 'South Carolina'
    if ele == 'District of Columbia':
        ele = 'Washington'
        
        
    state1.append(ele)
    
add1 = list(set(state1))
tweet2['state_48'] = state1
len(tweet2.uid.unique())


# ===============================转成小写，去除符号===========================
content = list(tweet2.content)
content1 = []
for ele in content:
    s =  re.sub("[＠@]\\s?[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}:?", '', ele)
    s = re.findall("[a-zA-Z]+", s)
    s = ' '.join(s)
    s = s.lower()
    content1.append(s)
tweet2['content_lower'] = content1
tweet2.to_csv('E:\\deep_network\\Geotext dataset\\code\\GCN\\367993tweets_state_content_aite.csv',
              encoding='ISO-8859-1', index=False)

# ============================ 整合到每个用户身上 ==============================
uid = list(set(list(tweet2.uid)))

state_list = []
content_list = []
aite_list = []
stopword = []
for i,u in enumerate(uid):
    print(i)
    df = tweet2[tweet2.uid == u]
    s = list(df.state_48)
    s = ','.join(s)
    state_list.append(s)
    
    c = list(df.content_lower)
    c = ' '.join(c)
    content_list.append(c)
    
    w = list(df.fenci)
    w = ' '.join(w)
    w = w.replace('[', '').replace(']', '').replace("'", '').replace(',', ' ')
    stopword.append(w)
    
    a = list(df.aite)
    ai = []
    for ele in a:
        if len(ele) > 0:
            ai.extend(ele)
    ai = ','.join(ai)
    aite_list.append(ai)

user_df = pd.DataFrame({'uid':uid, 'state':state_list, 'content':content_list, 'aite':aite_list, 'stopword':stopword})
user_df.to_csv('E:\\deep_network\\Geotext dataset\\code\\GCN\\9236user_state_content_aite.csv',
              encoding='ISO-8859-1', index=False)

#=============== 选取一个区域作为标签 =================================================
import collections
from collections import Counter

state_one = []
for ele in state_list:
    ele = ele.split(',')
    c = Counter(ele)
    max(list(c.values()))
    sta = list(c.keys())[list(c.values()).index(max(list(c.values())))]
    state_one.append(sta)
Counter(state_one)
user_df['state_one'] = state_one
user_df.columns
