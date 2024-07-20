# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:51:31 2024

@author: Mukhtar
"""

import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import ArabicStemmer # Arabic Stemmer gets rot word
import nltk
from text_and_voice_processing import *

print('Stopwords downloading...')
nltk.download('stopwords')

st = ArabicStemmer()

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', None)
sns.set()

print('CSV Readinging...')
df = pd.read_csv('Arabic-Original.csv', header = None, names=['text'])
print('Split Text in dataframe...')
df = df['text'].str.split('|', expand=True)
print('Columns defining...')
df.columns = ['surah_num', 'ayah_num', 'ayah_txt']
print('Still Preparing data...')
df['surah_num'] = df['surah_num'].astype('int')
df['ayah_num'] = df['ayah_num'].astype('int')


print('Text Cleaning...')
df['clean_txt'] = df['ayah_txt'].apply(lambda x: clean_txt(x))

print('Get Surah names...')
quran_surah_names = get_surah_names()
print('Add Surah names to dataframe...')
df['surah_name'] = df['surah_num'].apply(lambda x: quran_surah_names[x-1])

print('Grouping Surah...')
grouped_surah = {'surah_num': [],'surah_name': [], 'surah_txt': []}

for i in range(1,df['surah_num'].max()+1):
    surah_txt = ' '.join(df[df['surah_num']==i]['ayah_txt'].to_list())
    grouped_surah['surah_txt'].append(surah_txt)
    grouped_surah['surah_num'].append(i)
    grouped_surah['surah_name'].append(quran_surah_names[i-1])
    
df_surah = pd.DataFrame(grouped_surah)

print('Create Corpus from cleaned TxT...')
corpus = df["clean_txt"]
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
corpus_vectorized = vectorizer.fit_transform(corpus)

df.to_pickle('surah.plk')