# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:54:21 2024

@author: Mukhtar
"""

import re
import string
import requests
from bs4 import BeautifulSoup

import pyarabic.araby as araby
from nltk.corpus import stopwords # arabic stopwords
import arabicstopwords.arabicstopwords as stp # arabic stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import qalsadi.lemmatizer
#import nltk

#nltk.download('stopwords')

def normalize_chars(txt):
    txt = re.sub("[إأٱآا]", "ا", txt)
    txt = re.sub("ى", "ي", txt)
    txt = re.sub("ة", "ه", txt)
    return txt

def clean_txt(txt):
    
    lemmer = qalsadi.lemmatizer.Lemmatizer()
    stopwordlist = set(list(stp.stopwords_list()) + stopwords.words('arabic'))
    stopwordlist = [normalize_chars(word) for word in stopwordlist]
    # remove tashkeel & tatweel
    txt = araby.strip_diacritics(txt)
    txt = araby.strip_tatweel(txt)
    # normalize chars
    txt = normalize_chars(txt)
    # remove stopwords & punctuation
    txt = ' '.join([token.translate(str.maketrans('','',string.punctuation)) for token in txt.split(' ') if token not in stopwordlist])
    # lemmatizer
    txt_lemmatized = ' '.join([lemmer.lemmatize(token) for token in txt.split(' ')])
    return txt+" "+txt_lemmatized

def get_surah_names():
    surah_names = [] #surah names sorted
    URL = "https://surahquran.com/quran-search/quran.html"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    all_table = soup.find_all('table')[1]
    for elm in all_table.find_all("a"):
        surah_names.append(elm.text)
    return surah_names

def show_best_results(df_quran, scores_array, top_n=1):
    sorted_indices = scores_array.argsort()[::-1]
    for position, idx in enumerate(sorted_indices[:top_n]):
        row = df_quran.iloc[idx]
        ayah = row["ayah_txt"]
        ayah_num = row["ayah_num"]
        surah_name = row["surah_name"]
        score = scores_array[idx]
        inf =''
        if score > 0:
            inf = inf + ayah + '\n' + f'أيه رقم {ayah_num}  سورة {surah_name}'
            # print(ayah)
            # print(f'أيه رقم {ayah_num}  سورة {surah_name}')
            # print("====================================")
    return inf  

def run_tfidf(query,df):
    corpus = df["clean_txt"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    corpus_vectorized = vectorizer.fit_transform(corpus)
    query = clean_txt(query)
    query_vectorized = vectorizer.transform([query])
    scores = query_vectorized.dot(corpus_vectorized.transpose())
    scores_array = scores.toarray()[0]
    info = show_best_results(df, scores_array)
    return info

