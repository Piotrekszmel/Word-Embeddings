import nltk
import re
import spacy
import pandas as pd 
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
import unicodedata

#simple text pre-processing
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')


#basic
def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc


# more advanced
nlp = spacy.load('en', parse=False, tag=False, entity=False)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopword.words('english')
stopword_list.remove('no')
stopword_list.remove('not')