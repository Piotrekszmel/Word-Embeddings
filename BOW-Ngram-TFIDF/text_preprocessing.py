import nltk
import re
import spacy
import pandas as pd 
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import contractions
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
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

#Cleaning Text - strip HTML
def strip_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    stripped_text = soup.get_text()
    return stripped_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text):
    text = contractions.fix(text)
    return text


def removing_special_characters(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text