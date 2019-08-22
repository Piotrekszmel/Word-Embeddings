from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.sequence import skipgrams
from keras.layers import merge
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import nltk 
import re

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc


def tokenization(tokenizer, corpus):
    tokenizer.fit_on_texts(corpus)
    word2id = tokenizer.word_index
    word2id['PAD'] = 0
    id2word = {v:k for k, v in word2id.items()}
    sequences = tokenizer.texts_to_sequences(corpus)
    return word2id, id2word, sequences


def build_skip_grams(vocab_size, window_size, sequences):
    skip_grams = [skipgrams(token, vocabulary_size=vocab_size, window_size=window_size) for token in sequences]
    return skip_grams

