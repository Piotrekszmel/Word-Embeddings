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
    return word2id, id2word