from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
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


def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size * 2
    for sent in corpus:
        sentence_length = len(sent)
        for index, word in enumerate(sent):
            context_words = []
            label_words = []
            start = index - window_size
            end = index + window_size + 1

            context_words.append([sent[i] for i in range(start, end) 
                                if 0 <= i < sentence_length
                                and i != index])
            label_words.append(word)

            x = sequence.pad_sequences(context_words, maxlen=context_length)
            y = np_utils.to_categorical(label_words, vocab_size)
            yield(x, y)