from nltk.corpus import gutenberg
from string import punctuation
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
from text_preprocessing import normalize_document, tokenization, generate_context_word_pairs
from model import model

corpus = ['The sky is blue and beautiful.',
          'Love this blue and beautiful sky!',
          'The quick brown fox jumps over the lazy dog.',
          "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
          'I love green eggs, ham, sausages and bacon!',
          'The brown fox is quick and the blue dog is lazy!',
          'The sky is very blue and the sky is very beautiful today',
          'The dog is lazy but the brown fox is quick!'    
]
labels = ['weather', 'weather', 'animals', 'food', 'food', 'animals', 'weather', 'animals']

corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus,
                          'Category': labels})
corpus_df = corpus_df[['Document', 'Category']]

normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus(corpus)

bible = gutenberg.sents('bible-kjv.txt') 
remove_terms = punctuation + '0123456789'

norm_bible = [[word.lower() for word in sent if word not in remove_terms] for sent in bible]
norm_bible = [' '.join(tok_sent) for tok_sent in norm_bible]
norm_bible = filter(None, normalize_corpus(norm_bible))
norm_bible = [tok_sent for tok_sent in norm_bible if len(tok_sent.split()) > 2]

tokenizer = text.Tokenizer()
word2id, id2word, sequences = tokenization(tokenizer, norm_bible)

vocab_size = len(word2id)
embed_size = 100
window_size = 2

cbow = model(vocab_size, embed_size, window_size)
cbow.summary()

for epoch in range(1, 6):
    loss = 0
    i = 0
    for x, y in generate_context_word_pairs(corpus=sequences, window_size=window_size, vocab_size=vocab_size):
        i += 1
        loss += cbow.train_on_batch(x, y)
        if i % 100000 == 0:
            print('Processed {} (context, word) pairs'.format(i))
    
    print('Epoch:', epoch, '\tloss:', loss)
    print()