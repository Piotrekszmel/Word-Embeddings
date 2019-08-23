import pandas as pd
import numpy as np
from text_preprocessing import normalize_document
from sklearn.feature_extraction.text import CountVectorizer

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


#Bag of Words
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)
cv_matrix = cv_matrix.toarray()


c = CountVectorizer(min_df=1, max_df=2)
c_matrix = c.fit_transform(['I am am am am lol lol', 'am am am I lol','am am' ])
print(c_matrix.toarray())