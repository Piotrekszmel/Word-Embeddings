from keras.models import Model
from keras.layers import dot, Input
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential

def model(vocab_size, embed_size, input_length):
    input = Input(shape=(1, ))
    embedding = Embedding(vocab_size, embed_size, input_length=input_length)(input)
    reshape = Reshape((embed_size, ))(embedding)
    
    input_context = Input(shape=(1, ))
    embedding_context = Embedding(vocab_size, embed_size, input_length=input_length,  embeddings_initializer='glorot_uniform')(input_context)
    reshape_context = Reshape((embed_size, ))(embedding_context)
    
    dot_product = dot([reshape, reshape_context], axes=1, normalize=False)
    dense = Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid')(dot_product)

    model = Model(inputs=[input, input_context], outputs=dense)
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model