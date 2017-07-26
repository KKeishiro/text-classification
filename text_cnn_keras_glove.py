from __future__ import print_function

import os
import sys
import tensorflow as tf
import data_helpers
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
X_DATA_FILE = BASE_DIR + '/data/keras_reuters_x_train.csv'
Y_DATA_FILE = BASE_DIR + '/data/keras_reuters_y_train.csv'
MAX_SEQUENCE_LENGTH = 5000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1


# Data loading params
# FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_string("x_train", "./data/keras_reuters_x_train.csv", "Data source for x.")
# tf.flags.DEFINE_string("y_train", "./data/keras_reuters_y_train.csv", "Data source for y.")

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(X_DATA_FILE, Y_DATA_FILE)

# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(x_text)
sequences = tokenizer.texts_to_sequences(x_text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# max_document_length = max([len(x) for x in x_text])
x = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', x.shape)
print('Shape of label tensor:', y.shape)

# split the data into a training set and a validation set
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x_shuffled = x[indices]
y_shuffled = y[indices]

x_train, x_val, y_train, y_val = train_test_split(x_shuffled, y_shuffled, test_size = VALIDATION_SPLIT)

# build index mapping words in the embeddings set to their embedding vector
print('Indexing word vectors.')
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# prepare embedding matrix
print('Preparing embedding matrix.')
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
preds = Dense(y.shape[1], activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))
