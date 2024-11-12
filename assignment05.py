import glob
from random import shuffle
import os
import re
import tarfile
import tqdm
import pandas as pd

import numpy as np  # Keras takes care of most of this but it likes to see Numpy arrays
from keras.preprocessing import sequence    # A helper module to handle padding input
from keras.models import Sequential         # The base keras Neural Network model
from keras.layers import Dense, Dropout, Activation   # The layer objects we will pile into the model
from keras.layers import Conv1D, GlobalMaxPooling1D

from nltk.tokenize import TreebankWordTokenizer
from gensim.models import KeyedVectors

# pull in a dataframe of ade texts and labels
ade_df = pd.read_parquet("hf://datasets/ade-benchmark-corpus/ade_corpus_v2/Ade_corpus_v2_classification/train-00000-of-00001.parquet")
ade_df.to_csv('ade_df.csv')    
# Google pre-trained vectors, available here:
# https://code.google.com/archive/p/word2vec/
googlevecs = '/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin.gz'
word_vectors = KeyedVectors.load_word2vec_format(googlevecs, binary=True, limit=200000)

def pre_process_data(filepath):
    """
    This is dependent on your training data source but we will try to generalize it as best as possible.
    """
    positive_path = os.path.join(filepath, 'pos')
    negative_path = os.path.join(filepath, 'neg')

    pos_label = 1
    neg_label = 0

    dataset = []

    for filename in glob.glob(os.path.join(positive_path, '*.txt')):
        with open(filename, 'r') as f:
            dataset.append((pos_label, f.read()))

    for filename in glob.glob(os.path.join(negative_path, '*.txt')):
        with open(filename, 'r') as f:
            dataset.append((neg_label, f.read()))

    shuffle(dataset)

    return dataset







def tokenize_and_vectorize(dataset):
    tokenizer = TreebankWordTokenizer()

    vectorized_data = []
    expected = []

    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])

        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
#                print(f"not found: {token}")
                pass
             
        vectorized_data.append(sample_vecs)
    return vectorized_data



def collect_expected(dataset):
    """ Peel of the target values from the dataset """
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected

"""
Following the previous example, the Dataset format is a list of tuples,
with label as the first element and text as the second element.
Here we re-shape the ADE dataframe to fit this format.
"""
dataset = [(x[2], x[1]) for x in ade_df.itertuples()]

"""
Note: as discussed in class on Oct 17, many terms relevant to identifying adverse drug events,
such a drug names and the terms for particular symptoms/problems, may not be available in the
word2vec embeddings generated from Google news.

One solution would be to use another set of pre-computed vectors.
If there's not a good option for this type of text, we also might look into building our own.
"""
vectorized_data = tokenize_and_vectorize(dataset)

expected = collect_expected(dataset)

split_point = int(len(vectorized_data) * .8)

x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]



"""
Note: as discussed in class on Oct 17, 400 may be too large as a mex len for the ADE corpus.
The result would be many zero-vectors diluting the meaning of each document.
A solution wouldbe to pick a smaller value of maxlen.
"""
maxlen = 225
batch_size = 32         # How many samples to show the net before backpropogating the error and updating the weights
embedding_dims = 300    # Length of the token vectors we will create for passing into the Convnet
filters = 250           # Number of filters we will train
kernel_size = 3         # The width of the filters, actual filters will each be a matrix of weights of size: embedding_dims x kernel_size or 50 x 3 in our case
hidden_dims = 250       # Number of neurons in the plain feed forward net at the end of the chain
epochs = 3             # Number of times we will pass the entire training dataset through the network


def pad_trunc(data, maxlen):
    """ For a given dataset pad with zero vectors or truncate to maxlen """
    new_data = []

    # Create a vector of 0's the length of our word vectors
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)

    for sample in data:

        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data


x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)



model = Sequential()

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 input_shape=(maxlen, embedding_dims)))

# we use max pooling:
model.add(GlobalMaxPooling1D())
# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
