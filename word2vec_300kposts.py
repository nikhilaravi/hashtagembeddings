# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Basic word2vec example."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
from tensorflow.contrib.tensorboard.plugins import projector
import collections
import math
import os
import random
import zipfile
import pandas as pd
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from itertools import combinations

# Step 1:

# set filenames for logs and input files

TAG_FILENAME = '../taglists/taglist_300k.txt'
LOG_DIR = '../tensorboard_logs/300k'
METADATA_NAME = 'metadata_300k.tsv'
BATCH_LABEL_TUPLE_DIR = '../pickled_batches/300k_pairs'

# constants

LOAD_TUPLES = True

# generate vocabulary and sentences

with open(TAG_FILENAME) as f:
    vocabulary = f.read().split()

f = open(TAG_FILENAME, 'r')
sentence_array_original = f.readlines()
sentence_array = [x.split('\n')[:-1][0] for x in sentence_array_original]

# TURN ALL WORDS TO LOWERCASE

lowercase_vocabulary = []
for word in vocabulary:
    lowercase_vocabulary.append(word.lower())
lowercase_vocabulary = list(lowercase_vocabulary)

# set vocab size to lowercase vocab size
vocabulary_size = len(list(set(lowercase_vocabulary)))

# STATS
print('Total number of words in corpus', len(vocabulary))
print('Total number of words in corpus after lowercasing', len(lowercase_vocabulary))
print('Total number of unique words ', len(list(set(vocabulary))))
print('Total number of unique words after converting to lower case', vocabulary_size)

# Step 2: Build the dictionary and replace rare words with UNK token.

def build_dataset(words, sentence_array, vocabulary_size):
    """Process raw inputs into a dataset."""
    count = []
    count.extend(collections.Counter(words).most_common(vocabulary_size))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    unk_count = 0
    sentences = list()
    for sentence in sentence_array:
        data = list()
        for word in sentence.split():
            lower_word = word.lower()
            if lower_word in dictionary:
                index = dictionary[lower_word]
            else:
                print ('word not in dictionary ', word)
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        sentences.append(data)
    # count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return sentences, count, dictionary, reversed_dictionary

# generate data set

sentences_to_ids, count, dictionary, reverse_dictionary = build_dataset(lowercase_vocabulary, sentence_array, vocabulary_size)

# save out metadata for tensorboard

# metadata = pd.DataFrame({'Word': reverse_dictionary.values()})
# metadata.to_csv(LOG_DIR+'/'+METADATA_NAME, sep='\t', index=False)

def write_metadata(filename, words):
    with open(filename, 'w') as w:
        w.write("word" + "\t" + "tag" + "\n")
        for word in words:
            # w.write(word + "\t" + nltk.pos_tag([word])[0][1][:2] + "\n")
            w.write(word + "\n")

write_metadata(LOG_DIR+'/'+METADATA_NAME,reverse_dictionary.values())

del vocabulary  # Hint to reduce memory.
del lowercase_vocabulary

# print stats

print('Most common words', count[:5])
print('Sample data', sentences_to_ids[0][:4], [reverse_dictionary[i] for i in sentences_to_ids[0][:4]])
print('number of sentences shape', np.shape(sentences_to_ids))


def generate_word_label_pairs(sentences_to_ids, n):
    word_combinations = []
    for indx, sentence_words in enumerate(sentences_to_ids):
        sentence_length = len(sentence_words)
        if sentence_length > n:
            sentence_word_combinations = list(combinations(sentence_words, n))
            word_combinations  = word_combinations + sentence_word_combinations
        # print ('sentence num ', indx, sentence_word_combinations)
    return word_combinations

# UNCOMMENT SECTION BELOW TO GENERATE PAIRS FOR NEW DATA SET

# batch_label_pairs = generate_word_label_pairs(sentences_to_ids, 2)
# pickle.dump(batch_label_pairs,  open( "batch_label_pairs_unltd.p", "wb" ) )

# LOAD PRE PICKLED TUPLE PAIRS

if LOAD_TUPLES:
    batch_tuple_files = os.listdir(BATCH_LABEL_TUPLE_DIR)
    batches  = []
    for batch_file in batch_tuple_files:
        file_path = BATCH_LABEL_TUPLE_DIR + '/' + batch_file
        batches.append(pickle.load( open( file_path, "rb" ) ))
        print('loaded: {}'.format(file_path))
    batch_label_pairs = np.concatenate(batches)
else:
    batch_label_pairs = generate_word_label_pairs(sentences_to_ids, 2)

data_size = len(batch_label_pairs)
print('batch label pairs shape ', np.shape(batch_label_pairs))
batch = np.array([b[0] for b in batch_label_pairs])
labels = np.array([b[1] for b in batch_label_pairs]).reshape((-1,1))

# PRINT SOME EXAMPLES OF BATCHES AND LABELS

for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

data_index = 0

def generate_batch(batch_size):
    global data_index
    if data_index > data_size - batch_size:
        data_index = 0
    return batch[data_index: data_index+batch_size], labels[data_index: data_index+batch_size,0].reshape(-1,1)

# Step 4: Build and train a cbow model.

batch_size = 128
print ('number of steps to go through each tuple pair once: ', int(data_size/batch_size))
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()
config = projector.ProjectorConfig()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='word_embedding')
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 10000001

with tf.Session(graph=graph) as session:
  # save output
  # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto

  config = projector.ProjectorConfig()
  # You can add multiple embeddings. Here we add only one.
  embedding = config.embeddings.add()
  embedding.tensor_name = embeddings.name
  # Link this tensor to its metadata file (e.g. labels).
  embedding.metadata_path = METADATA_NAME
  # Use the same LOG_DIR where you stored your checkpoint.
  summary_writer = tf.summary.FileWriter(LOG_DIR, session.graph)
  # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
  # read this file during startup.
  projector.visualize_embeddings(summary_writer, config)
  saver = tf.train.Saver()

  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    # batch_inputs, batch_labels = generate_batch(
    #     batch_size, num_skips, skip_window)
    batch_inputs, batch_labels = generate_batch(
        batch_size)
    # INCREMENT THE DATA INDEX
    data_index = data_index + batch_size
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 10000 == 0:
      if step > 0:
        average_loss /= 10000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), step)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 3  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
