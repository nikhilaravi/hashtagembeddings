import tensorflow as tf
from six.moves import xrange
import pickle

####### INITALISE DICTIONARY ###########

dictionary = pickle.load( open( "./dictionary/dictionary.p", "rb" ) )
reverse_dictionary = pickle.load( open( "./dictionary/reverse_dictionary.p", "rb" ) )

###### INITIALISE MODEL ###########

def load_model():
    with tf.device('/cpu:0'):
        print ('Model Initalizing.....')
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('./tensorboard_logs/300k_lowercase_corrected/model.ckpt-6520000.meta')
        print ('Model Initalized')

        # Get default graph (supply your custom graph if you have one)
        graph = tf.get_default_graph()
        embeddings = graph.get_tensor_by_name('word_embedding:0')
        sess = tf.Session()

        print ('Initialising embeddings...')
        # To initialize variable values with saved data
        saver.restore(sess,tf.train.latest_checkpoint('./tensorboard_logs/300k_lowercase_corrected/'))
        print ('Initialised embeddings')

        # # Compute the cosine similarity between chosen words and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

        # return function which takes an array of hashtags and returns k nearest neighbors for each
        def find_k_nearest_words(words_to_look_up, top_k=10):
            '''
            Function which takes an array of hashtags and a value for the number of neighbors to return
            Returns a dictionary with each word mapped to a list of its nearest neighbors
            '''
            look_up_indices = [dictionary[word] for word in words_to_look_up]

            print ('word: ', words_to_look_up)
            print ('word index: ', look_up_indices)

            valid_dataset = tf.constant(look_up_indices, dtype=tf.int32)
            valid_embeddings = tf.nn.embedding_lookup(
              normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                  valid_embeddings, normalized_embeddings, transpose_b=True)

            ###### FIND NEAREST NEIGHBORS ###########

            print('Calculating nearest neighbors....')
            sim = sess.run(similarity)

            # save nearest hashtags in a dictionary
            nearest_words = dict()

            for i in xrange(len(look_up_indices)):
                valid_word = reverse_dictionary[look_up_indices[i]]
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                nearest_words[valid_word] = []
                for k in xrange(top_k):
                  close_word = reverse_dictionary[nearest[k]]
                  nearest_words[valid_word].append(close_word)
                  log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
            return nearest_words

    return find_k_nearest_words
