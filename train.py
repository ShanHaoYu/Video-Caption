import numpy as np
import tensorflow as tf
import sys
from os.path import join
from utils import MSVD, word_embedding_loader
from s2vt import S2VT

if len(sys.argv) != 3:
	assert False, 'Error: invalid number of arguments. Predicting should be [hw2.py data_path model_file]'

dataset_path = sys.argv[1]

x_dim = 4096
x_max_length = 80
y_max_length = 20
word_encoding_threshold = 1

num_units = 256
epochs = 420
batch_size = 50
optimizer = 'rmsprop' # 'gd', 'adam', or 'rmsprop'
learning_rate = 0.001
max_to_keep = 50
random_every_epoch = True #Let the caption and the id can be selected randomly
shuffle_training_data = True

rnn_type = 'gru' # 'lstm' or 'gru'
use_dropout = None # None or a float number (dropout_rate)
use_attention = True # Fasle or True
use_scheduled = False # False or True
sampling_decaying_rate = None # a float between 0~1 e.g. 0.99
sampling_decaying_mode = None # 'linear or 'exponential'
sampling_decaying_per_epoch = None # an integer
use_embedding = 'fasttext' # None, 'word2vec', 'glove', or 'fasttext'

model_folder = 'models'
model_name = sys.argv[2]
model_file = join(model_folder, model_name)

print('# {}'.format(model_name))

msvd = MSVD(dataset_path, y_max_length, word_encoding_threshold)
init_bias_vector = msvd.get_bias_vector()
num_vocal = len(msvd.sentenceEncoder.word2int)
word_list = msvd.get_word_list()

# num_vocal = 6000
s2vt = S2VT(x_dim, num_vocal, num_units, x_max_length, y_max_length, 
            batch_size, optimizer, learning_rate, rnn_type, 
            use_dropout, use_attention, use_scheduled, use_embedding, 
            word_list, y_bias_vector=init_bias_vector)
train_op, tf_loss, tf_probs, tf_preds, tf_x, tf_x_seq_len, tf_x_max_len, tf_y, tf_y_seq_len, tf_y_max_len, tf_sampling_rate, tf_word_embedding, tf_debug = s2vt.build_model_train()

saver = tf.train.Saver(max_to_keep=max_to_keep)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
assign_op = tf_word_embedding.assign(word_embedding_loader(use_embedding, word_list, 'word_embeddings'))
sess.run(assign_op)
print('Session started')

best_loss_avg = 1e10 # Just a super large number
inv_sampling_rate = 1.0 # Probability of initial true label sampling

msvd.load_training_data()

if not random_every_epoch:
    msvd.set_captions_randomly()

print('Training started.')

for epoch in range(epochs):
    if random_every_epoch:
        msvd.set_captions_randomly()
    count = 0
    loss_sum = 0.0
    for x_, y_, x_seq_len_, y_seq_len_ in msvd.next_batch(batch_size):
        if shuffle_training_data:
            p = np.random.permutation(x_.shape[0])
            x_, y_, x_seq_len_, y_seq_len_ = x_[p], y_[p], x_seq_len_[p], y_seq_len_[p]
        _, loss, probs, preds, d = sess.run(
            [train_op, tf_loss, tf_probs, tf_preds, tf_debug], 
            feed_dict={tf_x:         x_, 
                       tf_x_seq_len: x_seq_len_, 
                       tf_x_max_len: x_max_length, 
                       tf_y:         y_, 
                       tf_y_seq_len: y_seq_len_, 
                       tf_y_max_len: y_max_length
                       , tf_sampling_rate: 1.0-inv_sampling_rate
                      })

        count += 1
        loss_sum += loss
    loss_avg = loss_sum / count
    if loss_avg < best_loss_avg: best_loss_avg = loss_avg
    if use_scheduled: print('sampling_rate={}'.format(1.0-inv_sampling_rate))
    print('Epoch {}, average_train_loss: {}.'.format(epoch, loss_avg))
    if (epoch+1) % sampling_decaying_per_epoch== 0:
        if sampling_decaying_mode == 'linear':
            if inv_sampling_rate > 0.0:
                inv_sampling_rate -= (1 - sampling_decaying_rate)
            else:
                inv_sampling_rate = 0.0
        elif sampling_decaying_mode == 'exponential':
            inv_sampling_rate *= sampling_decaying_rate
        else:
            assert False, 'Error: Not supported decaying mode: [{}]'.format(sampling_decaying_mode)

print('Model saved.')
saver.save(sess, '{}.ckpt'.format(model_file))
print('Finished training. Best average_train_loss: {}.'.format(best_loss_avg))
