import numpy as np
import tensorflow as tf

class S2VT():
    RNNsAreInitialized = False
    def __init__(self, x_dim, num_vocal, num_units, 
                 enc_time_steps, dec_time_steps, 
                 batch_size, optimizer, learning_rate, 
                 rnn_type='lstm', use_dropout=False, use_attention=False, 
                 use_scheduled=False, use_embedding=False, word_list=None, 
                 x_bias_vector=None, y_bias_vector=None):
        # Parameters
        self.x_dim = x_dim
        self.num_vocal = num_vocal
        self.num_units = num_units
        self.enc_time_steps = enc_time_steps
        self.dec_time_steps = dec_time_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        S2VT.RNNsAreInitialized = False
        self.rnn_type = rnn_type
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        self.use_scheduled = use_scheduled
        self.use_embedding = use_embedding
        
        # Optimizer
        self.optimizer = self.get_optimizer(optimizer)
        # Word Embedding
        if use_embedding is None or not use_embedding:
            self.word_embedding = tf.Variable(tf.random_uniform(
                    [num_vocal, num_units], -0.1, 0.1), name='word_embedding')
        else:
            self.word_embedding = tf.Variable(tf.random_uniform(
                    [num_vocal, 300], -0.1, 0.1), name='word_embedding')
        # RNN Cells
        self.rnn1, self.rnn2, self.rnn3 = self.get_RNN_cells(rnn_type, num_units)
        
        # Input Encoder
        self.embed_x_W = tf.Variable(
            tf.random_uniform([x_dim, num_units], -0.1, 0.1), name='embed_x_W')
        if not x_bias_vector is None:
            self.embed_x_b = tf.Variable(
                x_bias_vector.astype(np.float32), name='embed_x_b')
        else:
            self.embed_x_b = tf.Variable(
                tf.zeros([num_units]), name='embed_x_b')
        
        # Attention Mechanism
        # Here, I use BasicLSTMCell or GRUCell as my RNN, both of the outputs are indeed their hidden states (h).
        # Therefore, I'll take outputs as inputs of the attention model.
        if use_attention:
            self.attention_e_W = tf.Variable(
                tf.random_uniform(
                    [num_units, num_units], -0.1, 0.1), name='attention_eW')
            self.attention_d_W = tf.Variable(
                tf.random_uniform(
                    [num_units, num_units], -0.1, 0.1), name='attention_dW')
            self.attention_v = tf.Variable(
                tf.random_uniform(
                    [num_units, 1], -0.1, 0.1), name='attention_v')
        
        # Output Projector
        self.project_y_W = tf.Variable(
            tf.random_uniform([num_units, num_vocal], -0.1, 0.1), name='project_y_W')
        if not y_bias_vector is None:
            self.project_y_b = tf.Variable(
                y_bias_vector.astype(np.float32), name='project_y_b')
        else:
            self.project_y_b = tf.Variable(
                tf.zeros([num_vocal]), name='project_y_b')
        
        print('S2VT initialized.')
        
    def build_model_train(self):
        # Placeholders
        x = tf.placeholder(tf.float32, [self.batch_size, self.enc_time_steps, self.x_dim])
        x_seq_len = tf.placeholder(tf.float32, [self.batch_size])
        x_max_len = tf.placeholder(tf.float32, [])
        x_masks = tf.sequence_mask(x_seq_len, x_max_len, dtype=tf.float32)
        y = tf.placeholder(tf.int64, [self.batch_size, self.dec_time_steps+1])
        y_seq_len = tf.placeholder(tf.float32, [self.batch_size])
        y_max_len = tf.placeholder(tf.float32, [])
        y_masks = tf.sequence_mask(y_seq_len, y_max_len, dtype=tf.float32)
        sampling_rate = tf.placeholder(tf.float32, [])
        
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        rnn3 = self.rnn3
        
        # Dropout Layer
        if self.use_dropout:
            rnn1 = tf.nn.rnn_cell.DropoutWrapper(
                self.rnn1, output_keep_prob=1.0-self.use_dropout, 
                variational_recurrent=True, dtype=tf.float32)
            rnn2 = tf.nn.rnn_cell.DropoutWrapper(
                self.rnn2, output_keep_prob=1.0-self.use_dropout, 
                variational_recurrent=True, dtype=tf.float32)
            rnn3 = tf.nn.rnn_cell.DropoutWrapper(
                self.rnn3, output_keep_prob=1.0-self.use_dropout, 
                variational_recurrent=True, dtype=tf.float32)
        
        # Embedding Stage
        x_flatten = tf.reshape(x, [-1, self.x_dim])
        x_embedded = tf.nn.xw_plus_b(x_flatten, 
                                      self.embed_x_W, self.embed_x_b)
        x_embedded = tf.reshape(x_embedded, 
                                 [self.batch_size, self.enc_time_steps, self.num_units])
        
        state1 = rnn1.zero_state(
            batch_size=self.batch_size, dtype=tf.float32)
        state2 = rnn2.zero_state(
            batch_size=self.batch_size, dtype=tf.float32)
        state3 = rnn3.zero_state(
            batch_size=self.batch_size, dtype=tf.float32)
        padding1 = tf.zeros([self.batch_size, self.num_units])
        padding2 = tf.zeros([self.batch_size, self.word_embedding.shape[1]])
        padding3 = tf.zeros([self.batch_size, self.num_units])
        
        loss = 0.0
        probs = []
        ids = None
        
        if self.use_attention:
            enc_atten = None
        
        # Encoding Stage
        for i in range(self.enc_time_steps):
            with tf.variable_scope('RNN1') as scope:
                if i > 0 or S2VT.RNNsAreInitialized: self.scope_reuse(scope)
                output1, state1 = rnn1(x_embedded[:, i, :], state1)
                if self.use_attention:
                    if i == 0:
                        enc_atten = output1
                    else:
                        enc_atten = tf.concat([enc_atten, output1], 1)
            with tf.variable_scope('RNN2') as scope:
                if i > 0 or S2VT.RNNsAreInitialized: self.scope_reuse(scope)
                output2, state2 = rnn2(tf.concat([padding2, output1], 1), state2)
            with tf.variable_scope('RNN3') as scope:
                if i > 0 or S2VT.RNNsAreInitialized: self.scope_reuse(scope)
                output3, state3 = rnn3(tf.concat([padding3, output2], 1), state3)
        
        # Decoding Stage
        for i in range(self.dec_time_steps):
            # Scheduled Sampling
            sample = y[:, i]
            if self.use_scheduled and i > 0:
                sample = tf.cond(tf.random_uniform([], 0.0, 1.0) < sampling_rate, 
                                 lambda: max_prob_indices[:, 0], 
                                 lambda: y[:, i])
            current_embed = tf.nn.embedding_lookup(self.word_embedding, sample)
                
            with tf.variable_scope('RNN1'):
                output1, state1 = rnn1(padding1, state1)
            with tf.variable_scope('RNN2'):
                output2, state2 = rnn2(tf.concat([current_embed, output1], 1), state2)

            # Attention Mechanism
            if self.use_attention:
                eWs = tf.matmul(tf.reshape(enc_atten, [-1, self.num_units]), self.attention_e_W)
                dWs = tf.tile(tf.matmul(output2, self.attention_d_W), [self.enc_time_steps, 1])
                v_mul_tanh_eWs_plus_dWs = tf.matmul(tf.tanh(eWs + dWs), self.attention_v)
                attention_weights = tf.nn.softmax(
                    tf.reshape(v_mul_tanh_eWs_plus_dWs, 
                               [self.batch_size, self.enc_time_steps, -1]), 
                    dim=1)
                attention = tf.reduce_sum(
                    tf.reshape(enc_atten, 
                               [self.batch_size, self.enc_time_steps, self.num_units]) * attention_weights, 
                    axis=1) 
                
            with tf.variable_scope('RNN3'):
                if not self.use_attention:
                    output3, state3 = rnn3(tf.concat([padding3, output2], 1), state3)
                else:
                    output3, state3 = rnn3(tf.concat([attention, output2], 1), state3)
            
            # Projecting Stage
            y_logits = tf.nn.xw_plus_b(output3, self.project_y_W, self.project_y_b)
            
            # # Original Cross Entropy
            # indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            # labels = tf.expand_dims(y[:, i+1], 1)
            # y_labels_onehot = tf.sparse_to_dense(
            #     tf.concat([indices, labels], 1), 
            #     tf.stack([self.batch_size, self.num_vocal]), 1.0, 0.0)
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_labels_onehot)
            
            # Cross Entropy
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=y[:, i+1])
            cross_entropy = cross_entropy * y_masks[:, i]
            current_loss = tf.reduce_sum(cross_entropy) / self.batch_size
            
            max_prob_indices = tf.argmax(y_logits, 1)
            
            loss += current_loss
            probs.append(y_logits)
            max_prob_indices = tf.expand_dims(max_prob_indices, 1)
            ids = max_prob_indices if ids is None else tf.concat([ids, max_prob_indices], 1)
        
        train_op = self.optimizer(self.learning_rate).minimize(loss)
        
        self.built = True
        print('Training model is built.')
        
        debug = []
        return train_op, loss, probs, ids, x, x_seq_len, x_max_len, y, y_seq_len, y_max_len, sampling_rate, self.word_embedding, debug
        
    def build_model_test(self):
        # Placeholders
        x = tf.placeholder(tf.float32, [self.batch_size, self.enc_time_steps, self.x_dim])
        x_seq_len = tf.placeholder(tf.float32, [self.batch_size])
        x_max_len = tf.placeholder(tf.float32, [])
        x_masks = tf.sequence_mask(x_seq_len, x_max_len, dtype=tf.float32)
        
        rnn1 = self.rnn1
        rnn2 = self.rnn2
        rnn3 = self.rnn3
        
        # Embedding Stage
        x_flatten = tf.reshape(x, [-1, self.x_dim])
        x_embedded = tf.nn.xw_plus_b(x_flatten, self.embed_x_W, self.embed_x_b)
        x_embedded = tf.reshape(x_embedded, [self.batch_size, self.enc_time_steps, self.num_units])
        
        state1 = rnn1.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        state2 = rnn2.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        state3 = rnn3.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        padding1 = tf.zeros([self.batch_size, self.num_units])
        padding2 = tf.zeros([self.batch_size, self.word_embedding.shape[1]])
        padding3 = tf.zeros([self.batch_size, self.num_units])
        
        embeds = []
        probs = []
        ids = None
        
        if self.use_attention:
            enc_atten = None
        
        # Encoding Stage
        for i in range(self.enc_time_steps):
            with tf.variable_scope('RNN1') as scope:
                if i > 0 or S2VT.RNNsAreInitialized: self.scope_reuse(scope)
                output1, state1 = rnn1(x_embedded[:, i, :], state1)
                if self.use_attention:
                    if i == 0:
                        enc_atten = output1
                    else:
                        enc_atten = tf.concat([enc_atten, output1], 1)
            with tf.variable_scope('RNN2') as scope:
                if i > 0 or S2VT.RNNsAreInitialized: self.scope_reuse(scope)
                output2, state2 = rnn2(tf.concat([padding2, output1], 1), state2)
            with tf.variable_scope('RNN3') as scope:
                if i > 0 or S2VT.RNNsAreInitialized: self.scope_reuse(scope)
                output3, state3 = rnn3(tf.concat([padding3, output2], 1), state3)
        
        # Decoding Stage
        for i in range(self.dec_time_steps):
            if i == 0:
                current_embed = tf.nn.embedding_lookup(
                    self.word_embedding, tf.ones([self.batch_size], dtype=tf.int64))
                
            with tf.variable_scope('RNN1'):
                output1, state1 = rnn1(padding1, state1)
            with tf.variable_scope('RNN2'):
                output2, state2 = rnn2(tf.concat([current_embed, output1], 1), state2)

            # Attention Mechanism
            if self.use_attention:
                eWs = tf.matmul(tf.reshape(enc_atten, [-1, self.num_units]), self.attention_e_W)
                dWs = tf.tile(tf.matmul(output2, self.attention_d_W), [self.enc_time_steps, 1])
                v_mul_tanh_eWs_plus_dWs = tf.matmul(tf.tanh(eWs + dWs), self.attention_v)
                attention_weights = tf.nn.softmax(
                    tf.reshape(v_mul_tanh_eWs_plus_dWs, 
                               [self.batch_size, self.enc_time_steps, -1]), 
                    dim=1)
                attention = tf.reduce_sum(
                    tf.reshape(enc_atten, 
                               [self.batch_size, self.enc_time_steps, self.num_units]) * attention_weights, 
                    axis=1) 
            
            with tf.variable_scope('RNN3'):
                if not self.use_attention:
                    output3, state3 = rnn3(tf.concat([padding3, output2], 1), state3)
                else:
                    output3, state3 = rnn3(tf.concat([attention, output2], 1), state3)
            
            # Projecting Stage
            y_logits = tf.nn.xw_plus_b(output3, self.project_y_W, self.project_y_b)
            max_prob_indices = tf.argmax(y_logits, 1)
            current_embed = tf.nn.embedding_lookup(self.word_embedding, max_prob_indices)
            
            embeds.append(current_embed)
            probs.append(y_logits)
            max_prob_indices = tf.expand_dims(max_prob_indices, 1)
            ids = max_prob_indices if ids is None else tf.concat([ids, max_prob_indices], 1)
        
        self.built = True
        print('Testing model is built.')
        debug = []
        return embeds, probs, ids, x, x_seq_len, x_max_len, debug
    
    def scope_reuse(self, scope):
        scope.reuse_variables()
        S2VT.RNNsAreInitialized = True
    
    def get_optimizer(self, optimizer):
        if optimizer == 'gd':
            return tf.train.GradientDescentOptimizer
        elif optimizer == 'adam':
            return tf.train.AdamOptimizer
        elif optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer
        else:
            assert False, 'Error: not supported optimizer [{}]'
    
    def get_RNN_cells(self, rnn_type, num_units):
        rnns = []
        if rnn_type.lower() == 'lstm':
            rnns.append(tf.nn.rnn_cell.BasicLSTMCell(
                num_units=num_units, state_is_tuple=True))
            rnns.append(tf.nn.rnn_cell.BasicLSTMCell(
                num_units=num_units, state_is_tuple=True))
            rnns.append(tf.nn.rnn_cell.BasicLSTMCell(
                num_units=num_units, state_is_tuple=True))
        elif rnn_type.lower() == 'gru':
            rnns.append(tf.nn.rnn_cell.GRUCell(
                num_units=num_units))
            rnns.append(tf.nn.rnn_cell.GRUCell(
                num_units=num_units))
            rnns.append(tf.nn.rnn_cell.GRUCell(
                num_units=num_units))
        else:
            assert False, 'Error: not supported rnn type [{}]'.format(rnn_type)
        return rnns
