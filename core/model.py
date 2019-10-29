import tensorflow as tf


class RNN(tf.keras.Model):
    """A wrapper of the RNN module."""

    def __init__(self, cell_type, hidden_dim):
        super(RNN, self).__init__()
        if cell_type == "GRU":
            self.rnn = tf.keras.layers.CuDNNGRU(
                units=hidden_dim,
                return_sequences=True,
                recurrent_initializer='glorot_uniform')
        elif cell_type == "LSTM":
            self.rnn = tf.keras.layers.CuDNNLSTM(
                units=hidden_dim,
                return_sequences=True,
                recurrent_initializer='glorot_uniform')
        else:
            raise ValueError('Only GRU and LSTM are supported')

        self.rnn = tf.keras.layers.Bidirectional(self.rnn)

        self.cell_type = cell_type
        self.hidden_dim = hidden_dim

    def call(self, inputs):
        """
        Inputs: 
            inputs -- (batch_size, seq_length, input_dim)
        Outputs: 
            outputs -- (batch_size, seq_length, hidden_dim) or
            (batch_size, seq_length, hidden_dim * 2) for bidirectional
        """
        outputs = self.rnn(inputs)
        return outputs


class Embedding(tf.keras.Model):
    """A wrapper of the Embedding module."""

    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        """
        Inputs:
            vocab_size -- the total number of unique words
            embedding_dim -- the embedding dimension
            pretrained_embedding -- a numpy array (embedding_dim, vocab_size)
        """
        super(Embedding, self).__init__()

        try:
            init = tf.keras.initializers.Constant(pretrained_embedding)
            print("Initialize the embedding from a pre-trained matrix.")
        except:
            init = "uniform"
            print("Initialize the embedding randomly.")

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim,
                                                   embeddings_initializer=init)

    def call(self, inputs):
        """
        Inputs:
            inputs -- (batch_size, seq_length)
        Outputs:
            outputs -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(inputs)


class TargetRNN(tf.keras.Model):
    """A RNN-based Target Dependent Rationalization Model."""

    def __init__(self, args):
        super(TargetRNN, self).__init__()

        self.args = args

        # initialize the three embedding layers
        self.gen_neg_embedding_layer = Embedding(args.vocab_size,
                                                 args.embedding_dim,
                                                 args.pretrained_embedding)

        self.gen_pos_embedding_layer = Embedding(args.vocab_size,
                                                 args.embedding_dim,
                                                 args.pretrained_embedding)

        self.dis_embedding_layer = Embedding(args.vocab_size,
                                             args.embedding_dim,
                                             args.pretrained_embedding)

        # initialize RNN generators for both pos and neg
        self.generator_pos = RNN(args.cell_type, args.hidden_dim)

        self.generator_neg = RNN(args.cell_type, args.hidden_dim)

        # generator output layer (binary selection)
        self.generator_pos_fc = tf.keras.layers.Dense(units=2)
        self.generator_neg_fc = tf.keras.layers.Dense(units=2)

        # initialize a RNN discriminator
        self.discriminator = RNN(args.cell_type, args.hidden_dim)

        # discriminator output layer (classification task)
        self.discriminator_fc = tf.keras.layers.Dense(units=args.num_classes)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
            z -- (batch_size, sequence_length, 2)
        """
        z = tf.nn.softmax(rationale_logits)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)        
        """
        z = self._independent_soft_sampling(rationale_logits)
        z_hard = tf.cast(tf.equal(z, tf.reduce_max(z, -1, keep_dims=True)),
                         z.dtype)
        z = tf.stop_gradient(z_hard - z) + z

        return z

    def call(self, inputs, masks, labels, path):
        """
        Inputs:
            inputs -- (batch_size, seq_length)
            masks -- (batch_size, seq_length)
            labels -- (batch_size, num_classes)            
            path -- either 0 or 1, 0 -- generate neg rationale, 1 -- posistive\         
        """
        # expand dim for masks
        masks_ = tf.cast(tf.expand_dims(masks, -1), tf.float32)

        # include labels
        max_seq_length = inputs.shape[1]
        labels_ = tf.tile(tf.expand_dims(labels, axis=1),
                          [1, max_seq_length, 1])
        labels_ = tf.cast(labels_, tf.float32)
        labels_ = masks_ * labels_

        ############## Generator ##############
        # (batch_size, seq_length, 1)
        all_ones = tf.expand_dims(tf.ones(inputs.shape), axis=-1)
        all_zeros = tf.zeros(all_ones.shape)

        # generator
        # shape -- (batch_size, seq_length, hidden_dim * 2) if bidirectional
        if path == 0:
            # (batch_size, seq_length, embedding_dim)
            gen_neg_embeddings = masks_ * self.gen_neg_embedding_layer(inputs)
            gen_neg_inputs = tf.concat([gen_neg_embeddings, labels_], axis=-1)
            generator_outputs = self.generator_neg(gen_neg_inputs)
            generator_logits = self.generator_neg_fc(generator_outputs)
            gen_idx = tf.concat([all_ones, all_zeros], axis=-1)
        elif path == 1:
            # (batch_size, seq_length, embedding_dim)
            gen_pos_embeddings = masks_ * self.gen_pos_embedding_layer(inputs)
            gen_pos_inputs = tf.concat([gen_pos_embeddings, labels_], axis=-1)
            generator_outputs = self.generator_pos(gen_pos_inputs)
            generator_logits = self.generator_pos_fc(generator_outputs)
            gen_idx = tf.concat([all_zeros, all_ones], axis=-1)
        else:
            raise ValueError("`path` must be either 0 or 1.")

        # sample rationale (batch_size, sequence_length, 2)
        rationale = self.independent_straight_through_sampling(generator_logits)
        # mask the rationale
        # make the rationale that corresponding to <pad> to [1, 0]
        rationale = masks_ * rationale + (1. - masks_) * tf.concat(
            [all_ones, all_zeros], axis=-1)

        ############## Discriminator ##############

        # mask the input with rationale
        # only rationale words are consider in further computation
        # shape -- (batch_size, seq_length, embedding_dim)
        dis_embeddings = masks_ * self.dis_embedding_layer(inputs)
        rationale_embeddings = dis_embeddings * tf.expand_dims(
            rationale[:, :, 1], 2)

        # concat the path info before feeding to descriminator
        rationale_embeddings = tf.concat([rationale_embeddings, gen_idx],
                                         axis=-1)

        # encoder for task prediction
        # shape -- (batch_size, seq_length, hidden_dim * 2) if bidirectional
        discriminator_outputs = self.discriminator(rationale_embeddings)

        # mask before max pooling
        # discriminator_outputs * mask --> mask <pad>
        # (1 - mask) * (-10e6) --> <pad> become very neg
        masked_dis_outputs = discriminator_outputs * masks_ + (1. -
                                                               masks_) * (-1e6)

        # aggregates hidden outputs via max pooling
        # shape -- (batch_size, hidden_dim * 2)
        discriminator_output = tf.reduce_max(masked_dis_outputs, axis=1)

        # task prediction
        # shape -- (batch_size, num_classes)
        discriminator_logits = self.discriminator_fc(discriminator_output)

        return discriminator_logits, rationale

    def generator_pos_trainable_variables(self):
        """
        Return a list of trainable variables of the pos generator.
        """
        variables = self.generator_pos.trainable_variables + self.generator_pos_fc.trainable_variables
        variables += self.gen_pos_embedding_layer.trainable_variables
        return variables

    def generator_neg_trainable_variables(self):
        """
        Return a list of trainable variables of the neg generator.
        """
        variables = self.generator_neg.trainable_variables + self.generator_neg_fc.trainable_variables
        variables += self.gen_neg_embedding_layer.trainable_variables
        return variables

    def discriminator_trainable_variables(self):
        """
        Return a list of trainable variables of the discriminator.
        """
        variables = self.discriminator.trainable_variables + self.discriminator_fc.trainable_variables
        variables += self.dis_embedding_layer.trainable_variables
        return variables
