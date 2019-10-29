import tensorflow as tf


def get_sparsity_loss(z, mask, level):
    """
    Exact sparsity loss in a batchwise sense. 
    Inputs: 
        z -- (batch_size, sequence_length)
        mask -- (batch_size, seq_length)
        level -- sparsity level
    """
    sparsity = tf.reduce_sum(z) / tf.reduce_sum(mask)
    return tf.abs(sparsity - level)


def get_continuity_loss(z):
    """
    Compute the continuity loss.
    Inputs:     
        z -- (batch_size, sequence_length)
    """
    return tf.reduce_mean(tf.abs(z[:, 1:] - z[:, :-1]))


def compute_accuracy(logits, labels):
    """
    Compute the batch-wise accuracy. 
    Inputs:         
        logits -- (batch_size, num_classes)    
        labels -- need to be one-hot / soft one-hot.
                  (batch_size, num_classes)    
    """
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    labels_ = tf.argmax(labels, axis=1, output_type=tf.int32)
    batch_size = int(logits.shape[0])
    return tf.reduce_sum(
        tf.cast(tf.equal(predictions, labels_), dtype=tf.float32)) / batch_size


def compute_micro_stats(labels, predictions):
    """
    Inputs:
        labels binary sequence indicates the if it is rationale
        predicitions -- sequence indicates the probability of being rationale
    
        labels -- (batch_size, sequence_length) 
        predictions -- (batch_size, sequence_length) in soft probability
    
    Outputs:
        Number of true positive among predicition (True positive)
        Number of predicted positive (True pos + false pos)
        Number of real positive in the labels (true pos + false neg)
    """
    labels = tf.cast(labels, tf.float32)
    predictions = tf.cast(predictions, tf.float32)

    # threshold predictions
    predictions = tf.cast(tf.greater_equal(predictions, 0.5), tf.float32)

    # cal precision, recall
    num_true_pos = tf.reduce_sum(labels * predictions)
    num_predicted_pos = tf.reduce_sum(predictions)
    num_real_pos = tf.reduce_sum(labels)

    return num_true_pos, num_predicted_pos, num_real_pos


def compute_detail_micro_stats(labels, predictions):
    """
    Inputs:
        labels binary sequence indicates the if it is rationale
        predicitions -- sequence indicates the probability of being rationale
    
        labels -- (batch_size, sequence_length) 
        predictions -- (batch_size, sequence_length) in soft probability
    
    Outputs:
        Number of true positive among predicition (True positive) in each examples
        Number of predicted positive (True pos + false pos) in each examples
        Number of real positive in the labels (true pos + false neg) in each examples
    """
    labels = tf.cast(labels, tf.float32)
    predictions = tf.cast(predictions, tf.float32)

    # threshold predictions
    predictions = tf.cast(tf.greater_equal(predictions, 0.5), tf.float32)

    # (batch_size, )
    true_pos = tf.reduce_sum(labels * predictions, axis=-1)
    predicted_pos = tf.reduce_sum(predictions, axis=-1)
    real_pos = tf.reduce_sum(labels, axis=-1)

    return true_pos, predicted_pos, real_pos
