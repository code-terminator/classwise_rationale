import sys
import random
import tensorflow as tf
from visualize import show_binary_rationale, show_binary_rationale_with_annotation
from metric import compute_micro_stats, compute_detail_micro_stats


def flush(model, dataset, idx2word, file):
    """
    Feedforward inference of the dataset and plot to file. 
    """
    f = open(file, "wt")

    neg_predicted_words = 0.
    pos_predicted_words = 0.
    num_words = 0

    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        batch_size = inputs.get_shape()[0]

        # (batch_size, 1)
        all_ones = tf.expand_dims(tf.ones(batch_size), axis=-1)
        all_zeros = tf.zeros(all_ones.shape)

        label_zeros = tf.cast(tf.concat([all_ones, all_zeros], axis=-1),
                              tf.int32)
        label_ones = tf.cast(tf.concat([all_zeros, all_ones], axis=-1),
                             tf.int32)

        # go through both G0 & G1
        # rationales -- (batch_size, seq_length, 2)
        _, neg_rationales = model(inputs, masks, label_zeros, path=0)
        _, pos_rationales = model(inputs, masks, label_ones, path=1)

        np_inputs = inputs.numpy()
        np_neg_rationales = neg_rationales[:, :, 1].numpy()
        np_pos_rationales = pos_rationales[:, :, 1].numpy()

        for idx in range(batch_size):
            f.write('===== true label: %d ===== \n' % tf.argmax(labels[idx, :]))
            f.write(" ---------------- negative ----------------- \n")
            neg_plot = show_binary_rationale(np_inputs[idx, :],
                                             np_neg_rationales[idx, :],
                                             idx2word,
                                             tofile=True)
            f.write(neg_plot)
            f.write("\n ---------------- positve ----------------- \n")
            pos_plot = show_binary_rationale(np_inputs[idx, :],
                                             np_pos_rationales[idx, :],
                                             idx2word,
                                             tofile=True)
            f.write(pos_plot)
            f.write("\n")

        neg_predicted_words += tf.reduce_sum(neg_rationales[:, :, 1])
        pos_predicted_words += tf.reduce_sum(pos_rationales[:, :, 1])
        num_words += tf.reduce_sum(masks)

    neg_sparsity = neg_predicted_words / tf.cast(num_words, tf.float32)
    pos_sparsity = pos_predicted_words / tf.cast(num_words, tf.float32)

    output_string = "Actual negative sparsity: %4f%%\n" % (100 * neg_sparsity)
    output_string += "Actual positive sparsity: %4f%%\n" % (100 * pos_sparsity)
    f.write(output_string)
    f.close()


def validate(model, annotation_dataset, idx2word, visual_interval=5, file=None):
    """
    Compared to validate, it outputs both factual and counter to a file.
    """
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    if file:
        f = open(file, "wt")

    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_dataset):
        batch_size = inputs.get_shape()[0]

        # (batch_size, 1)
        all_ones = tf.expand_dims(tf.ones(batch_size), axis=-1)
        all_zeros = tf.zeros(all_ones.shape)

        label_zeros = tf.cast(tf.concat([all_ones, all_zeros], axis=-1),
                              tf.int32)
        label_ones = tf.cast(tf.concat([all_zeros, all_ones], axis=-1),
                             tf.int32)

        # go through both G0 & G1
        # rationales -- (batch_size, seq_length, 2)
        _, neg_rationales = model(inputs, masks, label_zeros, path=0)
        _, pos_rationales = model(inputs, masks, label_ones, path=1)

        neg_mask = tf.cast(tf.expand_dims(tf.expand_dims(labels[:, 0], -1), -1),
                           dtype=tf.float32)
        pos_mask = tf.cast(tf.expand_dims(tf.expand_dims(labels[:, 1], -1), -1),
                           dtype=tf.float32)

        # factual rationale (rationales that consist with the label)
        factual_rationales = neg_mask * neg_rationales + pos_mask * pos_rationales
        counter_rationales = pos_mask * neg_rationales + neg_mask * pos_rationales

        num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
            annotations, factual_rationales[:, :, 1])

        num_true_pos += num_true_pos_
        num_predicted_pos += num_predicted_pos_
        num_real_pos += num_real_pos_
        num_words += tf.reduce_sum(masks)

        if file:
            np_inputs = inputs.numpy()
            # annotation -- (batch_size, )
            np_annotations = annotations.numpy()
            np_factual_rationales = factual_rationales[:, :, 1].numpy()
            np_counter_rationales = counter_rationales[:, :, 1].numpy()

            for idx in range(batch_size):
                f.write('===== true label: %d ===== \n' %
                        tf.argmax(labels[idx, :]))
                f.write("------ factual prediction ------\n")
                rationale_plot = show_binary_rationale_with_annotation(
                    np_inputs[idx, :],
                    np_factual_rationales[idx, :],
                    np_annotations[idx, :],
                    idx2word,
                    tofile=True)
                f.write(rationale_plot)
                f.write("\n")

                # plot counter facutal rationale
                f.write("------ counterfactual prediction ------\n")
                rationale_plot = show_binary_rationale_with_annotation(
                    np_inputs[idx, :],
                    np_counter_rationales[idx, :],
                    np_annotations[idx, :],
                    idx2word,
                    tofile=True)
                f.write(rationale_plot)
                f.write("\n")

        if (batch + 1) % visual_interval == 0:
            np_inputs = inputs.numpy()
            # annotation -- (batch_size, )
            np_annotations = annotations.numpy()
            np_factual_rationales = factual_rationales[:, :, 1].numpy()

            # plot the factual rationale
            idx = random.randint(0, batch_size - 1)
            show_binary_rationale_with_annotation(np_inputs[idx, :],
                                                  np_factual_rationales[idx, :],
                                                  np_annotations[idx, :],
                                                  idx2word,
                                                  tofile=False)

    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                       micro_recall)
    sparsity = num_predicted_pos / tf.cast(num_words, tf.float32)

    output_string = 'Validate rationales: precision: %.4f, recall: %.4f, f1: %4f%%\n' % (
        100 * micro_precision, 100 * micro_recall, 100 * micro_f1)
    output_string += "Actual sparsity: %4f%%\n" % (100 * sparsity)

    if file:
        f.write(output_string)
        f.close()
    else:
        print(output_string)
        sys.stdout.flush()

    return sparsity, micro_precision, micro_recall, micro_f1
