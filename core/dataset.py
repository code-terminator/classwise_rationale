import os
import six
import csv
import numpy as np
import tensorflow as tf
from language import LanguageIndex, get_pretained_glove


class DataProcessor(object):
    """
    Base class for data converters for sequence classification data sets.
    """

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _convert_to_unicode(cls, text):
        """
        Converts `text` to Unicode (if it's not already), assuming utf-8 input.
        """
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")


def convert_single_text(text, max_seq_length, word2idx):
    """
    Converts a single text into a list of ids with mask.
    """
    input_ids = []

    text_ = text.strip().split(" ")

    if len(text_) > max_seq_length:
        text_ = text_[0:max_seq_length]

    for word in text_:
        word = word.strip()
        try:
            input_ids.append(word2idx[word])
        except:
            # if the word is not exist in word2idx, use <unknown> token
            input_ids.append(1)

    # The mask has 1 for real tokens and 0 for padding tokens.
    input_mask = [1] * len(input_ids)

    # zero-pad up to the max_seq_length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return input_ids, input_mask


def convert_examples_to_np_arrays(examples, max_seq_length, word2idx):
    """
    Convert a set of train/dev examples numpy arrays.
    Outputs: 
        data -- (num_examples, max_seq_length).
        masks -- (num_examples, max_seq_length).
        labels -- (num_examples, num_classes) in a one-hot format.
    """

    data = []
    labels = []
    masks = []
    for example in examples:
        input_ids, input_mask = convert_single_text(example["text"],
                                                    max_seq_length, word2idx)

        data.append(input_ids)
        masks.append(input_mask)
        labels.append(example["label"])

    data = np.array(data, dtype=np.int32)
    masks = np.array(masks, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)

    return data, masks, labels


def get_dataset(train_examples, dev_examples, max_seq_length, word_threshold):
    """
    Return tf datasets (train and dev) and language index. 
    """
    # create language index
    texts = [data["text"] for data in train_examples
            ] + [data["text"] for data in dev_examples]
    language_index = LanguageIndex(texts, word_threshold)

    # convert examples to index
    train_np_arrays = convert_examples_to_np_arrays(train_examples,
                                                    max_seq_length,
                                                    language_index.word2idx)

    dev_np_arrays = convert_examples_to_np_arrays(dev_examples, max_seq_length,
                                                  language_index.word2idx)
    # convert data into tf dataset format
    train_dataset = tf.data.Dataset.from_tensor_slices(train_np_arrays)
    dev_dataset = tf.data.Dataset.from_tensor_slices(dev_np_arrays)

    return train_dataset, dev_dataset, language_index
