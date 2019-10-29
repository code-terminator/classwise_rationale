import numpy as np


class LanguageIndex(object):
    """
    Creates a word -> index mapping (e.g,. "dad" -> 5) 
    and vice-versa.
    """

    def __init__(self, texts, threshold=1):
        """
        Inputs: 
            texts -- a list of text (after tokenization)
            threshold -- threshold to filter less frequent words
        """
        self.threshold = threshold

        self.word2idx = {}
        self.idx2word = {}
        self._create_index(texts)

    def _create_index(self, texts):

        # counting for unique words
        word2count = {}
        for text in texts:
            for word in text.split(' '):
                if word in word2count:
                    word2count[word] += 1
                else:
                    word2count[word] = 1

        # counting unqiue words
        vocab = set()
        for word, count in word2count.items():
            if count >= self.threshold:
                vocab.add(word)
        vocab = sorted(vocab)

        # create word2idx
        self.word2idx["<pad>"] = 0
        self.word2idx["<unknown>"] = 1
        for index, word in enumerate(vocab):
            self.word2idx[word] = index + 2

        # create reverse index
        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def get_pretained_glove(word2idx, fpath):
    """
    Construct a numpy embedding matrix. 
    The column number indicates the word index.
    For the words do not appear in pretrained embeddings, 
    we use random embeddings.

    Inputs:
        word2idx -- a dictionary, key -- word, value -- word index
        fpath -- the path of pretrained embedding.
    Outputs:
        embedding_matrix -- an ordered numpy array, 
                            shape -- (embedding_dim, len(word2idx))
    """

    def load_glove_embedding():
        """
        Load glove embedding from disk. 
        """
        word2embedding = {}
        with open(fpath, "r", errors='ignore') as f:
            for (i, line) in enumerate(f):
                data = line.strip().split(" ")
                word = data[0].strip()
                embedding = list(map(float, data[1:]))
                word2embedding[word] = np.array(
                    embedding)  # shape -- (embedding_dim, )
            embedding_dim = len(embedding)

        return word2embedding, embedding_dim

    # load glove embedding
    word2embedding, embedding_dim = load_glove_embedding()
    embedding_matrix = np.random.randn(embedding_dim, len(word2idx))

    # replace the embedding matrix by pretrained embedding
    counter = 0
    for word, index in word2idx.items():
        if word in word2embedding:
            embedding_matrix[:, index] = word2embedding[word]
            counter += 1

    # replace the embedding to all zeros for <pad>
    embedding_matrix[:, word2idx["<pad>"]] = np.zeros(embedding_dim)
    print("%d out of %d words are covered by the pre-trained embedding." %
          (counter, len(word2idx)))

    return embedding_matrix
