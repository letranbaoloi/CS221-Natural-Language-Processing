import logging
import numpy as np
from keras.utils import pad_sequences
from data_reader import load_vocab, get_indices

def get_data(maxlen, word_list_path, vocab_path, input):
    vocab = load_vocab(vocab_path)
    X_test_data, test_chars, ruling_embedding_test, category_embedding_test = get_indices([input], vocab, word_list_path)
    task_idx_test = np.tile(np.array([1., 0.]), (1, 1))
    X_test_data = pad_sequences(X_test_data, maxlen=maxlen)
    category_embedding_test = pad_sequences(category_embedding_test, maxlen=maxlen)

    return X_test_data, task_idx_test, np.array(ruling_embedding_test, dtype=float), category_embedding_test