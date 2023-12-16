import logging
import numpy as np
from keras.utils import pad_sequences
from data_reader import load_vocab, get_indices
logger = logging.getLogger(__name__)

def read_dataset(args, input):
    vocab = load_vocab(args.vocab_path)
    X_test_data, test_chars, ruling_embedding_test, category_embedding_test = get_indices([input], vocab, args.word_list_path)

    task_idx_test = np.tile(np.array([1., 0.]), (1, 1))
    X_test_data = pad_sequences(X_test_data, maxlen=args.maxlen)
    category_embedding_test = pad_sequences(category_embedding_test, maxlen=args.maxlen)

    return X_test_data, task_idx_test, \
        np.array(ruling_embedding_test, dtype=float), category_embedding_test

def get_data(args, input):
    return read_dataset(args,input)