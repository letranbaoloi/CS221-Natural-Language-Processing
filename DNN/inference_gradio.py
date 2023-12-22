#!/usr/bin/env python
import os.path

import input_reader as Input
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def model_SemEval(path_model):
  return tf.keras.models.load_model(path_model)
def model_davidson(path_model):
  return tf.keras.models.load_model(path_model)
def process_input(seq, vocab_path, word_list_path="data/word_list/word_all.txt", maxlen = 50):
  test_x, task_idx_test, ruling_embedding_test, category_embedding_test = Input.get_data(maxlen, word_list_path, vocab_path, seq)
  return [test_x, task_idx_test, ruling_embedding_test]