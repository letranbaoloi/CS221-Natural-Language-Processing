#!/usr/bin/env python
import argparse
import logging
import input_reader as Input
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import numpy as np
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def model_SemEval(seq):
  model_path = "/content/drive/MyDrive/UIT/Year3/HKI/CS221_NLP/Project_CS221/code/model/model_SemEval_task5"
  maxlen = 50
  word_list_path = "/content/drive/MyDrive/UIT/Year3/HKI/CS221_NLP/Project_CS221/code/SKS/data/word_list/word_all.txt"
  vocab_path = "/content/drive/MyDrive/UIT/Year3/HKI/CS221_NLP/Project_CS221/code/log/output_dir_SemEval/vocab.pkl"
  test_x, task_idx_test, ruling_embedding_test, category_embedding_test = Input.get_data(maxlen, word_list_path, vocab_path, seq)
  model2 = tf.keras.models.load_model(model_path)
  predictions = model2.predict([test_x, task_idx_test, ruling_embedding_test])
  if np.argmax(np.array(predictions)):
    return "Hate speech"
  else:
    return "Non-hate speech"
def model_davidson(seq):
  model_path = "/content/drive/MyDrive/UIT/Year3/HKI/CS221_NLP/Project_CS221/code/model/model_davidson"
  maxlen = 50
  word_list_path = "/content/drive/MyDrive/UIT/Year3/HKI/CS221_NLP/Project_CS221/code/SKS/data/word_list/word_all.txt"
  vocab_path = "/content/drive/MyDrive/UIT/Year3/HKI/CS221_NLP/Project_CS221/code/log/output_dir_davidson/vocab.pkl"
  test_x, task_idx_test, ruling_embedding_test, category_embedding_test = Input.get_data(maxlen, word_list_path, vocab_path, seq)
  model2 = tf.keras.models.load_model(model_path)
  predictions = model2.predict([test_x, task_idx_test, ruling_embedding_test])
  if np.argmax(np.array(predictions)):
    return "Hate speech"
  else:
    return "Non-hate speech"