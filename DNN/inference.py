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

logger = logging.getLogger(__name__)
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",dest="model_path", type=str, metavar='<str>', help="(required) The path to model")
    parser.add_argument("--word_list", dest="word_list_path", type=str, metavar='<str>', help="The path to the sarcasm data set")
    parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', help="(Optional) The path to the existing vocab file (*.pkl)")
    parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=50, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
    return parser.parse_args()
def main():
    args = parse_arguments()
    input = "fuck shit bitch"
    test_x, task_idx_test, ruling_embedding_test, category_embedding_test = Input.get_data(args, input)
    model2 = tf.keras.models.load_model(args.model_path)
    predictions = model2.predict([test_x, task_idx_test, ruling_embedding_test])
    print("label: ", np.argmax(np.array(predictions)))
    print("score: ",predictions)
if __name__ == "__main__":
    main()