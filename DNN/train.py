#!/usr/bin/env python
import argparse
import logging
import os.path
import numpy as np
from time import time
import utils as U
import pickle as pk
from model_evaluator import Evaluator
from tensorflow.keras.callbacks import ModelCheckpoint
import data_reader as dataset
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.model_selection import StratifiedKFold
from models import CustomModelBuilder
import tensorflow as tf
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="data_path", type=str, metavar='<str>', required=True, help="The path to the data set")
    parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
    parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='SWEM', help="Model type (SWEM|regp|breg|bregp) (default=SWEM)")
    parser.add_argument("-l", "--loss", dest="loss", type=str, metavar='<str>', default='ce', help="Loss function (mse|ce) (default=ce)")
    parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=50)")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")
    parser.add_argument("--trial", dest="trial_data_path", type=str, metavar='<str>', help="The path to the trial data set")
    parser.add_argument("-s", "--sentiment", dest="sentiment_data_path", type=str, metavar='<str>', help="The path to the sentiment data set")
    parser.add_argument("--humor", dest="humor_data_path", type=str, metavar='<str>', help="The path to the humor data set")
    parser.add_argument("--sarcasm", dest="sarcasm_data_path", type=str, metavar='<str>', help="The path to the sarcasm data set")
    parser.add_argument("--word_list", dest="word_list_path", type=str, metavar='<str>', help="The path to the sarcasm data set")
    parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.1, help="The dropout probability. To disable, give a negative number (default=0.5)")
    parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', help="(Optional) The path to the existing vocab file (*.pkl)")
    parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file (Word2Vec format)")
    parser.add_argument("--lr", dest="learn_rate", type=float, metavar='<float>', help="the learn rate")
    parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=20, help="Number of epochs (default=60)")
    parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=50, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
    parser.add_argument("--word_norm", dest="word_norm", type=int, metavar='<int>', default=1, help="0-stemming, 1-lemma, other-do nothing")
    parser.add_argument("--non_gate", dest="non_gate", action='store_true', help="Model type (SWEM|regp|breg|bregp) (default=SWEM)")
    parser.add_argument("--cross_validation", dest="cross_validation", action='store_true', help="Perform cross-validation training")
    parser.add_argument("--retrain_cross_validation", dest="retrain_cross_validation", action='store_true', help="Perform cross-validation training")
    parser.add_argument("--model_retrain_cross_validation_path", dest="model_retrain_cross_validation_path", default="", help="model to train again")
    return parser.parse_args()

def prepare_data(args):
    return dataset.get_data(args)

def optimizer_selection():
    return 'rmsprop'

def build_model(args, ruling_embedding_test, vocab, output_dim):
    if args.loss == 'mse':
        loss = 'mean_squared_error'
        metric = 'mean_absolute_error'
    else:
        loss = 'categorical_crossentropy'
        metric = 'accuracy'

    model = CustomModelBuilder().create_model(args, args.maxlen, len(ruling_embedding_test[0]), vocab, output_dim)
    model.compile(loss=loss, optimizer=optimizer_selection(), metrics=[metric])
    return model

def save_model_architecture(model, out_dir):
    with open(out_dir + '/model_arch.json', 'w') as arch:
        arch.write(model.to_json(indent=2))
def main():
    args = parse_arguments()
    out_dir = args.out_dir_path
    U.set_logger(out_dir=out_dir, model_type=args.model_type)
    U.print_args(args)

    assert args.loss in {'mse', 'ce'}
    if args.retrain_cross_validation:
        retrain_cross_validataion(args)
    elif args.cross_validation:
        train_with_cross_validation(args)
    else:
        train_normal(args)

def retrain_cross_validataion(args):
    train_x, test_x, train_y, test_y, train_chars, test_chars, task_idx_train, task_idx_test, ruling_embedding_train, ruling_embedding_test,\
        category_embedding_train, category_embedding_test, vocab = prepare_data(args)

    if not args.vocab_path:
        with open(args.out_dir_path + '/vocab.pkl', 'wb') as vocab_file:
            pk.dump(vocab, vocab_file)
    bincounts, mfs_list = U.bincounts(train_y)
    with open('%s/bincounts.txt' % args.out_dir_path, 'w') as output_file:
        for bincount in bincounts:
            output_file.write(str(bincount) + '\n')

    logger.info('Statistics:')
    logger.info('  train_x shape: ' + str(np.array(train_x).shape))
    logger.info('  test_x shape:  ' + str(np.array(test_x).shape))
    logger.info('  train_chars shape: ' + str(np.array(train_chars).shape))
    logger.info('  test_chars shape:  ' + str(np.array(test_chars).shape))
    logger.info('  train_y shape: ' + str(train_y.shape))
    logger.info('  test_y shape:  ' + str(test_y.shape))

    optimizer = optimizer_selection()

    # model = build_model(args, ruling_embedding_test, vocab, len(train_y[0]))
    model = tf.keras.models.load_model(args.model_retrain_cross_validation_path)

    logger.info('Saving model architecture')
    save_model_architecture(model, args.out_dir_path)

    evl = Evaluator(args, dataset, args.out_dir_path, test_x, test_chars, task_idx_test, ruling_embedding_test, test_y, args.batch_size)

    total_train_time = 0
    total_eval_time = 0
    t1 = time()

    for ii in range(args.epochs):
        t0 = time()
        if args.model_type in {'CNN'}:
            train_history = model.fit(train_chars, train_y, batch_size=args.batch_size, epochs=1, validation_data=(test_chars, test_y), verbose=1)
        elif args.model_type in {'HHMM', 'HHMM_transformer'}:
            train_history = model.fit([train_x, task_idx_train, ruling_embedding_train], train_y, batch_size=args.batch_size, epochs=1, validation_data=([test_x, task_idx_test, ruling_embedding_test], test_y), verbose=1)
        else:
            train_history = model.fit(train_x, train_y, batch_size=args.batch_size, epochs=1, validation_data=(test_x, test_y), verbose=1)
        tr_time = time() - t0
        total_train_time += tr_time

        t0 = time()
        evl.evaluate(model, ii)
        evl_time = time() - t0
        total_eval_time += evl_time
        total_time = time()-t1

        train_loss = train_history.history['loss'][0]
        if args.loss == 'mse':
            train_metric = train_history.history['mean_absolute_error'][0]
        else:
            train_metric = train_history.history['accuracy'][0]
        logger.info('Epoch %d, train: %is, evaluation: %is, total_time: %is' % (ii, tr_time, evl_time, total_time))
        logger.info('[Train] loss: %.4f, metric: %.4f' % (train_loss, train_metric))
        evl.print_info()

    logger.info('Training:   %i seconds in total' % total_train_time)
    logger.info('Evaluation: %i seconds in total' % total_eval_time)

    evl.print_final_info()
def train_with_cross_validation(args):
    train_x, test_x, train_y, test_y, train_chars, test_chars, task_idx_train, task_idx_test, ruling_embedding_train, ruling_embedding_test,\
        category_embedding_train, category_embedding_test, vocab = prepare_data(args)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(skf.split(train_x, np.argmax(train_y, axis=1))):
        fold_out_dir = os.path.join(args.out_dir_path, f'fold_{fold+1}')
        U.mkdir_p(fold_out_dir)
        U.set_logger(out_dir=fold_out_dir, model_type=args.model_type)

        train_x_fold, val_x_fold = train_x[train_index], train_x[val_index]
        train_y_fold, val_y_fold = train_y[train_index], train_y[val_index]
        train_chars_fold, val_chars_fold = train_chars[train_index], train_chars[val_index]
        task_idx_train_fold, ruling_embedding_train_fold = task_idx_train[train_index], ruling_embedding_train[train_index]
        task_idx_val_fold, ruling_embedding_val_fold = task_idx_train[val_index], ruling_embedding_train[val_index]

        model = build_model(args, ruling_embedding_test, vocab, len(train_y_fold[0]))

        if not args.vocab_path:
            with open(fold_out_dir + '/vocab.pkl', 'wb') as vocab_file:
                pk.dump(vocab, vocab_file)

        bincounts, mfs_list = U.bincounts(train_y_fold)
        with open(f'{fold_out_dir}/bincounts.txt', 'w') as output_file:
            for bincount in bincounts:
                output_file.write(str(bincount) + '\n')

        logger.info(f'Training fold {fold + 1}/{skf.get_n_splits()}')

        # Train with the ModelCheckpoint callback
        if args.model_type in {'CNN'}:
            history = model.fit(train_chars_fold, train_y_fold, batch_size=args.batch_size, epochs=args.epochs,
                            validation_data=(val_chars_fold, val_y_fold), verbose=1)
        elif args.model_type in {'HHMM', 'HHMM_transformer'}:
            history = model.fit([train_x_fold, task_idx_train_fold, ruling_embedding_train_fold], train_y_fold,
                                batch_size=args.batch_size, epochs=args.epochs,
                                validation_data=([val_x_fold, task_idx_val_fold, ruling_embedding_val_fold], val_y_fold),
                                verbose=1)
        else:
            history = model.fit(train_x_fold, train_y_fold, batch_size=args.batch_size, epochs=args.epochs,
                                validation_data=(val_x_fold, val_y_fold), verbose=1)

        logger.info(f'Saving model architecture for fold {fold + 1}')
        save_model_architecture(model, fold_out_dir)

        # Save the entire model in TensorFlow SavedModel format
        model.save(os.path.join(fold_out_dir, f'my_model_{fold+1}/'), save_format="tf")

        # Evaluate the model on the test set after cross-validation
        print("Evaluate the model on the test set after cross-validation")
        evl_test = Evaluator(args, dataset, args.out_dir_path, test_x, test_chars, task_idx_test, ruling_embedding_test, test_y,
                             args.batch_size)
        evl_test.evaluate(model, args.epochs)
        evl_test.print_final_info()


def train_normal(args):
    train_x, test_x, train_y, test_y, train_chars, test_chars, task_idx_train, task_idx_test, ruling_embedding_train, ruling_embedding_test,\
        category_embedding_train, category_embedding_test, vocab = prepare_data(args)

    if not args.vocab_path:
        with open(args.out_dir_path + '/vocab.pkl', 'wb') as vocab_file:
            pk.dump(vocab, vocab_file)
    bincounts, mfs_list = U.bincounts(train_y)
    with open('%s/bincounts.txt' % args.out_dir_path, 'w') as output_file:
        for bincount in bincounts:
            output_file.write(str(bincount) + '\n')

    logger.info('Statistics:')
    logger.info('  train_x shape: ' + str(np.array(train_x).shape))
    logger.info('  test_x shape:  ' + str(np.array(test_x).shape))
    logger.info('  train_chars shape: ' + str(np.array(train_chars).shape))
    logger.info('  test_chars shape:  ' + str(np.array(test_chars).shape))
    logger.info('  train_y shape: ' + str(train_y.shape))
    logger.info('  test_y shape:  ' + str(test_y.shape))

    optimizer = optimizer_selection()

    model = build_model(args, ruling_embedding_test, vocab, len(train_y[0]))

    logger.info('Saving model architecture')
    save_model_architecture(model, args.out_dir_path)

    evl = Evaluator(args, dataset, args.out_dir_path, test_x, test_chars, task_idx_test, ruling_embedding_test, test_y, args.batch_size)

    total_train_time = 0
    total_eval_time = 0
    t1 = time()

    for ii in range(args.epochs):
        t0 = time()
        if args.model_type in {'CNN'}:
            train_history = model.fit(train_chars, train_y, batch_size=args.batch_size, epochs=1, validation_data=(test_chars, test_y), verbose=1)
        elif args.model_type in {'HHMM', 'HHMM_transformer'}:
            train_history = model.fit([train_x, task_idx_train, ruling_embedding_train], train_y, batch_size=args.batch_size, epochs=1, validation_data=([test_x, task_idx_test, ruling_embedding_test], test_y), verbose=1)
        else:
            train_history = model.fit(train_x, train_y, batch_size=args.batch_size, epochs=1, validation_data=(test_x, test_y), verbose=1)
        tr_time = time() - t0
        total_train_time += tr_time

        t0 = time()
        evl.evaluate(model, ii)
        evl_time = time() - t0
        total_eval_time += evl_time
        total_time = time()-t1

        train_loss = train_history.history['loss'][0]
        if args.loss == 'mse':
            train_metric = train_history.history['mean_absolute_error'][0]
        else:
            train_metric = train_history.history['accuracy'][0]
        logger.info('Epoch %d, train: %is, evaluation: %is, total_time: %is' % (ii, tr_time, evl_time, total_time))
        logger.info('[Train] loss: %.4f, metric: %.4f' % (train_loss, train_metric))
        evl.print_info()

    logger.info('Training:   %i seconds in total' % total_train_time)
    logger.info('Evaluation: %i seconds in total' % total_eval_time)

    evl.print_final_info()

if __name__ == "__main__":
    main()
