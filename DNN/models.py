import logging
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, concatenate, Dense, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D
from my_layers import HSMMBottom, HSMMTower, MultiHeadAttention
from w2vEmbReader import W2VEmbReader as EmbReader
from keras.models import load_model
logger = logging.getLogger(__name__)

class CustomModelBuilder:
    def __init__(self):
        self.model = None
    def create_model(self, args, overal_maxlen, ruling_dim, vocab, num_class):
        self.args = args
        self.overal_maxlen = overal_maxlen
        self.ruling_dim = ruling_dim
        self.vocab = vocab
        self.num_class = num_class
        if self.args.model_type == 'cls':
            raise NotImplementedError
        elif self.args.model_type == 'HHMM_transformer':
            self.build_hhmm_transformer()
        elif self.args.model_type == 'Trm':
            self.build_trm()
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")

        if self.args.emb_path and self.args.model_type not in {'FNN', 'CNN', 'HHMM'}:
            self.initialize_embeddings()

        return self.model

    def expand_dim(self, x):
        return tf.expand_dims(x, 1)

    def matmul(self, conv_output, swem_output, gate_output):
        return tf.linalg.matmul(tf.stack([conv_output, swem_output], axis=1), gate_output)

    def build_hhmm_transformer(self):
        logger.info('Building a HHMM_transformer')
        task_num = 2
        sequence_input_word = Input(shape=(self.overal_maxlen,), dtype='int32', name='sequence_input')
        taskid_input = Input(shape=(task_num,), dtype='float32', name='taskid_input')
        ruling_input = Input(shape=(self.ruling_dim,), dtype='float32')

        embedded_sequences_word = Embedding(len(self.vocab), self.args.emb_dim, name='emb')(sequence_input_word)
        emb_ruling = Embedding(len(self.vocab), 100)(ruling_input)
        emb_output = concatenate([embedded_sequences_word, emb_ruling], axis=-1)

        tower_outputs = []
        expert_outputs = HSMMBottom(self.args.model_type, self.args.non_gate, expert_units=[150, 150], gate_unit=150, task_num=task_num)(emb_output)

        for i in range(task_num):
            tower_outputs.append(HSMMTower(units=[50, 2])(expert_outputs[:, i, :]))

        out = tf.linalg.matmul(tf.stack(tower_outputs, axis=-1), tf.expand_dims(taskid_input, -1))
        pred = tf.squeeze(out, axis=-1)

        self.model = tf.keras.Model(inputs=[sequence_input_word, taskid_input, ruling_input], outputs=pred)
        self.model.emb_index = 0
        self.model.summary()

    def build_trm(self):
        logger.info("Building a Simple Word Embedding Model")
        input = Input(shape=(self.overal_maxlen,), dtype='int32')
        emb_output = Embedding(len(self.vocab), self.args.emb_dim, name='emb')(input)
        mlp_output = MultiHeadAttention(300)(emb_output)
        mlp_output = Dense(300, activation='relu')(mlp_output)
        avg = GlobalAveragePooling1D()(mlp_output)
        max1 = GlobalMaxPooling1D()(mlp_output)
        concat = concatenate([avg, max1], axis=-1)
        dense1 = Dense(50, activation='relu')(concat)
        dense2 = Dense(50, activation='relu')(dense1)
        dropout = Dropout(0.5)(dense2)
        output = Dense(self.num_class, activation='softmax')(dropout)
        self.model = tf.keras.Model(inputs=input, outputs=output)
        self.model.emb_index = 1
        self.model.summary()

    def initialize_embeddings(self):
        logger.info('Initializing lookup table')
        emb_reader = EmbReader(self.args.emb_path, emb_dim=self.args.emb_dim)
        self.model.get_layer(name='emb').set_weights(emb_reader.get_emb_matrix_given_vocab(self.vocab, self.model.get_layer(name='emb').get_weights()))  # Upgrade to 2.0.8

    def save_model_weights(self, save_path):
        """
        Save the model weights to a file.

        Parameters:
        - save_path (str): Path to save the weights file.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call create_model() before saving weights.")

        self.model.save_weights(save_path)
        print(f"Model weights saved to {save_path}")

    def load_model_weights(self, weights_path):
        """
        Load the model weights from a file.

        Parameters:
        - weights_path (str): Path to the weights file.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call create_model() before loading weights.")

        self.model.load_weights(weights_path)
        print(f"Model weights loaded from {weights_path}")

    def save_model_architecture(self, save_path):
        """
        Save the model architecture to a file.

        Parameters:
        - save_path (str): Path to save the architecture file.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call create_model() before saving architecture.")

        with open(save_path, "w") as json_file:
            json_file.write(self.model.to_json())
        print(f"Model architecture saved to {save_path}")

    def load_model_architecture(self, json_path):
        """
        Load the model architecture from a file.

        Parameters:
        - json_path (str): Path to the architecture file.
        """
        if self.model is not None:
            raise ValueError("Model is already built. Call create_model() with a new configuration.")

        with open(json_path, "r") as json_file:
            json_config = json_file.read()

        self.model = tf.keras.models.model_from_json(json_config)
        print(f"Model architecture loaded from {json_path}")