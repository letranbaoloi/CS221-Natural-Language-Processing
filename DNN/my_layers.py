import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate

class BaseLayer(layers.Layer):
    def build_layers(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)

class ExpertModuleTrm(layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.layers = [
            MultiHeadAttention(4, 100),
            Dropout(0.1),
            Dense(400, activation='relu'),
            GlobalMaxPooling1D(),
            GlobalAveragePooling1D(),
            Concatenate(),
            Dropout(0.1),
            Dense(units[0], activation='relu'),
            Dense(units[1], activation='relu'),
            Dropout(0.1)
        ]
        super(ExpertModuleTrm, self).__init__(**kwargs)

    def call(self, inputs):
        xs = self.layers[0](inputs)
        xs = self.layers[1](xs)
        xs = self.layers[2](xs)
        xs_max = self.layers[3](xs)
        xs_avg = self.layers[4](xs)
        xs = self.layers[5]([xs_max, xs_avg])
        for layer in self.layers[6:]:
            xs = layer(xs)
        return xs

    def compute_output_shape(self, input_shape):
        return input_shape[0] + [self.units[1]]

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

class GateModule(BaseLayer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.layers = [
            MultiHeadAttention(4, 100),
            Dropout(0.1),
            Dense(400, activation='relu'),
            GlobalMaxPooling1D(),
            GlobalAveragePooling1D(),
            Concatenate(),
            Dropout(0.1),
            Dense(units[0], activation='relu'),
            Dense(units[0], activation='relu'),
            Dropout(0.1),
            Dense(units[1], activation='softmax')
        ]
        super(GateModule, self).__init__(**kwargs)

    def call(self, inputs):
        xs = self.layers[0](inputs)
        xs = self.layers[1](xs)
        xs = self.layers[2](xs)
        xs_max = self.layers[3](xs)
        xs_avg = self.layers[4](xs)
        xs = self.layers[5]([xs_max, xs_avg])
        for layer in self.layers[6:]:
            xs = layer(xs)
        return xs

    def compute_output_shape(self, input_shape):
        return input_shape[0] + [self.units[-1]]

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

class HSMMBottom(BaseLayer):
    def __init__(self, model_type, non_gate, expert_units, gate_unit=100, task_num=2, expert_num=3, **kwargs):
        self.model_type = model_type
        self.non_gate = non_gate
        self.gate_unit = gate_unit
        self.expert_units = expert_units
        self.task_num = task_num
        self.expert_num = expert_num
        self.experts = []
        self.gates = []
        super(HSMMBottom, self).__init__(**kwargs)

    def build(self, input_shape):
        for i in range(self.expert_num):
            expert = ExpertModuleTrm(units=self.expert_units)
            expert.build(input_shape)
            self.experts.append(expert)
        for i in range(self.task_num):
            gate = GateModule(units=[self.gate_unit, self.expert_num])
            gate.build(input_shape)
            self.gates.append(gate)
        super(HSMMBottom, self).build(input_shape)

    def call(self, inputs):
        expert_outputs = [expert(inputs) for expert in self.experts]

        gate_outputs = []
        if self.non_gate:
            print('No gating')
            expert_output = tf.stack(expert_outputs, axis=1)
            m1 = tf.reduce_mean(expert_output, axis=1)
            outputs = tf.stack([m1, m1], axis=1)
            return outputs
        else:
            for gate in self.gates:
                gate_outputs.append(gate(inputs))
            expert_output = tf.stack(expert_outputs, axis=1)
            gate_output = tf.stack(gate_outputs, axis=1)
            outputs = tf.matmul(gate_output, expert_output)
            return outputs

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.task_num, self.expert_units[-1]]

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_type': self.model_type,
            'non_gate': self.non_gate,
            'gate_unit': self.gate_unit,
            'expert_units': self.expert_units,
            'task_num': self.task_num,
            'expert_num': self.expert_num,
            'experts': self.experts,
            'gates': self.gates
        })
        return config

class HSMMTower(BaseLayer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.layers = [Dense(unit, activation='relu') for unit in units[:-1]]
        self.layers.extend([Dropout(0.1), Dense(units[-1], activation='softmax')])
        super(HSMMTower, self).__init__(**kwargs)

    def build(self, input_shape):
        self.build_layers(input_shape)
        super(HSMMTower, self).build(input_shape)

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.units[-1]]

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

class MultiHeadAttention(layers.Layer):
    def __init__(self, heads, head_size, output_dim=None, **kwargs):
        self.heads = heads
        self.head_size = head_size
        self.output_dim = output_dim or heads * head_size
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(3, input_shape[2], self.head_size),
                                      initializer='uniform', trainable=True)
        self.dense = self.add_weight(name='dense', shape=(input_shape[2], self.output_dim),
                                     initializer='uniform', trainable=True)
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, x):
        out = []
        for i in range(self.heads):
            WQ = K.dot(x, self.kernel[0])
            WK = K.dot(x, self.kernel[1])
            WV = K.dot(x, self.kernel[2])

            QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
            QK = QK / (100 ** 0.5)
            QK = K.softmax(QK)

            V = K.batch_dot(QK, WV)
            out.append(V)
        out = Concatenate(axis=-1)(out)
        out = K.dot(out, self.dense)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim

    def get_config(self):
        config = super().get_config()
        config.update({'heads': self.heads, 'head_size': self.head_size, 'output_dim': self.output_dim})
        return config
