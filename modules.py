"""
Created on Oct 26, 2020

modules of DIN: attention mechanism

@author: Ziyao Geng
"""


import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, BatchNormalization, Dense


class Attention_Layer(Layer):
    def __init__(self, att_hidden_units, activation='prelu'):
        """
        """
        super(Attention_Layer, self).__init__()
        self.att_dense = [Dense(unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = Dense(1, activation=None)

    def call(self, inputs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2) 
        # value: hist items  (None, seq_len, d * 2) 
        # mask: (None, seq_len)
        q, k, v, mask, activation_values = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])  # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])  # (None, seq_len, d * 2)
        # 相当于复制很多份， 使得可以并行计算
        # q, k, out product should concat
        info = tf.concat([q, k, q - k, q * k], axis=-1) # None, sqe_len, d * 8

        # dense
        for dense in self.att_dense:
            info = dense(info)
        outputs = self.att_final_dense(info)  # (None, seq_len, 1)


        # 对于每一个instance，中的每一个items 得到对应的得分
        outputs = tf.squeeze(outputs, axis=-1)  # (None, seq_len)

        paddings = tf.zeros_like(outputs)
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # (None, seq_len)
        activation_values.append(outputs)
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len)

        outputs = tf.matmul(outputs, v)  # (None, 1, d * 2) # 正式计算得分
        outputs = tf.squeeze(outputs, axis=1)  # (None, d * 2)
        return outputs


class MyAttention(Layer):
    def __init__(self, att_hidden_units, activation='prelu'):
        """
        """
        super(MyAttention, self).__init__()
        self.att_dense = [Dense(unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = Dense(1, activation=None)
        self.dense_q1 = Dense(60, activation=activation)
        self.dense_q2 = Dense(120, activation=activation)

    def call(self, inputs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2)
        # value: hist items  (None, seq_len, d * 2)
        # mask: (None, seq_len)
        q, k, v, mask, activation_values = inputs
        q = self.dense_q1(q)
        q = self.dense_q2(q)
        q = tf.tile(q, multiples=[1, k.shape[1]])  # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])  # (None, seq_len, d * 2)
        # 相当于复制很多份， 使得可以并行计算
        # q, k, out product should concat
        info = tf.concat([q, k, q - k, q * k], axis=-1) # None, sqe_len, d * 8

        # dense
        for dense in self.att_dense:
            info = dense(info)
        outputs = self.att_final_dense(info)  # (None, seq_len, 1)


        # 对于每一个instance，中的每一个items 得到对应的得分
        outputs = tf.squeeze(outputs, axis=-1)  # (None, seq_len)

        paddings = tf.zeros_like(outputs)
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # (None, seq_len)
        activation_values.append(outputs)
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len)

        outputs = tf.matmul(outputs, v)  # (None, 1, d * 2) # 正式计算得分
        outputs = tf.squeeze(outputs, axis=1)  # (None, d * 2)
        return outputs



class Attention_Layer_for_AUGRU(Layer):
    def __init__(self, att_hidden_units, activation='prelu', embed_dim=None):
        """
        """
        super(Attention_Layer_for_AUGRU, self).__init__()
        self.att_dense = [Dense(unit, activation=activation) for unit in att_hidden_units]

        self.dense_q = Dense(embed_dim * 3, activation=activation)

        self.att_final_dense = Dense(1, activation=None)

    def call(self, inputs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2)
        # value: hist items  (None, seq_len, d * 2)
        # mask: (None, seq_len)
        q, k, v, mask = inputs
        q = self.dense_q(q)
        q = tf.tile(q, multiples=[1, k.shape[1]])  # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])  # (None, seq_len, d * 2)
        # 相当于复制很多份， 使得可以并行计算
        # q, k, out product should concat
        info = tf.concat([q, k, q - k, q * k], axis=-1) # None, sqe_len, d * 8

        # dense
        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)  # (None, seq_len, 1)
        # 对于每一个instance，中的每一个items 得到对应的得分
        outputs = tf.squeeze(outputs, axis=-1)  # (None, seq_len)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1) # (None, seq_len) 目的是不让无关的item影响 计算 e ** 无穷小 约等于0
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # (None, seq_len)

        # softmax
        outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len) 归一化得分 忽略被mask的部分
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len) # 这个是得分

        return outputs


class Attention_Layer_for_AUGRU2(Layer):
    def __init__(self, att_hidden_units, activation='prelu', embed_dim=None):
        """
        """
        super(Attention_Layer_for_AUGRU2, self).__init__()
        self.att_dense = [Dense(unit, activation=activation) for unit in att_hidden_units]

        self.att_final_dense = Dense(1, activation=None)

    def call(self, inputs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2)
        # value: hist items  (None, seq_len, d * 2)
        # mask: (None, seq_len)
        q, k, v, mask = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])  # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])  # (None, seq_len, d * 2)
        # 相当于复制很多份， 使得可以并行计算
        # q, k, out product should concat
        info = tf.concat([q, k, q - k, q * k], axis=-1) # None, sqe_len, d * 8
        # dense
        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)  # (None, seq_len, 1)
        # 对于每一个instance，中的每一个items 得到对应的得分
        outputs = tf.squeeze(outputs, axis=-1)  # (None, seq_len)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1) # (None, seq_len) 目的是不让无关的item影响 计算 e ** 无穷小 约等于0
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # (None, seq_len)

        # softmax
        outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len) 归一化得分 忽略被mask的部分
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len) # 这个是得分

        return outputs

class Dice2(Layer):
    def __init__(self):
        super(Dice2, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x



class Bilinear(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Bilinear, self).__init__()
        self.linear_act = Dense(units, activation=None, use_bias=True)
        self.linear_noact = Dense(units, activation=None, use_bias=False)

    def call(self, a, b, gate_b=None):
        if gate_b is None:
            return tf.keras.activations.sigmoid(self.linear_act(a) + self.linear_noact(b))
        else:
            return tf.keras.activations.tanh(self.linear_act(a) + tf.math.multiply(gate_b, self.linear_noact(b)))

class AUGRU(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AUGRU, self).__init__()

        self.u_gate = Bilinear(units)
        self.r_gate = Bilinear(units)
        self.c_memo = Bilinear(units)

    def call(self, inputs, state, att_score, hiddens):
        u = self.u_gate(inputs, state)
        r = self.r_gate(inputs, state)
        c = self.c_memo(inputs, state, r)

        u_ = att_score * u
        final = (1 - u_) * state + u_ * c
        hiddens.append(final)
        return final

class AUGRU_modified(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AUGRU_modified, self).__init__()

        self.u_gate = Bilinear(units)
        self.r_gate = Bilinear(units)
        self.c_memo = Bilinear(units)

    def call(self, inputs, state, att_score, hidden_statess, mask, max_len ):
        for i in range(max_len):
            input_ = inputs[i] # 取出 batch 一个时间步骤 (batch, embdim*2)
            mask_ = mask[:, i] # 取出 batch 一个时间步骤 (batch, )
            mask_ = tf.expand_dims(mask_, -1)
            att_score_ = att_score[i]
            u = self.u_gate(input_, state)
            r = self.r_gate(input_, state)
            c = self.c_memo(input_, state, r)
            u_ = att_score_ * u
            temp = (1 - u_) * state + u_ * c
            state = temp * mask_ + state * (1-mask_)
            hidden_statess.append(state)
        return state
class MyGRU(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AUGRU_modified, self).__init__()

        self.u_gate = Bilinear(units)
        self.r_gate = Bilinear(units)
        self.c_memo = Bilinear(units)

    def call(self, inputs, state, att_score, hidden_statess, mask, max_len ):
        for i in range(max_len):
            input_ = inputs[i] # 取出 batch 一个时间步骤 (batch, embdim*2)
            att_score_ = att_score[i]
            u = self.u_gate(input_, state)
            r = self.r_gate(input_, state)
            c = self.c_memo(input_, state, r)
            state = (1 - u) * state + u * c
            hidden_statess.append(state)
        return state




class AuxiliaryNet(tf.keras.layers.Layer):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.bn = BatchNormalization(trainable=True)
        self.dnn1 = tf.keras.layers.Dense(100)
        self.dnn2 = tf.keras.layers.Dense(50)
        self.dnn3 = tf.keras.layers.Dense(2)

    def call(self, inputs):
        x1 = self.bn(inputs)
        x2 = self.dnn1(x1)
        x3 = tf.nn.sigmoid(x2)
        x4 = self.dnn2(x3)
        x5 = tf.nn.sigmoid(x4)
        x6 = self.dnn3(x5)
        y_hat = tf.nn.softmax(x6) + 0.00000001
        return y_hat

