"""
Created on May 23, 2020

model: Deep interest network for click-through rate prediction

@author: Ziyao Geng
"""
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout, GRU, LSTM
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from modules import *


class DIN(Model):
    def __init__(self, feature_columns, att_hidden_units,
                 ffn_hidden_units, att_activation='sigmoid', ffn_activation='prelu', maxlen = None, dnn_dropout=0.):
        """
        DIN
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param att_hidden_units: A tuple or list. Attention hidden units.
        :param ffn_hidden_units: A tuple or list. Hidden units list of FFN.
        :param att_activation: A String. The activation of attention.
        :param ffn_activation: A String. Prelu or Dice.
        :param maxlen: A scalar. Maximum sequence length.
        :param dropout: A scalar. The number of Dropout.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(DIN, self).__init__()
        self.maxlen = maxlen

        self.dense_feature_columns, self.sparse_seq_columns, self.sparse_item_side, self.sparse_user_side, = feature_columns

        # len
        self.dense_len = len(self.dense_feature_columns)

        # other embedding layers
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform')
                                 for feat in self.sparse_seq_columns]

        # behavior embedding layers, item id and category id
        self.embed_user_side = [Embedding(input_dim=feat['feat_num'],
                                          input_length=1,
                                          output_dim=feat['embed_dim'],
                                          embeddings_initializer='random_uniform'
                                          )
                                for feat in self.sparse_user_side]
        # behavior embedding layers, item id and category id
        self.embed_item_side = [Embedding(input_dim=feat['feat_num'],
                                          input_length=1,
                                          output_dim=feat['embed_dim'],
                                          embeddings_initializer='random_uniform'
                                          )
                                for feat in self.sparse_item_side]

        # attention layer
        self.attention_layer = Attention_Layer(att_hidden_units, att_activation)

        self.bn = BatchNormalization(trainable=True)
        # ffn
        if ffn_activation != 'prelu':
            print("using dice")
        self.ffn = [Dense(unit,activation=PReLU() if ffn_activation == 'prelu' else Dice())\
             for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(2)

    def call(self, inputs, activation_values):
        # dense_inputs and sparse_inputs is empty
        # seq_inputs (None, maxlen, behavior_num)
        # item_inputs (None, behavior_num)
        dense_inputs, target_user_side, seq_inputs, seq_inputs_neg, target_item_seq, target_item_side = inputs
        # attention ---> mask, if the element of seq_inputs is equal 0, it must be filled in. 
        mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)  # (None, maxlen)
        # other
        other_info = dense_inputs
        user_side = tf.concat([self.embed_user_side[i](target_user_side[:, i]) for i in range(5)], axis=-1)
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(3)],
                              axis=-1)
        target_embed_side = tf.concat([self.embed_item_side[i](target_item_side[:, i]) for i in range(3)],
                                      axis=-1)
        # seq, item embedding and category embedding should concatenate
        # seq_embed : (None, max_length, embed_dim * 2)
        # item_embed: (None, embed_dim * 2)

        target_embed_seq = tf.concat([self.embed_seq_layers[i](target_item_seq[:, i]) for i in range(3)], axis=-1)


        # user_info : (None, embed_dim * 2)
        user_info = self.attention_layer([target_embed_seq, seq_embed, seq_embed, mask, activation_values])
        # concat user_info(att hist), cadidate item embedding, other features

        info_all = tf.concat([user_info, target_embed_seq, target_embed_side, user_side], axis=-1)


        info_all = self.bn(info_all)
        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)

#        info_all = self.dropout(info_all)
        logits = self.dense_final(info_all)
        outputs = tf.nn.softmax(logits)
        return outputs, logits


class DIEN(Model):
    def __init__(self, feature_columns, att_hidden_units=(80, 40),
                 ffn_hidden_units=(80, 40), att_activation='sigmoid', ffn_activation='prelu', maxlen=40, dnn_dropout=0., embed_dim=None):
        """
        DIN
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param att_hidden_units: A tuple or list. Attention hidden units.
        :param ffn_hidden_units: A tuple or list. Hidden units list of FFN.
        :param att_activation: A String. The activation of attention.
        :param ffn_activation: A String. Prelu or Dice.
        :param maxlen: A scalar. Maximum sequence length.
        :param dropout: A scalar. The number of Dropout.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(DIEN, self).__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.aux_net = AuxiliaryNet()

        self.dense_feature_columns, self.sparse_seq_columns, self.sparse_item_side, self.sparse_user_side,  = feature_columns

        # len
        self.dense_len = len(self.dense_feature_columns)

        # seq_embedding_layer
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform')
                                    for feat in self.sparse_seq_columns]

        # behavior embedding layers, item id and category id
        self.embed_user_side = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform'
                                           )
                                 for feat in self.sparse_user_side]
        # behavior embedding layers, item id and category id
        self.embed_item_side = [Embedding(input_dim=feat['feat_num'],
                                          input_length=1,
                                          output_dim=feat['embed_dim'],
                                          embeddings_initializer='random_uniform'
                                          )
                                for feat in self.sparse_item_side]

        # attention layer
        self.attention_layer = Attention_Layer_for_AUGRU(
            att_hidden_units=att_hidden_units,
            activation=att_activation,
            embed_dim=self.embed_dim
        )
        self.hist_gru = GRU(self.embed_dim*3, return_sequences=True)
        self.hist_augru = AUGRU_modified(self.embed_dim*3)
        self.bn = BatchNormalization(trainable=True)
        # ffn
        if ffn_activation != 'prelu':
            print("using dice")
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation == 'prelu' else Dice()) \
                    for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(2)

    def call(self, inputs, hidden_statess, atten_scores):

        # dense_inputs and sparse_inputs is empty
        # seq_inputs (None, maxlen, behavior_num)
        # item_inputs (None, behavior_num)
        dense_inputs, target_user_side, seq_inputs, seq_inputs_neg, target_item_seq, target_item_side = inputs
        # attention ---> mask, if the element of seq_inputs is equal 0, it must be filled in.
        mask_bool = tf.not_equal(seq_inputs[:, :, 0], 0)  # (None, maxlen)
        # 对于所有的 用户序列 None, maxlen, embed_dim 只要值为非0 则返回1 为 0 则返回 0 得到 None, maxlen
        # other
        user_side = tf.concat([self.embed_user_side[i](target_user_side[:, i]) for i in range(5)], axis=-1)
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(3)],
                              axis=-1)
        seq_embed_neg = tf.concat([self.embed_seq_layers[i](seq_inputs_neg[:, :, i]) for i in range(3)],
                              axis=-1)
        target_embed_seq = tf.concat([self.embed_seq_layers[i](target_item_seq[:, i]) for i in range(3)], axis=-1)

        target_embed_side = tf.concat([self.embed_item_side[i](target_item_side[:, i]) for i in range(3)],
                              axis=-1)
        gru_embed = self.hist_gru(seq_embed, mask=mask_bool)

        mask_value = tf.cast(mask_bool, dtype=tf.float32)


        auxiliary_loss = self.compute_auxiliary(gru_embed[:, :-1, :], seq_embed[:, 1:, :],
                               seq_embed_neg[:, 1:, :],
                               mask_value[:, 1:])

        # att_score : None, 1, maxlen
        att_score = self.attention_layer([target_embed_seq, gru_embed, gru_embed, mask_value])
        atten_scores.append(att_score)
#        augru_hidden_state = tf.zeros_like(gru_embed[:, 0, :])
        augru_hidden_state = tf.zeros([gru_embed.shape[0],48])
        augru_hidden_state = self.hist_augru(
            tf.transpose(gru_embed, [1, 0, 2]),
            # gru_embed: (None, maxlen, gru_hidden) -> (maxlen, None, gru_hidden)
            augru_hidden_state,
            tf.transpose(att_score, [2, 0, 1]),  # None, 1, maxlen -> maxlen, None, 1 1
            hidden_statess,
            mask=mask_value,
            max_len=self.maxlen,

        )


        # concat user_info(att hist), cadidate item embedding, other features

        info_all = tf.concat([augru_hidden_state, target_embed_seq, target_embed_side, user_side], axis=-1)

        info_all = self.bn(info_all)

        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)

#        info_all = self.dropout(info_all)
        logits = self.dense_final(info_all)
        outputs = tf.nn.sigmoid(logits)
        return outputs, logits, auxiliary_loss

    def compute_auxiliary(self, h_states, click_seq, noclick_seq, mask):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.aux_net(click_input_)[:, :, 0]
        noclick_prop_ = self.aux_net(noclick_input_)[:, :, 0]
        click_loss_ = - tf.reshape(tf.math.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.math.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_


class BaseModel(Model):
    def __init__(self, feature_columns, att_hidden_units=(80, 40),
                 ffn_hidden_units=(80, 40), att_activation='prelu', ffn_activation='prelu', maxlen=40, dnn_dropout=0., embed_dim=None):
        """
        DIN
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param att_hidden_units: A tuple or list. Attention hidden units.
        :param ffn_hidden_units: A tuple or list. Hidden units list of FFN.
        :param att_activation: A String. The activation of attention.
        :param ffn_activation: A String. Prelu or Dice.
        :param maxlen: A scalar. Maximum sequence length.
        :param dropout: A scalar. The number of Dropout.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(BaseModel, self).__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim

        self.dense_feature_columns, self.sparse_seq_columns, self.sparse_item_side, self.sparse_user_side,  = feature_columns

        # len
        self.dense_len = len(self.dense_feature_columns)

        # seq_embedding_layer
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform')
                                    for feat in self.sparse_seq_columns]

        # behavior embedding layers, item id and category id
        self.embed_user_side = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform'
                                           )
                                 for feat in self.sparse_user_side]
        # behavior embedding layers, item id and category id
        self.embed_item_side = [Embedding(input_dim=feat['feat_num'],
                                          input_length=1,
                                          output_dim=feat['embed_dim'],
                                          embeddings_initializer='random_uniform'
                                          )
                                for feat in self.sparse_item_side]

        # ffn
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation == 'prelu' else Dice()) \
                    for unit in ffn_hidden_units]
#        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(2)

    def call(self, inputs):

        # dense_inputs and sparse_inputs is empty
        # seq_inputs (None, maxlen, behavior_num)
        # item_inputs (None, behavior_num)
        dense_inputs, target_user_side, seq_inputs, seq_inputs_neg, target_item_seq, target_item_side = inputs
        # attention ---> mask, if the element of seq_inputs is equal 0, it must be filled in.
        mask_bool = tf.not_equal(seq_inputs[:, :, 0], 0)  # (None, maxlen)
        mask_value = tf.cast(mask_bool, dtype=tf.float32)
        # 对于所有的 用户序列 None, maxlen, embed_dim 只要值为非0 则返回1 为 0 则返回 0 得到 None, maxlen
        # other


        user_side = tf.concat([self.embed_user_side[i](target_user_side[:, i]) for i in range(5)], axis=-1)
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(3)],
                              axis=-1)
        seq_embed_neg = tf.concat([self.embed_seq_layers[i](seq_inputs_neg[:, :, i]) for i in range(3)],
                              axis=-1)
        target_embed_seq = tf.concat([self.embed_seq_layers[i](target_item_seq[:, i]) for i in range(3)], axis=-1)

        target_embed_side = tf.concat([self.embed_item_side[i](target_item_side[:, i]) for i in range(3)],
                              axis=-1)


        seq_embed_maked = seq_embed * tf.expand_dims(mask_value,  axis=-1)
        seq_embed_sum = tf.reduce_mean(seq_embed_maked, axis=1)
        info_all = tf.concat([seq_embed_sum, target_embed_seq, target_embed_side, user_side], axis=-1)

        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)

#        info_all = self.dropout(info_all)
        logits = self.dense_final(info_all)
        outputs = tf.nn.sigmoid(logits)
        return outputs, logits


class LR(Model):
    def __init__(self, feature_columns, att_hidden_units=(80, 40),
                 ffn_hidden_units=(80, 40), att_activation='prelu', ffn_activation='prelu', maxlen=40, dnn_dropout=0., embed_dim=None):
        """
        DIN
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param att_hidden_units: A tuple or list. Attention hidden units.
        :param ffn_hidden_units: A tuple or list. Hidden units list of FFN.
        :param att_activation: A String. The activation of attention.
        :param ffn_activation: A String. Prelu or Dice.
        :param maxlen: A scalar. Maximum sequence length.
        :param dropout: A scalar. The number of Dropout.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(LR, self).__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim

        self.dense_feature_columns, self.sparse_seq_columns, self.sparse_item_side, self.sparse_user_side,  = feature_columns

        # len
        self.dense_len = len(self.dense_feature_columns)

        # seq_embedding_layer
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform')
                                    for feat in self.sparse_seq_columns]

        # behavior embedding layers, item id and category id
        self.embed_user_side = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform'
                                           )
                                 for feat in self.sparse_user_side]
        # behavior embedding layers, item id and category id
        self.embed_item_side = [Embedding(input_dim=feat['feat_num'],
                                          input_length=1,
                                          output_dim=feat['embed_dim'],
                                          embeddings_initializer='random_uniform'
                                          )
                                for feat in self.sparse_item_side]

        # ffn
        self.dense_final = Dense(2)

    def call(self, inputs):

        # dense_inputs and sparse_inputs is empty
        # seq_inputs (None, maxlen, behavior_num)
        # item_inputs (None, behavior_num)
        dense_inputs, target_user_side, seq_inputs, seq_inputs_neg, target_item_seq, target_item_side = inputs
        # attention ---> mask, if the element of seq_inputs is equal 0, it must be filled in.
        mask_bool = tf.not_equal(seq_inputs[:, :, 0], 0)  # (None, maxlen)
        mask_value = tf.cast(mask_bool, dtype=tf.float32)
        # 对于所有的 用户序列 None, maxlen, embed_dim 只要值为非0 则返回1 为 0 则返回 0 得到 None, maxlen
        # other


        user_side = tf.concat([self.embed_user_side[i](target_user_side[:, i]) for i in range(5)], axis=-1)
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(3)],
                              axis=-1)
        seq_embed_neg = tf.concat([self.embed_seq_layers[i](seq_inputs_neg[:, :, i]) for i in range(3)],
                              axis=-1)
        target_embed_seq = tf.concat([self.embed_seq_layers[i](target_item_seq[:, i]) for i in range(3)], axis=-1)

        target_embed_side = tf.concat([self.embed_item_side[i](target_item_side[:, i]) for i in range(3)],
                              axis=-1)


        seq_embed_maked = seq_embed * tf.expand_dims(mask_value,  axis=-1)
        seq_embed_sum = tf.reduce_sum(seq_embed_maked, axis=1)
        info_all = tf.concat([seq_embed_sum, target_embed_seq, target_embed_side, user_side], axis=-1)

        logits = self.dense_final(info_all)
        outputs = tf.nn.sigmoid(logits)
        return outputs, logits


class MyModel(Model):
    def __init__(self, feature_columns, att_hidden_units=(80, 40),
                 ffn_hidden_units=(80, 40), att_activation='sigmoid', ffn_activation='prelu', maxlen=40, dnn_dropout=0., embed_dim=None):
        """
        DIN
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param att_hidden_units: A tuple or list. Attention hidden units.
        :param ffn_hidden_units: A tuple or list. Hidden units list of FFN.
        :param att_activation: A String. The activation of attention.
        :param ffn_activation: A String. Prelu or Dice.
        :param maxlen: A scalar. Maximum sequence length.
        :param dropout: A scalar. The number of Dropout.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(MyModel, self).__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.aux_net = AuxiliaryNet()

        self.dense_feature_columns, self.sparse_seq_columns, self.sparse_item_side, self.sparse_user_side,  = feature_columns

        # len
        self.dense_len = len(self.dense_feature_columns)

        # seq_embedding_layer
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform')
                                    for feat in self.sparse_seq_columns]

        # behavior embedding layers, item id and category id
        self.embed_user_side = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform'
                                           )
                                 for feat in self.sparse_user_side]
        # behavior embedding layers, item id and category id
        self.embed_item_side = [Embedding(input_dim=feat['feat_num'],
                                          input_length=1,
                                          output_dim=feat['embed_dim'],
                                          embeddings_initializer='random_uniform'
                                          )
                                for feat in self.sparse_item_side]

        # attention layer
        self.attention_layer = MyAttention(att_hidden_units, att_activation)

        self.hist_gru = LSTM(120)
        self.bn = BatchNormalization(trainable=True)
        # ffn
        if ffn_activation != 'prelu':
            print("using dice")
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation == 'prelu' else Dice()) \
                    for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(2)

    def call(self, inputs, atten_scores):

        # dense_inputs and sparse_inputs is empty
        # seq_inputs (None, maxlen, behavior_num)
        # item_inputs (None, behavior_num)
        dense_inputs, target_user_side, seq_inputs, seq_inputs_neg, target_item_seq, target_item_side = inputs
        # attention ---> mask, if the element of seq_inputs is equal 0, it must be filled in.
        mask_bool = tf.not_equal(seq_inputs[:, :, 0], 0)  # (None, maxlen)
        mask_value = tf.cast(mask_bool, dtype=tf.float32)
        # 对于所有的 用户序列 None, maxlen, embed_dim 只要值为非0 则返回1 为 0 则返回 0 得到 None, maxlen
        # other
        user_side = tf.concat([self.embed_user_side[i](target_user_side[:, i]) for i in range(5)], axis=-1)
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(3)],
                              axis=-1)
        seq_embed_neg = tf.concat([self.embed_seq_layers[i](seq_inputs_neg[:, :, i]) for i in range(3)],
                              axis=-1)
        target_embed_seq = tf.concat([self.embed_seq_layers[i](target_item_seq[:, i]) for i in range(3)], axis=-1)

        target_embed_side = tf.concat([self.embed_item_side[i](target_item_side[:, i]) for i in range(3)],
                              axis=-1)


        # user_info : (None, embed_dim * 2)
        activation_val = []
        user_info = self.attention_layer([target_embed_seq, seq_embed, seq_embed, mask_value, activation_val])
        gru_embed = self.hist_gru(user_info, mask=mask_bool)
        # concat user_info(att hist), cadidate item embedding, other features

        info_all = tf.concat([gru_embed, target_embed_seq, target_embed_side, user_side], axis=-1)

        info_all = self.bn(info_all)

        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)

#        info_all = self.dropout(info_all)
        logits = self.dense_final(info_all)
        outputs = tf.nn.sigmoid(logits)
        return outputs, logits
