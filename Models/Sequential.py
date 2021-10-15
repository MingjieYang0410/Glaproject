import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn


class Base(tf.keras.Model):
    def __init__(self, user_count, item_count, user_dim, item_dim):
        super(Base, self).__init__()
        self.item_dim = item_dim
        self.user_dim = user_dim
        self.user_emb = nn.Embedding(user_count, user_dim)
        self.item_emb = nn.Embedding(item_count, item_dim)
        self.fc = tf.keras.Sequential()
        self.fc.add(nn.BatchNormalization())
        self.fc.add(nn.Dense(50, activation='relu'))
        self.fc.add(nn.Dense(50, activation='relu'))
        self.fc.add(nn.Dense(1, activation=None))

    def get_emb(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)

        return user_emb, item_emb

    def call(self, user, item):
        user_emb, item_emb = self.get_emb(user, item)
        user_item_joint = tf.concat([item_emb, user_emb], -1)

        output = self.fc(user_item_joint)
        logit = tf.keras.activations.sigmoid(output)

        return logit