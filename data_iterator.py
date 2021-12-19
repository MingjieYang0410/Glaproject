import tensorflow as tf
import numpy as np


class DataLoader:
    def __init__(self, batch_size, data, label):
        self.batch_size = batch_size
        self.data = data
#        label = tf.expand_dims(label, -1)
        self.label = label
        self.data_size = self.data[0]
        self.epoch_size = len(self.data_size) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data[0]):
            self.epoch_size += 1
        self.i = 0
        self.indicates = list(range(len(data[0])))
        np.random.shuffle(self.indicates)
        print("start !!!!!!")

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        '''
        train_X = [np.array([0.] * len(train)), np.array(train['target_user_side'].tolist()),
               pad_sequences(train['hist'], maxlen=maxlen),
               pad_sequences(train['neg_hist'], maxlen=maxlen),
               np.array(train['target_item_seq'].tolist()),
               np.array(train['target_item_side'].tolist())]
        '''
        temp = self.indicates[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                      len(self.data_size))]
        f1 = self.data[0][temp]

        f2 = self.data[1][temp]
        f3 = self.data[2][temp]
        f4 = self.data[3][temp]
        f5 = self.data[4][temp]
        f6 = self.data[5][temp]

        label_ = self.label[temp]

        self.i += 1


        final = [f1, f2, f3, f4, f5, f6]

        return final, tf.cast(label_, dtype=tf.float32)

def get_dataloader(train_batch_size, train_set, train_label):
    return DataLoader(train_batch_size, train_set, train_label)