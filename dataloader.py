from util import *
import numpy as np
from Models.Sequential import *
import tensorflow as tf

root_voc = "./ddata/vocs/"
item_cnt = len(load_data(root_voc + "user_voc.pkl"))
user_cnt = len(load_data(root_voc + "iid_voc.pkl"))

root_data = "./ddata/start_data/"
df_train = load_data(root_data + "df_train.pkl")
df_val = load_data(root_data + "df_val.pkl")

df_train.iloc[:, :-1].astype('float')
df_val.iloc[:, :-1].astype('float')

train_array = np.array(df_train.loc[:, ["user_log_acct", "item_sku_id"]])
train_label = np.array(df_train.loc[:, "label"]).reshape(len(train_array), 1)
test_array = np.array(df_val.loc[:, ["user_log_acct", "item_sku_id"]])
test_label = np.array(df_val.loc[:, "label"]).reshape(len(test_array), 1)
print(user_cnt)
#378471
# = tf.data.Dataset.from_tensor_slices((train_array, train_label))
#test_set = tf.data.Dataset.from_tensor_slices((test_array, test_label))

class DataLoader:
    def __init__(self, batch_size, train, label):
        self.batch_size = batch_size
        self.data = train
        self.label = label
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                      len(self.data))]
        lab = self.label[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.label))]
        self.i += 1

        u, i , y= [], [], []
        for t in ts:
            u.append(t[0]) # user_id
            i.append(t[1]) # item_id
        for l in lab:
            y.append(l)
        k = 0

        return tf.convert_to_tensor(u), tf.convert_to_tensor(i), tf.convert_to_tensor(y)

def get_dataloader(train_batch_size, test_batch_size):

    return DataLoader(train_batch_size, train_array, train_label), DataLoader(test_batch_size, test_array, test_label), \
           user_cnt, item_cnt
