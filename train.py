import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
import numpy as np
from sklearn import metrics
from model import DIN, DIEN, BaseModel,  LR, MyModel
from prepare_data import *
from data_iterator import *
import os
import tensorflow as tf
import datetime
def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first
setup_seed(16)
# ========================= File Paths =======================
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/test/' + current_time + '/train'
summary_writer = tf.summary.create_file_writer(train_log_dir)
mode_save_path = "./saved_model"
# ========================= Training Setting =======================
func_span = 100   # How many iteration to call functions
global_best_auc = -1  # Keep to -1
global_patients = 0  # Keep to 0
epochs = 100
mode = 'normal' # 'normal', 'crossvalidation'
# ========================= Hyper Parameters =======================
maxlen = 100  #100
embed_dim = 16 # to make life easier all sparse features have same embedding dim 16
att_hidden_units = [160, 80, 40]  # FFN for Attention Layer
ffn_hidden_units = [256, 128, 64] # FFN for final output
dnn_dropout = 0.5 # Need to ensure this
att_activation = 'sigmoid'
ffn_activation = 'prelu' #prelu
train_batch_size = 128  # 128
test_val_batch_size = 2048
learning_rate = 0.001
l2_weight = 0
weight_auxiliary = 0
# ========================== Create dataset =======================
if mode == 'learning_curve':
    feature_columns, _, _, _ = process_data(embed_dim, maxlen)
if mode != 'learning_curve':
    feature_columns, train, dev, test = process_data(embed_dim, maxlen)
    train_X, train_y = train
    dev_X, dev_y = dev
    test_X, test_y = test
    dev_data_all = get_dataloader(test_val_batch_size, dev_X, dev_y)
    test_data_all = get_dataloader(test_val_batch_size, test_X, test_y)
    train_data_alll2 = get_dataloader(test_val_batch_size, train_X, train_y)
    num_instance = len(train_X[0])
# ==================================================================
# ========================== Evaluation Recorders =======================
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_acc = tf.keras.metrics.CategoricalAccuracy("train_acc", dtype=tf.float32)
dev_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
dev_acc = tf.keras.metrics.CategoricalAccuracy("test_acc", dtype=tf.float32)

train_loss_base = tf.keras.metrics.Mean('train_loss_base', dtype=tf.float32)
train_loss_alluxiry = tf.keras.metrics.Mean('train_loss_auxiliry', dtype=tf.float32)
val_loss_base = tf.keras.metrics.Mean('vall_loss_base', dtype=tf.float32)
val_loss_alluxiry = tf.keras.metrics.Mean('vall_loss_auxiliry', dtype=tf.float32)
# ==================================================================
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
if mode == 'normal':
#    model = DIN(feature_columns, att_hidden_units, ffn_hidden_units, att_activation,ffn_activation, maxlen, dnn_dropout)
#    model = DIEN(feature_columns, att_hidden_units, ffn_hidden_units, att_activation,ffn_activation, maxlen, dnn_dropout,  embed_dim=embed_dim)
#    model = BaseModel(feature_columns, att_hidden_units, ffn_hidden_units, att_activation,ffn_activation, maxlen, dnn_dropout,  embed_dim=embed_dim)
#    model = LR(feature_columns, att_hidden_units, ffn_hidden_units, att_activation,ffn_activation, maxlen, dnn_dropout,  embed_dim=embed_dim)
    model = MyModel(feature_columns, att_hidden_units, ffn_hidden_units, att_activation,ffn_activation, maxlen, dnn_dropout,  embed_dim=embed_dim)
    model_name = "MyModel"

def evaluate(dev_data_all):
    global global_best_auc
    outputs = []
    labels = []
    for step, (mini_batch, label) in enumerate(dev_data_all, start=1):
        if model_name == 'DIEN':
            hidden = []
            att = []
            output, logits, auxiliary_loss = model(mini_batch, hidden, att)
        elif model_name == 'DIN' or "MyModel":
            activation_value = []
            output, logits = model(mini_batch, activation_value)
        else:
            output, logits = model(mini_batch)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(label, dtype=tf.float32)))
#111111111111111111
#        val_loss_base(loss)
 #       val_loss_alluxiry(auxiliary_loss)
        if model_name == 'DIEN':
            loss += auxiliary_loss * weight_auxiliary
        loss_regularization = []
        for p in model.trainable_variables:
            loss_regularization.append(tf.nn.l2_loss(p))
        loss_regularization = tf.reduce_sum(loss_regularization)
        loss = loss + l2_weight * loss_regularization

        dev_loss(loss)
        dev_acc(label, output)
        outputs.append(output)
        labels.append(label)
    pred = tf.concat(outputs, 0)[:, 0]
    y = tf.concat(labels, 0)[:, 0]
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    if auc > global_best_auc:
        print(f"Validation AUC improved from {global_best_auc} to {auc}")
    else:
        print("No improvement...")
    return auc

def evaluate_train(train_data_all):
    global global_best_auc
    for step, (mini_batch, label) in enumerate(train_data_all, start=1):
        hidden = []
        att = []
        output, logits, auxiliary_loss = model(mini_batch, hidden, att)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(label, dtype=tf.float32)))
        train_loss_base(loss)
        train_loss_alluxiry(auxiliary_loss)




def early_stopping(auc, patients):
    global global_best_auc
    global global_patients
    if auc < global_best_auc:
        global_patients += 1
        if global_patients == patients:
            return True
        print(f"Wait for {global_patients - patients}, current best auc is {global_best_auc}")
    else:
        global_patients = 0
        return False


def save_model(auc):
    global global_best_auc
    if auc > global_best_auc:
        print(f"A better model have been saved with AUC:{auc}")
        model_cur_path = f"./saved_weights/{model_name}_{auc}.ckpt"
        model.save_weights(model_cur_path)
        global_best_auc = max(auc, global_best_auc)
        return
    else:
        print(f"This model is not better than before!---auc:{auc}")


def train_one_step(train_data, label):
    with tf.GradientTape() as tape:
        if model_name == 'DIEN':
            hidden = []
            att = []
            output, logits, auxiliary_loss = model(train_data, hidden, att)
        elif model_name == 'DIN' or "MyModel":
            activation_value = []
            output, logits = model(train_data, activation_value)
        else:
            output, logits = model(train_data)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(label, dtype=tf.float32)))
        if model_name == 'DIEN':
            loss += auxiliary_loss * weight_auxiliary
        loss_regularization = []
        for p in model.trainable_variables:
            loss_regularization.append(tf.nn.l2_loss(p))
        loss_regularization = tf.reduce_sum(loss_regularization)
        loss = loss + l2_weight * loss_regularization
        train_loss(loss)
        train_acc(label, output)
    gradient = tape.gradient(loss, model.trainable_variables)
    clip_gradient, _ = tf.clip_by_global_norm(gradient, 5.0)
    optimizer.apply_gradients(zip(clip_gradient, model.trainable_variables))
    if model_name == "DIEN":
        return loss, auxiliary_loss
    return loss, None


def reset_states(train_loss, train_acc, test_loss, test_acc):
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    test_acc.reset_states()

def reset_global():
    global global_best_auc
    global global_patients
    global_best_auc = -1  # Keep to -1
    global_patients = 0

def main_train(x, y, dev_all):
    reset_global()
    global learning_rate
    global_step = 0
    early_stopping_flag = False
    patients = 1000
    train_loss_list = []
    clocks = []
    if model_name == "DIEN":
        auxiliary_loss_list = []

    for epoch in range(1, epochs + 1):
        print("New epoch start, reshuffling...")
        train_data_all = get_dataloader(train_batch_size, x, y)
        t0 = time.clock()
        base_losses_train = []
        auxiliary_losses_train = []
        base_losses_val = []
        auxiliary_losses_val = []
        for step, (mini_batch, label) in enumerate(train_data_all, start=1):
            if step % 10 == 0:
                print(f"{step} / {len(x[0]) // train_batch_size} epoch {epoch}")
            train_loss_itr, auxiliary_loss = train_one_step(mini_batch, label)
            train_loss_list.append(train_loss_itr)
            if auxiliary_loss:
                auxiliary_loss_list.append(auxiliary_loss)
            global_step += 1
            if global_step % func_span == 0:
                auc = evaluate(dev_all)
#                evaluate_train(train_data_alll2)
                print("============================================")
                print(f"Train_loss is {train_loss.result().numpy()}, acc is {train_acc.result().numpy()}")
                print(f"test_loss is {dev_loss.result().numpy()}, acc is {dev_acc.result().numpy()}")
                print("============================================")
 #               base_losses_train.append(train_loss_base.result().numpy())
  #              auxiliary_losses_train.append(train_loss_alluxiry.result().numpy())
   #             base_losses_val.append(val_loss_base.result().numpy())
   #             auxiliary_losses_val.append(val_loss_alluxiry.result().numpy())
  #              reset_states(val_loss_alluxiry, val_loss_base, train_loss_base, train_loss_alluxiry)
                reset_states(train_loss, train_acc, dev_loss, dev_acc)
                if early_stopping(auc, patients):
                    print(
                        f"The AUC for validation set have not been improved for {patients * func_span} iterations, stopping training...")
                    early_stopping_flag = True
                    break
                save_model(auc)
                print("============================================")
            store_data(base_losses_train, "./results/base_train")
            store_data(auxiliary_losses_train, "./results/auxiliary_train")
            store_data(auxiliary_losses_val, "./results/auxiliary_val")
            store_data(base_losses_val, "./results/base_val")
        if early_stopping_flag:
            time_taken = 0
            for t in clocks:
                time_taken += t
            time_taken /= len(clocks)
            print(f"Stop training... The best AUC is {global_best_auc}")
            print(f"Average time per epoch is {time_taken}")
            break
        clocks.append(time.clock() - t0)
        learning_rate *= 0.5

def cross_validation(num_fold=5):

    return global_best_auc_list



if mode == 'normal':
    print("start_normal")
    main_train(x=train_X, y=train_y, dev_all=dev_data_all)
if mode == 'crossvalidation':
    print("start_cross_validation")
    feature_columns, train, dev, test = process_data(embed_dim, maxlen)
    train_X, train_y = train
    dev_X, dev_y = dev
    #    test_X, test_y = test
    train_piece_size = len(train_X[0]) // 5
    val_piece_size = len(dev_X[0]) // 5
    train_piece_X = {}
    train_piece_y = {}
    for fold in range(5):
        temp_x = []
        for part in train_X:
            temp_x.append(part[fold * train_piece_size: (fold + 1) * train_piece_size])
        train_piece_X[fold] = list(temp_x)
        train_piece_y[fold] = np.array(list(train_y[fold * train_piece_size: (fold + 1) * train_piece_size]))
    # dict{ fordid : data }
    dev_piece_X = {}
    dev_piece_y = {}
    for fold in range(5):
        temp_x = []
        for part in dev_X:
            temp_x.append(part[fold * val_piece_size: (fold + 1) * val_piece_size])
        dev_piece_X[fold] = list(temp_x)
        dev_piece_y[fold] = np.array(list(dev_y[fold * val_piece_size: (fold + 1) * val_piece_size]))

    fords_x = {}
    fords_y = {}
    # dict { testid : data }
    for index, (one, two, three, four) in enumerate(
            [(0, 1, 2, 3), (1, 2, 3, 4), (0, 2, 3, 4), (0, 1, 3, 4), (0, 1, 2, 4)]):
        temp_x = []
        for i in range(len(train_piece_X[0])):
            piece_arr = [train_piece_X[one][i], train_piece_X[two][i], train_piece_X[three][i], train_piece_X[four][i]]
            temp_x.append(np.concatenate(piece_arr, axis=0))
        temp_y = np.concatenate([train_piece_y[one], train_piece_y[two], train_piece_y[three], train_piece_y[four]],
                                axis=0)
        fords_x[index] = list(temp_x)
        fords_y[index] = np.array(list(temp_y))
    global_best_auc_list = []
    for i in range(5):
        dev_data_all = get_dataloader(test_val_batch_size, dev_piece_X[i], dev_piece_y[i])
        model = BaseModel(feature_columns, att_hidden_units, ffn_hidden_units, att_activation, ffn_activation, maxlen,
                    dnn_dropout)
        model_name = "BaseModel"
        print(f"using {model_name}")
        main_train(x=fords_x[i], y=fords_y[i], dev_all=dev_data_all)
        global_best_auc_list.append(global_best_auc)
    store_data(global_best_auc_list, "./results/glob_list")
if mode == 'learning_curve':
    print("starting learning curve")
    res = {}
    for i in [3]:
        feature_columns, train, dev, test = process_data(embed_dim, maxlen, learning_curve=True, i=i)
        train_X, train_y = train
        dev_X, dev_y = dev
        test_X, test_y = test
        dev_data_all = get_dataloader(test_val_batch_size, dev_X, dev_y)
        test_data_all = get_dataloader(test_val_batch_size, test_X, test_y)
        num_instance = len(train_X[0])

        model = DIN(feature_columns, att_hidden_units, ffn_hidden_units, att_activation,ffn_activation, maxlen, dnn_dropout)
        model_name = "DIN"
        main_train(x=train_X, y=train_y, dev_all=dev_data_all)

        res[i] = (num_instance, global_best_auc)
    store_data(res, "./results/learning_curve")



