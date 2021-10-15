import os
import time
import tensorflow as tf
from dataloader import get_dataloader
from Models.Sequential import Base
from util import *

# Config
print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())


# Data Load
train_data, test_data, user_count, item_count= get_dataloader(16, 32)
user_count = 400000
# 对于train_data #1. user_id (batch_size, 1)， 2. item_id(batch_size, 1), 3. target(batch_size, 1),
# #4. 每个用户的行为序列（batch_size, max_length）   5. 最大长度 max_length

# Loss, Optim
lr = 0.001
optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0)
loss_metric = tf.keras.metrics.Sum()
auc_metric = tf.keras.metrics.AUC()

# Model
model = Base(user_count, item_count,
             5, 5)

# Board
#train_summary_writer = tf.summary.create_file_writer(args.log_path)


def train_one_step(u,i,y):
    with tf.GradientTape() as tape:
        output = model(u,i)
        loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=output,
                                                        labels=tf.cast(y, dtype=tf.float32)))
    gradient = tape.gradient(loss, model.trainable_variables)
    clip_gradient, _ = tf.clip_by_global_norm(gradient, 5.0)
    optimizer.apply_gradients(zip(clip_gradient, model.trainable_variables))

    loss_metric(loss)
epochs = 3
print_step = 1000
# Train
def train(optimizer):
    best_loss= 0.
    best_auc = 0.
    start_time = time.time()
    for epoch in range(epochs):
        for step, (u, i, y) in enumerate(train_data, start=1):
            train_one_step(u, i, y)

            if print_step == 0:
                test_gauc, auc = eval(model, test_data)
                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                      (epoch, step, loss_metric.result() / print_step, test_gauc, auc))

                if best_auc < test_gauc:
                    best_loss= loss_metric.result() / print_step
                    best_auc = test_gauc
 #                   model.save_weights(model_path+'cp-%d.ckpt'%epoch)
                loss_metric.reset_states()

        loss_metric.reset_states()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

        print('Epoch %d DONE\tCost time: %.2f' % (epoch, time.time()-start_time))
    print('Best test_gauc: ', best_auc)


# Main
if __name__ == '__main__':
    train(optimizer)