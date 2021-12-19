from util import *
import pandas as pd
import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import random


def process_data(embed_dim, maxlen, learning_curve=False, i=-1):
    if learning_curve:
        return create_amazon_electronic_dataset(embed_dim=embed_dim, maxlen=maxlen, learning_curve=learning_curve, i=i)
    if os.path.exists("./ddata/process_feature_columns.pkl"):
        feature_columns = load_data("./ddata/process_feature_columns.pkl")
        train_X = load_data("./ddata/process_train_X.pkl")
        train_y = load_data("./ddata/process_train_y.pkl")
        dev_X = load_data("./ddata/process_dev_X.pkl")
        dev_y = load_data("./ddata/process_dev_y.pkl")
        test_X = load_data("./ddata/process_test_X.pkl")
        test_y = load_data("./ddata/process_test_y.pkl")
        print(f"embedding size is {embed_dim}, max sequence length is {maxlen}")
        if feature_columns[1][0]['embed_dim'] is not embed_dim or len(train_X[2][0]) is not maxlen:
            print("You changed the hyperparameter setting, remaking......")
            return create_amazon_electronic_dataset(embed_dim=embed_dim, maxlen=maxlen)

        print("Data is already, start training...")
        return feature_columns, (train_X, train_y), (dev_X, dev_y), (test_X, test_y)
    else:
        print("Don't have data yet...")
        return create_amazon_electronic_dataset(embed_dim=embed_dim, maxlen=maxlen)



def sparseFeature(feat, feat_num, embed_dim):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}






def create_amazon_electronic_dataset(embed_dim, maxlen, learning_curve=False,i=-1):
    """
    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """

    random.seed(16)
    print('==========Data Preprocess Start============')
    print(f"embedding size is {embed_dim}, max sequence length is {maxlen}")
    if learning_curve:
        df_final = load_data("./ddata/start_data/df_final.pkl")

        df_test_dev = df_final[0].sample(frac=1)
        df_test = df_test_dev[:len(df_test_dev) // 2]
        df_dev = df_test_dev[len(df_test_dev) // 2:]
        df_train = df_final[1:i]
        df_train = pd.concat(df_train, axis=0, ignore_index=True)

    else:
        df_final = load_data("./ddata/start_data/df_final.pkl")

        df_test_dev = df_final[0].sample(frac=1)
        df_test = df_test_dev[:len(df_test_dev) // 2]
        df_dev = df_test_dev[len(df_test_dev) // 2:]
        df_train = df_final[1:]
        df_train = pd.concat(df_train, axis=0, ignore_index=True)

    num_users = load_data("./ddata/start_data/num_users.pkl")
    num_items = load_data("./ddata/start_data/num_items.pkl")
    num_cats = load_data("./ddata/start_data/num_cats.pkl")
    num_sex = load_data("./ddata/start_data/num_sex.pkl")
    num_ulevel = load_data("./ddata/start_data/num_ulevel.pkl")
    num_atype = load_data("./ddata/start_data/num_atype.pkl")
    num_city = load_data("./ddata/start_data/num_city.pkl")

    num_province = load_data("./ddata/start_data/num_province.pkl")
    num_county = load_data("./ddata/start_data/num_county.pkl")
    num_brand_code = load_data("./ddata/start_data/num_brand_code.pkl")
    num_shope = load_data("./ddata/start_data/num_shope.pkl")
    num_vender = load_data("./ddata/start_data/num_vender.pkl")

    cat_list = load_data("./ddata/start_data/cat_list.pkl")  # remember to add one when use

    train_df = df_train.loc[:,
               ["item_sku_id", "item_third_cate_cd", 'brand_code', 'shop_id', 'vender_id',
                "item_seq", "cate_seq", "type_seq",
                "city", "sex", "user_level", 'province', 'county',
                "label", "shop_score"]]
    dev_df = df_dev.loc[:,
               ["item_sku_id", "item_third_cate_cd", 'brand_code', 'shop_id', 'vender_id',
                "item_seq", "cate_seq", "type_seq",
                "city", "sex", "user_level", 'province', 'county',
                "label", "shop_score"]]
    test_df = df_test.loc[:,
               ["item_sku_id", "item_third_cate_cd", 'brand_code', 'shop_id', 'vender_id',
                "item_seq", "cate_seq", "type_seq",
                "city", "sex", "user_level", 'province', 'county',
                "label", "shop_score"]]


    train_set = []
    dev_set = []
    test_set = []
    max_lenth = maxlen

    for index, row in train_df.iterrows():
        train_set.append(
            (row["item_seq"][-max_lenth:], row["cate_seq"][-max_lenth:], row["type_seq"][-max_lenth:],
             row["item_sku_id"], row["item_third_cate_cd"], row["brand_code"], row["shop_id"], row["vender_id"],
             row["city"], row["sex"], row["user_level"], row["province"], row["county"],
             row["label"], row["shop_score"]
             ))
    for index, row in dev_df.iterrows():
        dev_set.append(
            (row["item_seq"][-max_lenth:], row["cate_seq"][-max_lenth:], row["type_seq"][-max_lenth:],
             row["item_sku_id"], row["item_third_cate_cd"], row["brand_code"], row["shop_id"], row["vender_id"],
             row["city"], row["sex"], row["user_level"], row["province"], row["county"],
             row["label"], row["shop_score"]
             ))
    for index, row in test_df.iterrows():
        test_set.append(
            (row["item_seq"][-max_lenth:], row["cate_seq"][-max_lenth:], row["type_seq"][-max_lenth:],
             row["item_sku_id"], row["item_third_cate_cd"], row["brand_code"], row["shop_id"], row["vender_id"],
             row["city"], row["sex"], row["user_level"], row["province"], row["county"],
             row["label"], row["shop_score"]
             ))


    # feature columns
    feature_columns = [[],
                       [sparseFeature('item_sku_id', num_items, embed_dim),
                        sparseFeature('cate_id', num_cats, embed_dim),
                        sparseFeature('action_type', num_atype, embed_dim)],

                        [sparseFeature('brand_code', num_brand_code, embed_dim),
                        sparseFeature('shop_id', num_shope, embed_dim),
                        sparseFeature('vender_id', num_vender, embed_dim)],

                        [sparseFeature('city', num_city, embed_dim),
                        sparseFeature('sex', num_sex, embed_dim),
                        sparseFeature('user_level', num_ulevel, embed_dim),
                        sparseFeature('province', num_province, embed_dim),
                        sparseFeature('county', num_county, embed_dim)
                        ]]
    # behavior
#    behavior_list = ['item_sku_id', 'cate_id', 'action_type']
###########################################
    item_pool = set()
    for i in range(len(train_set)):
        for item in train_set[i][0]:
            item_pool.add(item)
    for i in range(len(dev_set)):
        for item in dev_set[i][0]:
            item_pool.add(item)
    for i in range(len(test_set)):
        for item in test_set[i][0]:
            item_pool.add(item)
    item_pool = list(item_pool)
    neg_sample_train = []
    for i in range(len(train_set)):
        iid_temp = []
        cat_temp = []
        ac_temp = []
        for item in train_set[i][0]:
            while True:
                neg_index = random.randint(0, len(item_pool) - 1)
                neg_sample = item_pool[neg_index]
                if neg_sample in train_set[i][0]:
                    continue
                iid_temp.append(neg_sample)
                cat = cat_list[neg_sample - 1]
                cat_temp.append(cat)
                neg_action = random.randint(1, 4)
                ac_temp.append(neg_action)
                break
        neg_sample_train.append([iid_temp, cat_temp, ac_temp])
    neg_sample_dev = []
    for i in range(len(dev_set)):
        iid_temp = []
        cat_temp = []
        ac_temp = []
        for item in dev_set[i][0]:
            while True:
                neg_index = random.randint(0, len(item_pool) - 1)
                neg_sample = item_pool[neg_index]
                if neg_sample in dev_set[i][0]:
                    continue
                iid_temp.append(neg_sample)
                cat = cat_list[neg_sample - 1]
                cat_temp.append(cat)
                neg_action = random.randint(1, 4)
                ac_temp.append(neg_action)
                break
        neg_sample_dev.append([iid_temp, cat_temp, ac_temp])

    neg_sample_test = []
    for i in range(len(test_set)):
        iid_temp = []
        cat_temp = []
        ac_temp = []
        for item in test_set[i][0]:
            while True:
                neg_index = random.randint(0, len(item_pool) - 1)
                neg_sample = item_pool[neg_index]
                if neg_sample in test_set[i][0]:
                    continue
                iid_temp.append(neg_sample)
                cat = cat_list[neg_sample - 1]
                cat_temp.append(cat)
                neg_action = random.randint(1, 4)
                ac_temp.append(neg_action)
                break
        neg_sample_test.append([iid_temp, cat_temp, ac_temp])
###########################################


    train_data = []
    for i in range(len(train_set)):
        sub_hist = []
        neg_hist = []
        for j in range(len(train_set[i][0])):
            sub_hist.append([train_set[i][0][j], train_set[i][1][j], train_set[i][2][j]]) # hist
            neg_hist.append([neg_sample_train[i][0][j], neg_sample_train[i][1][j], neg_sample_train[i][2][j]])
        train_data.append([sub_hist,
                           neg_hist,
                           [train_set[i][3], train_set[i][4], 2], #target_item_seq
                          [train_set[i][5], train_set[i][6], train_set[i][7]], # target_item_side
                           [train_set[i][8], train_set[i][9], train_set[i][10], train_set[i][11], train_set[i][12]], # target_user_side
                           [train_set[i][14]],
                           train_set[i][13]]) # label
    dev_data = []
    for i in range(len(dev_set)):
        sub_hist = []
        neg_hist = []
        for j in range(len(dev_set[i][0])):
            sub_hist.append([dev_set[i][0][j], dev_set[i][1][j], dev_set[i][2][j]])  # hist
            neg_hist.append([neg_sample_dev[i][0][j], neg_sample_dev[i][1][j], neg_sample_dev[i][2][j]])
        dev_data.append([sub_hist,
                           neg_hist,
                           [dev_set[i][3], dev_set[i][4], 2],  # target_item_seq
                           [dev_set[i][5], dev_set[i][6], dev_set[i][7]],  # target_item_side
                           [dev_set[i][8], dev_set[i][9], dev_set[i][10], dev_set[i][11], dev_set[i][12]],
                           # target_user_side
                           [dev_set[i][14]],
                           dev_set[i][13]])  # label

    test_data = []
    for i in range(len(test_set)):
        sub_hist = []
        neg_hist = []
        for j in range(len(test_set[i][0])):
            sub_hist.append([test_set[i][0][j], test_set[i][1][j], test_set[i][2][j]])  # hist
            neg_hist.append([neg_sample_test[i][0][j], neg_sample_test[i][1][j], neg_sample_test[i][2][j]])
        test_data.append([sub_hist,
                           neg_hist,
                           [test_set[i][3], test_set[i][4], 2],  # target_item_seq
                           [test_set[i][5], test_set[i][6], test_set[i][7]],  # target_item_side
                           [test_set[i][8], test_set[i][9], test_set[i][10], test_set[i][11], test_set[i][12]],# target_user_side
                           # target_user_side
                            [test_set[i][14]], # dense
                           test_set[i][13]])  # label

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    # create dataframe
    train = pd.DataFrame(train_data, columns=['hist', 'neg_hist', 'target_item_seq', 'target_item_side', 'target_user_side', 'dense', 'label'])
    dev = pd.DataFrame(dev_data,
                         columns=['hist', 'neg_hist', 'target_item_seq', 'target_item_side', 'target_user_side',
                                  'dense', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'neg_hist', 'target_item_seq', 'target_item_side', 'target_user_side', 'dense', 'label'])

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    train_X = [np.array(train['dense'].tolist()), np.array(train['target_user_side'].tolist()),
               pad_sequences(train['hist'], maxlen=maxlen),
               pad_sequences(train['neg_hist'], maxlen=maxlen),
               np.array(train['target_item_seq'].tolist()),
               np.array(train['target_item_side'].tolist())]

    train_y = []
    noise = 0.1
    for lb in train['label']:
        if lb == 1:
            train_y.append([1 - noise, 0 + noise])
        else:
            train_y.append([0 + noise, 1-noise])
    train_y = np.array(train_y)

    dev_X = [np.array(dev['dense'].tolist()), np.array(dev['target_user_side'].tolist()),
               pad_sequences(dev['hist'], maxlen=maxlen),
               pad_sequences(dev['neg_hist'], maxlen=maxlen),
               np.array(dev['target_item_seq'].tolist()),
               np.array(dev['target_item_side'].tolist())]
    dev_y = []
    for lb in dev['label']:
        if lb == 1:
            dev_y.append([1 , 0 ])
        else:
            dev_y.append([0, 1])
    dev_y = np.array(dev_y)

    test_X = [np.array(test['dense'].tolist()), np.array(test['target_user_side'].tolist()),
               pad_sequences(test['hist'], maxlen=maxlen),
               pad_sequences(test['neg_hist'], maxlen=maxlen),
               np.array(test['target_item_seq'].tolist()),
               np.array(test['target_item_side'].tolist())]

    test_y = []
    for lb in test['label']:
        if lb == 1:
            test_y.append([1, 0])
        else:
            test_y.append([0, 1])
    test_y = np.array(test_y)

    print('============Data Preprocess End=============')
    store_data(feature_columns, "./ddata/process_feature_columns.pkl")
    store_data(train_X, "./ddata/process_train_X.pkl")
    store_data(train_y, "./ddata/process_train_y.pkl")
    store_data(dev_X, "./ddata/process_dev_X.pkl")
    store_data(dev_y, "./ddata/process_dev_y.pkl")
    store_data(test_X, "./ddata/process_test_X.pkl")
    store_data(test_y, "./ddata/process_test_y.pkl")
    return feature_columns, (train_X, train_y), (dev_X, dev_y), (test_X, test_y)



