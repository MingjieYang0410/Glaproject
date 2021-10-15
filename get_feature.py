from util import *
import numpy as np
import pandas as pd


def _label_trans(x, dic_):
    try:
        return dic_[x]
    except:
        return 0


def sliding_window(df, label_st=(4, 11), label_en=(4, 15), candidate_st=(4, 6), candidate_en=(4, 10), fea_en=(4, 10)):
    """
    1. For all the records time start from condidate_st to candidate_en will be take out
    for every user_item interactions in this period, we remove duplicate and use the unique pair as we dataset's unique index
    shape like userid_itemid  userid  item id is the basic form, again, userid_itemid is all the interactions between
    candidate period.

    2. The unique indexes will then be label according to weather users buy this item or not from label_st to label_en

    3. The user side info and item side info from the beginning to fea_en are stack to the unique indexed record as features

    It is not efficiently designed on purpose for flexibility

    :param df: input_df
    :param label_st: starting point to get label
    :param label_en: ending point to get label
    :param candidate_st: starting point to get dataset
    :param candidate_en: ending point to get dataset
    :param fea_en: ending point to get feature
    :return:
    df_candidates: dataset basic form with user_item interaction as unique id
    data_fea: feature range for the dataset, will be stacked to the basic form of dataset
    """

    lb_st = df.loc[(df['month'] == label_st[0]) & (df['day'] == label_st[1]), 'month_day'].values[0]
    lb_en = df.loc[(df['month'] == label_en[0]) & (df['day'] == label_en[1]), 'month_day'].values[0]

    cand_st = df.loc[(df['month'] == candidate_st[0]) & (df['day'] == candidate_st[1]), 'month_day'].values[0]
    cand_en = df.loc[(df['month'] == candidate_en[0]) & (df['day'] == candidate_en[1]), 'month_day'].values[0]
    fea_position = df.loc[(df['month'] == fea_en[0]) & (df['day'] == fea_en[1]), 'month_day'].values[0]

    ind_label = (df['month_day'] >= lb_st) & (df['month_day'] <= lb_en) & (df['action_type'] == 2)
    ind_candidate = (df['month_day'] >= cand_st) & (df['month_day'] <= cand_en)
    ind_fea = (df['month_day'] <= fea_position)

    data_label = df.loc[ind_label].copy()
    data_fea = df.loc[ind_fea].copy()  # 用来构建特征集合
    data_candidates = df.loc[ind_candidate].copy()

    # 构建候选集
    df_candidates = data_candidates[['user_log_acct', 'item_sku_id', 'month_day']].copy()
    df_candidates = df_candidates.drop_duplicates(subset=['user_log_acct', 'item_sku_id'])
    df_candidates = df_candidates.loc[(df_candidates.item_sku_id.isnull() == False)]

    # 构建标签
    label = data_label[['user_log_acct', 'item_sku_id', 'day']].copy()

    # 打标签
    df_candidates['label_cnt'] = 0
    df_candidates['label_days'] = 0
    df_candidates['user_item'] = df_candidates['user_log_acct'].astype(str) + '_' + df_candidates['item_sku_id'].astype(
        str)
    label['user_item'] = label['user_log_acct'].astype(str) + '_' + label['item_sku_id'].astype(str)
    dic_cnt = label['user_item'].value_counts().to_dict()
    dic_days = label.groupby('user_item')['day'].nunique().to_dict()
    df_candidates['label_cnt'] = df_candidates['user_item'].apply(lambda x: _label_trans(x, dic_cnt)).values
    df_candidates['label_days'] = df_candidates['user_item'].apply(lambda x: _label_trans(x, dic_days)).values

    return df_candidates, data_fea  # basic_form, feature_range


def get_feature(df_original, df_basic_train, df_basic_val, how="basic"):
    """
    Input must by the most basic form return by sliding_window
    :param df_basic_train: Basic form for training set
    :param df_basic_val: Bsic for for validation set
    :param how: How many feature required?
    :return: merged basic form with other required information
    """
    jd_user = df_original[['user_log_acct', 'age', 'sex', 'user_level', 'province', 'city', 'county']].drop_duplicates(
        ['user_log_acct'], keep='first')
    jd_item = df_original[
        ['item_sku_id', 'brand_code', 'shop_id', 'item_third_cate_cd', 'vender_id', 'shop_score']].drop_duplicates(
        ['item_sku_id'], keep='first')

    u_fea_cols = [col for col in jd_user.columns if col not in ['user_log_acct']]
    i_fea_cols = [col for col in jd_item.columns if col not in ['item_sku_id']]

    train_cols = ['user_log_acct', 'item_sku_id'] + u_fea_cols + i_fea_cols

    df_train = df_basic_train.merge(jd_user, on='user_log_acct', how='left')  # Merge user features to basic form
    df_train = df_train.merge(jd_item, on='item_sku_id', how='left')  # Merger item features to basic form

    neg_df_train = df_train[df_train.label_cnt == 0].reset_index(drop=True)
    pos_df_train = df_train[df_train.label_cnt != 0].reset_index(drop=True)

    neg_df_train = neg_df_train.sample(n=200000)  # Negative Sampling
    df_train = pd.concat([neg_df_train, pos_df_train], axis=0, ignore_index=True)

    df_train['label'] = df_train['label_cnt'] > 0
    df_train['label'] = df_train['label'].astype(int)

    # 验证集
    df_val = df_basic_val.merge(jd_user, on='user_log_acct', how='left')
    df_val = df_val.merge(jd_item, on='item_sku_id', how='left')

    df_val['label'] = df_val['label_cnt'] > 0
    df_val['label'] = df_val['label'].astype(int)

    return df_train, df_val, train_cols


def get_history(df_train, df_val, train_fea_range, val_fea_range):
    """
    This function is used to get the historical interaction of user
    :param df_train: output of get_feature
    :param df_val: output of get_feature
    :param train_fea_range:output of sliding_window
    :param val_fea_range: output of sliding_window
    :return: merged DataFrame from get_feature with additional users' historical info
    """
    df_train.sort_values(['user_log_acct', 'month_day'], inplace=True)
    df_val.sort_values(['user_log_acct', 'month_day'], inplace=True)

    valid_item_seq = val_fea_range.groupby(['user_log_acct'])['item_sku_id'].agg(list).reset_index()
    valid_item_seq.columns = ['user_log_acct', 'item_seq']
    df_val = df_val.merge(valid_item_seq, on='user_log_acct', how='left')

    train_item_seq = train_fea_range.groupby(['user_log_acct'])['item_sku_id'].agg(list).reset_index()
    train_item_seq.columns = ['user_log_acct', 'item_seq']
    df_train = df_train.merge(train_item_seq, on='user_log_acct', how='left')

    return df_train, df_val


df = load_data("./ddata/mapped_merged_jdata.pkl")  # This is the starting file !

df_valid_label, data_valid_fea = \
    sliding_window(df, label_st=(4, 11), label_en=(4, 15), candidate_st=(4, 6), candidate_en=(4, 10), fea_en=(4, 10))

df_train_label, data_train_fea = \
    sliding_window(df, label_st=(4, 6), label_en=(4, 10), candidate_st=(4, 1), candidate_en=(4, 5), fea_en=(4, 5))

df_train, df_val, train_cols = get_feature(df, df_train_label, df_valid_label)
df_train, df_val = get_history(df_train, df_val, data_train_fea, data_valid_fea)

'''
train_array = np.array(df_train.loc[:, df_train.columns.drop("label")])
train_label = np.array(df_train.loc[:, "label"])
test_array = np.array(df_val.loc[:, df_val.columns.drop("label")])
test_label = np.array(df_val.loc[:, "label"])
'''

store_data(df_train, "./ddata/start_data/df_train.pkl")
store_data(df_val, "./ddata/start_data/df_val.pkl")

