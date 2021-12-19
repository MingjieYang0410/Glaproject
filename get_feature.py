from util import *
import pandas as pd
import datetime


# 'province', 'county','brand_code', 'shop_id', 'vender_id'
def map_to_id(df, col_name):
    for col in col_name:
        key = sorted(df[col].unique().tolist())
        dict_ = dict(zip(key, range(1, len(key) + 1)))  # 为了给mask留位置 否则0号会被严重影响
        df.loc[:, col] = df[col].map(lambda x: dict_[x])

    num_items = len(sorted(df["item_sku_id"].drop_duplicates(keep='first'))) + 1
    num_cats = len(sorted(df["item_third_cate_cd"].drop_duplicates(keep='first'))) + 1
    num_sex = len(sorted(df["sex"].drop_duplicates(keep='first'))) + 1
    num_ulevel = len(sorted(df["user_level"].drop_duplicates(keep='first'))) + 1
    num_atype = len(sorted(df["action_type"].drop_duplicates(keep='first'))) + 1
    num_city = len(sorted(df["city"].drop_duplicates(keep='first'))) + 1

    num_province = len(sorted(df["province"].drop_duplicates(keep='first'))) + 1
    num_county = len(sorted(df["county"].drop_duplicates(keep='first'))) + 1
    num_brand_code = len(sorted(df["brand_code"].drop_duplicates(keep='first'))) + 1
    num_shope = len(sorted(df["shop_id"].drop_duplicates(keep='first'))) + 1
    num_vender = len(sorted(df["vender_id"].drop_duplicates(keep='first'))) + 1

    temp = df[["item_sku_id", "item_third_cate_cd"]].sort_values("item_sku_id").drop_duplicates(subset='item_sku_id',
                                                                                                keep='first')
    cat_list = temp["item_third_cate_cd"].tolist()
    return num_items, num_cats, num_sex, num_ulevel, num_atype, num_city, \
           num_province, num_county, num_brand_code, num_shope, num_vender, cat_list


def _label_trans(x, dic_):
    if x in dic_:
        return 1
    else:
        return 0


def sliding_window_2_basic_form(df, label_start, label_end, inter_start, inter_end, fea_end):
    fea_list = []
    all_data = []
    for i in range(len(label_start)):
        # get times
        lb_st = df.loc[(df['month'] == label_start[i][0]) & (df['day'] == label_start[i][1]), 'month_day'].values[0]
        lb_en = df.loc[(df['month'] == label_end[i][0]) & (df['day'] == label_end[i][1]), 'month_day'].values[0]
        cand_st = df.loc[(df['month'] == inter_start[i][0]) & (df['day'] == inter_start[i][1]), 'month_day'].values[0]
        cand_en = df.loc[(df['month'] == inter_end[i][0]) & (df['day'] == inter_end[i][1]), 'month_day'].values[0]
        fea_position = df.loc[(df['month'] == fea_end[i][0]) & (df['day'] == fea_end[i][1]), 'month_day'].values[0]

        cand_bool = (df['month_day'] >= cand_st) & (df['month_day'] <= cand_en)
        label_bool = (df['month_day'] >= lb_st) & (df['month_day'] <= lb_en) & (df['action_type'] == 2)

        df_inter = df.loc[cand_bool].copy()  # get potential interactions
        df_inter = df_inter[['user_log_acct', 'item_sku_id', 'month_day']].copy()
        df_inter = df_inter.drop_duplicates(subset=['user_log_acct', 'item_sku_id'])
        df_inter = df_inter.loc[(df_inter.item_sku_id.isnull() == False)]  # process

        df_label = df.loc[label_bool].copy()  # get interactions of buying
        label = df_label[['user_log_acct', 'item_sku_id', 'day']].copy()  # process
        # add new columns
        df_inter['label'] = 0
        df_inter['user_item'] = df_inter['user_log_acct'].astype(str) + '_' + df_inter['item_sku_id'].astype(str)
        label['user_item'] = label['user_log_acct'].astype(str) + '_' + label['item_sku_id'].astype(str)

        dic_cnt = label['user_item'].value_counts().to_dict()
        df_inter['label'] = df_inter['user_item'].apply(lambda x: _label_trans(x, dic_cnt)).values
        all_data.append(df_inter)
        fea_list.append(fea_position)

    return all_data, fea_list


def get_feature(df, df_basic_list, feature_columns_user, feature_columns_item):
    """
    Input must by the most basic form return by sliding_window
    :param df_basic_train: Basic form for training set
    :param df_basic_val: Bsic for for validation set
    :param how: How many feature required?
    :return: merged basic form with other required information
    """
    data_with_feature = []
    for df_basic in df_basic_list:
        jd_user = df[feature_columns_user].drop_duplicates(['user_log_acct'], keep='first')
        jd_item = df[feature_columns_item].drop_duplicates(['item_sku_id'], keep='first')

        u_fea_cols = [col for col in jd_user.columns if col not in ['user_log_acct']]
        i_fea_cols = [col for col in jd_item.columns if col not in ['item_sku_id']]

        df_with_feature = df_basic.merge(jd_user, on='user_log_acct', how='left')  # Merge user features to basic form
        df_with_feature = df_with_feature.merge(jd_item, on='item_sku_id',
                                                how='left')  # Merger item features to basic form

        neg_df_train = df_with_feature[df_with_feature.label == 0].reset_index(drop=True)
        pos_df_train = df_with_feature[df_with_feature.label != 0].reset_index(drop=True)
        neg_df_train = neg_df_train.sample(n=int(len(pos_df_train) * 3))  # Negative Sampling

        df_with_feature = pd.concat([neg_df_train, pos_df_train], axis=0, ignore_index=True)
        data_with_feature.append(df_with_feature)
    return data_with_feature


def get_history_convert_type(df, df_withfea_mapped, fea_range_list):
    """
    This function is used to get the historical interaction of user
    :param df_train: output of get_feature
    :param df_val: output of get_feature
    :param train_fea_range:output of sliding_window
    :param val_fea_range: output of sliding_window
    :return: merged DataFrame from get_feature with additional users' historical info
    """
    df_final = []
    for i, df_sub in enumerate(df_withfea_mapped):
        ind_fea = (df['month_day'] <= fea_range_list[i])
        data_fea = df.loc[ind_fea].copy()

        df_sub.sort_values(['user_log_acct', 'month_day'], inplace=True)
        data_fea.sort_values(['user_log_acct', 'month_day'], inplace=True)

        item_seq = data_fea.groupby(['user_log_acct'])['item_sku_id'].agg(list).reset_index()
        item_seq.columns = ['user_log_acct', 'item_seq']
        df_sub = df_sub.merge(item_seq, on='user_log_acct', how='left')

        cate_seq = data_fea.groupby(['user_log_acct'])['item_third_cate_cd'].agg(list).reset_index()
        cate_seq.columns = ['user_log_acct', 'cate_seq']
        df_sub = df_sub.merge(cate_seq, on='user_log_acct', how='left')

        type_seq = data_fea.groupby(['user_log_acct'])['action_type'].agg(list).reset_index()
        type_seq.columns = ['user_log_acct', 'type_seq']
        df_sub = df_sub.merge(type_seq, on='user_log_acct', how='left')

        df_sub = df_sub.loc[(df_sub.item_seq.isnull() == False)]  # process
        df_final.append(df_sub)

    return df_final


def map_user_to_id(df_final):
    df_all = pd.concat(df_final, axis=0, ignore_index=True)
    key = sorted(df_all["user_log_acct"].unique().tolist())
    num_users = len(key)
    dict_ = dict(zip(key, range(len(key))))
    for i in range(len(df_final)):
        df_final[i].loc[:, "user_log_acct"] = df_final[i]["user_log_acct"].map(lambda x: dict_[x])

    return num_users


def gen_action_freq_feats(df, df_withfea_mapped, fea_range_list, start_date):
    df_final = []
    for i, df_sub in enumerate(df_withfea_mapped):
        ind_fea = (df['month_day'] <= fea_range_list[i])
        data_fea = df.loc[ind_fea].copy()
        key = ['user_log_acct']
        action = data_fea[key + ['action_type', 'action_time']].copy()

        for w in [1, 3, 5, 7, 15, 30]:
            bef_start_date = start_date[i] - datetime.timedelta(days=w)  # 留下从特征截至期间往前 到 前 1 ,3 ,5 ,7.. 天的行为

            action_cl = action[action['action_time'] >= bef_start_date].copy()
            data_fea = pd.get_dummies(action_cl['action_type'], prefix='_'.join(key) + '_last{}_days_action'.format(w))
            action_cl = pd.concat([action_cl, data_fea], axis=1)
            action_cl = action_cl.groupby(key, as_index=False).sum()
            action_cl['_'.join(key) + '_last{}_days_action_1_rt'.format(w)] = action_cl['_'.join(
                key) + '_last{}_days_action_2'.format(w)] / (1 + action_cl[
                '_'.join(key) + '_last{}_days_action_1'.format(w)])
            action_cl['_'.join(key) + '_last{}_days_action_3_rt'.format(w)] = action_cl['_'.join(
                key) + '_last{}_days_action_2'.format(w)] / (1 + action_cl[
                '_'.join(key) + '_last{}_days_action_3'.format(w)])
            action_cl['_'.join(key) + '_last{}_days_action_4_rt'.format(w)] = action_cl['_'.join(
                key) + '_last{}_days_action_2'.format(w)] / (1 + action_cl[
                '_'.join(key) + '_last{}_days_action_4'.format(w)])
            del action_cl['action_type']
            df_with_feature = df_sub.merge(action_cl, on=key, how='left')
        df_final.append(df_with_feature)

    return df_final


def gen_item_feats(df_item, df_final):
    df_item_fea = df_item.copy()

    for col in ['item_third_cate_cd', 'vender_id']:
        dic_ = df_item[col].value_counts().to_dict()
        df_item_fea['{}_cnt'.format(col)] = df_item_fea[col].map(dic_).values

    for col in ['shop_score']:
        dic_ = df_item.groupby('item_third_cate_cd')[col].mean().to_dict()
        df_item_fea['cate_{}_mean'.format(col)] = df_item_fea['item_third_cate_cd'].map(dic_).values

    for col in ['item_sku_id', 'brand_code']:
        dic_ = df_item.groupby('shop_id')[col].nunique()
        df_item_fea['shop_id_{}_nunique'.format(col)] = df_item_fea['shop_id'].map(dic_).values

    for col in ['item_sku_id', 'brand_code']:
        dic_ = df_item.groupby('item_third_cate_cd')[col].nunique()
        df_item_fea['item_third_cate_cd_{}_nunique'.format(col)] = df_item_fea['item_third_cate_cd'].map(dic_).values

    del df_item_fea['item_third_cate_cd']
    del df_item_fea['shop_id']
    del df_item_fea['brand_code']
    del df_item_fea['vender_id']
    df_with_item_fea = []
    for df in df_final:
        temp = df.merge(df_item_fea, on='item_sku_id', how='left')
        df_with_item_fea.append(temp)

    return df_with_item_fea


def get_ui_feats(df_origin, df_withfea_mapped, fea_range_list, start_date):
    df_final_ui = []
    for i, df_sub in enumerate(df_withfea_mapped):
        ind_fea = (df_origin['month_day'] <= fea_range_list[i])
        df = df_origin.loc[ind_fea].copy()
        df['user_item'] = df['user_log_acct'].astype(str) + '_' + df['item_sku_id'].astype(str)
        df_fea = df[['user_item']].copy()
        df_fea = df_fea.drop_duplicates(subset=['user_item'])
        # 1.宏观的特征: 不管是浏览还是其他操作，我们往下瞬移一个单位
        df['action_time_diff'] = df.groupby('user_item')['action_time'].shift().values
        df['action_time_diff'] = df['action_time'] - df['action_time_diff']
        df['action_time_diff'] = df['action_time_diff'].dt.seconds // 60  # 上一次对这个商品操作的时间到最近一次操作时间的时间间隔
        df['action_time_to_now'] = start_date[i] - df['action_time']  #
        df['action_time_to_now'] = df['action_time_to_now'].dt.seconds // 60  # 所有操作距离特征终点的时间间隔
        # 最后一次操作距离当前的时间
        dic_ = df.groupby('user_item')['action_time_to_now'].min().to_dict()
        df_fea['user_item_action_time_to_now_last'] = df_fea['user_item'].map(dic_).values  # 取最近一次操作距离当前的时间
        # 以当前位置为核心位置往前移动,过去三周每周的情况
        for days in [1, 3, 7, 14, 21]:
            tmp_ind = (df['action_time'] >= start_date[i] + datetime.timedelta(
                days=-1 * days))  # &(df['action_time'] <= st_time
            # 相邻两次操作 #
            df_tmp = df[tmp_ind].copy()
            dic_ = df_tmp.groupby('user_item')['day'].count().to_dict()  # 多少次出现这个user_item对
            df_fea['user_item_{}_day_cnt'.format(days)] = df_fea['user_item'].map(dic_).values  #
            dic_ = df_tmp.groupby('user_item')['day'].nunique().to_dict()  # 这个区间内 1，3，7.。。多少次交互
            df_fea['user_item_{}_day_nunique_pct'.format(days)] = df_fea['user_item'].map(
                dic_).values * 1.0 / days  # 平均每天
            dic_ = df_tmp.groupby('user_item')['action_time_diff'].mean().to_dict()  # 对这个商品平均间隔时间
            df_fea['user_item_{}_timediff_mean'.format(days)] = df_fea['user_item'].map(dic_).values
            dic_ = df_tmp.groupby('user_item')['action_time_diff'].std().to_dict()  # 时间间隔标准差
            df_fea['user_item_{}_timediff_std'.format(days)] = df_fea['user_item'].map(dic_).values
            dic_ = df_tmp.groupby('user_item')['action_time_diff'].median().to_dict()  # 时间间隔中位数
            df_fea['user_item_{}_timediff_median'.format(days)] = df_fea['user_item'].map(dic_).values

            for type_ in [1, 2, 3, 4]:

                ind_type = df['action_type'] == type_
                ind = tmp_ind & ind_type  # 只时间区间内并且类别符合的记录
                df_tmp = df[ind].copy()

                dic_ = df_tmp.groupby('user_item')['day'].count().to_dict()
                df_fea['type_{}_user_item_{}_day_cnt'.format(type_, days)] = df_fea['user_item'].map(
                    dic_).values  # 时间间隔内 某种行为的次数
                if days > 1 and type_ == 2:  # 单独对购买行为进行刻画
                    # 本次下单距离上一次下单的时间差的统计特征
                    df_tmp['action_time_diff'] = df_tmp.groupby('user_item')['action_time'].shift().values
                    df_tmp['action_time_diff'] = df_tmp['action_time'] - df_tmp['action_time_diff']
                    df_tmp['action_time_diff'] = df_tmp['action_time_diff'].dt.seconds // 60  # 本次下单距离上次下单时间
                    dic_ = df_tmp.groupby('user_item')['day'].nunique().to_dict()
                    df_fea['type_{}_user_item_{}_day_nunique_pct'.format(type_, days)] = df_fea['user_item'].map(
                        dic_).values * 1.0 / days
                    dic_ = df_tmp.groupby('user_item')['action_time_diff'].mean().to_dict()
                    df_fea['type_{}_user_item_{}_timediff_mean'.format(type_, days)] = df_fea['user_item'].map(
                        dic_).values
                    dic_ = df_tmp.groupby('user_item')['action_time_diff'].std().to_dict()
                    df_fea['type_{}_user_item_{}_timediff_std'.format(type_, days)] = df_fea['user_item'].map(
                        dic_).values
                    dic_ = df_tmp.groupby('user_item')['action_time_diff'].median().to_dict()
                    df_fea['type_{}_user_item_{}_timediff_median'.format(type_, days)] = df_fea['user_item'].map(
                        dic_).values
        df_temp = df_sub.merge(df_fea, on='user_item', how='left')
        df_final_ui.append(df_temp)

    return df_final_ui


def get_uc_feats(df_origin, df_withfea_mapped, fea_range_list, start_date):
    df_final_uc = []
    for i, df_sub in enumerate(df_withfea_mapped):
        df_sub['user_cate'] = df_sub['user_log_acct'].astype(str) + '_' + df_sub['item_third_cate_cd'].astype(str)
        ind_fea = (df_origin['month_day'] <= fea_range_list[i])
        df = df_origin.loc[ind_fea].copy()
        df['user_cate'] = df['user_log_acct'].astype(str) + '_' + df['item_third_cate_cd'].astype(str)
        df_fea = df[['user_cate']].copy()
        df_fea = df_fea.drop_duplicates(subset=['user_cate'])
        # 1.宏观的特征: 不管是浏览还是其他操作，我们往下瞬移一个单位
        df['action_time_diff'] = df.groupby('user_cate')['action_time'].shift().values
        df['action_time_diff'] = df['action_time'] - df['action_time_diff']
        df['action_time_diff'] = df['action_time_diff'].dt.seconds // 60
        df['action_time_to_now'] = start_date[i] - df['action_time']
        df['action_time_to_now'] = df['action_time_to_now'].dt.seconds // 60
        # 最后一次操作距离当前的时间
        dic_ = df.groupby('user_cate')['action_time_to_now'].min().to_dict()
        df_fea['user_cate_action_time_to_now_last'] = df_fea['user_cate'].map(dic_).values
        # 以当前位置为核心位置往前移动,过去三周每周的情况
        for days in [1, 3, 7, 14, 21, 30]:
            tmp_ind = (df['action_time'] >= start_date[i] + datetime.timedelta(
                days=-1 * days))  # &(df['action_time'] <= st_time
            # 相邻两次操作 #
            df_tmp = df[tmp_ind].copy()
            dic_ = df_tmp.groupby('user_cate')['day'].count().to_dict()
            df_fea['user_cate_{}_day_cnt'.format(days)] = df_fea['user_cate'].map(dic_).values
            dic_ = df_tmp.groupby('user_cate')['day'].nunique().to_dict()
            df_fea['user_cate_{}_day_nunique_pct'.format(days)] = df_fea['user_cate'].map(dic_).values * 1.0 / days
            dic_ = df_tmp.groupby('user_cate')['action_time_diff'].mean().to_dict()
            df_fea['user_cate_{}_timediff_mean'.format(days)] = df_fea['user_cate'].map(dic_).values
            dic_ = df_tmp.groupby('user_cate')['action_time_diff'].std().to_dict()
            df_fea['user_cate_{}_timediff_std'.format(days)] = df_fea['user_cate'].map(dic_).values
            dic_ = df_tmp.groupby('user_cate')['action_time_diff'].median().to_dict()
            df_fea['user_cate_{}_timediff_median'.format(days)] = df_fea['user_cate'].map(dic_).values

            for type_ in [1, 2, 3, 4]:
                ind_type = df['action_type'] == type_
                ind = tmp_ind & ind_type
                df_tmp = df[ind].copy()

                dic_ = df_tmp.groupby('user_cate')['day'].count().to_dict()
                df_fea['type_{}_user_cate_{}_day_cnt'.format(type_, days)] = df_fea['user_cate'].map(dic_).values
                if days > 1 and type_ == 2:
                    # 本次下单距离上一次下单的时间差的统计特征
                    df_tmp['action_time_diff'] = df_tmp.groupby('user_cate')['action_time'].shift().values
                    df_tmp['action_time_diff'] = df_tmp['action_time'] - df_tmp['action_time_diff']
                    df_tmp['action_time_diff'] = df_tmp['action_time_diff'].dt.seconds // 60
                    dic_ = df_tmp.groupby('user_cate')['day'].nunique().to_dict()
                    df_fea['type_{}_user_cate_{}_day_nunique_pct'.format(type_, days)] = df_fea['user_cate'].map(
                        dic_).values * 1.0 / days
                    dic_ = df_tmp.groupby('user_cate')['action_time_diff'].mean().to_dict()
                    df_fea['type_{}_user_cate_{}_timediff_mean'.format(type_, days)] = df_fea['user_cate'].map(
                        dic_).values
                    dic_ = df_tmp.groupby('user_cate')['action_time_diff'].std().to_dict()
                    df_fea['type_{}_user_cate_{}_timediff_std'.format(type_, days)] = df_fea['user_cate'].map(
                        dic_).values
                    dic_ = df_tmp.groupby('user_cate')['action_time_diff'].median().to_dict()
                    df_fea['type_{}_user_cate_{}_timediff_median'.format(type_, days)] = df_fea['user_cate'].map(
                        dic_).values
        df_temp = df_sub.merge(df_fea, on='user_cate', how='left')
        df_final_uc.append(df_temp)

    return df_final_uc


def get_ucs_feats(df_origin, df_withfea_mapped, fea_range_list, start_date):
    df_final_ucs = []
    for i, df_sub in enumerate(df_withfea_mapped):
        df_sub['user_cate_shop_id'] = df_sub['user_log_acct'].astype(str) + '_' + df_sub['item_third_cate_cd'].astype(
            str) + '_' + df_sub['shop_id'].astype(str)
        ind_fea = (df_origin['month_day'] <= fea_range_list[i])
        df = df_origin.loc[ind_fea].copy()
        df['user_cate_shop_id'] = df['user_log_acct'].astype(str) + '_' + df['item_third_cate_cd'].astype(str) + '_' + \
                                  df['shop_id'].astype(str)
        df_fea = df[['user_cate_shop_id']].copy()
        df_fea = df_fea.drop_duplicates(subset=['user_cate_shop_id'])
        # 1.宏观的特征: 不管是浏览还是其他操作，我们往下瞬移一个单位 #
        df['action_time_diff'] = df.groupby('user_cate_shop_id')['action_time'].shift().values
        print('shift')
        df['action_time_diff'] = df['action_time'] - df['action_time_diff']
        df['action_time_diff'] = df['action_time_diff'].dt.seconds // 60
        df['action_time_to_now'] = start_date[i] - df['action_time']
        df['action_time_to_now'] = df['action_time_to_now'].dt.seconds // 60

        # 最后一次操作距离当前的时间 #
        dic_ = df.groupby('user_cate_shop_id')['action_time_to_now'].min().to_dict()
        df_fea['user_cate_shop_id_action_time_to_now_last'] = df_fea['user_cate_shop_id'].map(dic_).values
        # 以当前位置为核心位置往前移动,过去三周每周的情况 #
        for days in [1, 3, 7, 14, 21]:
            tmp_ind = (df['action_time'] >= start_date[i] + datetime.timedelta(
                days=-1 * days))  # &(df['action_time'] <= st_time
            # 相邻两次操作 #
            df_tmp = df[tmp_ind].copy()
            dic_ = df_tmp.groupby('user_cate_shop_id')['day'].count().to_dict()
            df_fea['user_cate_shop_id_{}_day_cnt'.format(days)] = df_fea['user_cate_shop_id'].map(dic_).values
            dic_ = df_tmp.groupby('user_cate_shop_id')['day'].nunique().to_dict()
            df_fea['user_cate_shop_id_{}_day_nunique_pct'.format(days)] = df_fea['user_cate_shop_id'].map(
                dic_).values * 1.0 / days
            dic_ = df_tmp.groupby('user_cate_shop_id')['action_time_diff'].mean().to_dict()
            df_fea['user_cate_shop_id_{}_timediff_mean'.format(days)] = df_fea['user_cate_shop_id'].map(dic_).values
            dic_ = df_tmp.groupby('user_cate_shop_id')['action_time_diff'].std().to_dict()
            df_fea['user_cate_shop_id_{}_timediff_std'.format(days)] = df_fea['user_cate_shop_id'].map(dic_).values
            dic_ = df_tmp.groupby('user_cate_shop_id')['action_time_diff'].median().to_dict()
            df_fea['user_cate_shop_id_{}_timediff_median'.format(days)] = df_fea['user_cate_shop_id'].map(dic_).values

            for type_ in [1, 2, 3, 4]:
                ind_type = df['action_type'] == type_
                ind = tmp_ind & ind_type
                df_tmp = df[ind].copy()

                dic_ = df_tmp.groupby('user_cate_shop_id')['day'].count().to_dict()
                df_fea['type_{}_user_cate_shop_id_{}_day_cnt'.format(type_, days)] = df_fea['user_cate_shop_id'].map(
                    dic_).values
                if days > 1 and type_ == 2:
                    # 本次下单距离上一次下单的时间差的统计特征 #
                    df_tmp['action_time_diff'] = df_tmp.groupby('user_cate_shop_id')['action_time'].shift().values
                    df_tmp['action_time_diff'] = df_tmp['action_time'] - df_tmp['action_time_diff']
                    df_tmp['action_time_diff'] = df_tmp['action_time_diff'].dt.seconds // 60
                    dic_ = df_tmp.groupby('user_cate_shop_id')['day'].nunique().to_dict()
                    df_fea['type_{}_user_cate_shop_id_{}_day_nunique_pct'.format(type_, days)] = df_fea[
                                                                                                     'user_cate_shop_id'].map(
                        dic_).values * 1.0 / days
                    dic_ = df_tmp.groupby('user_cate_shop_id')['action_time_diff'].mean().to_dict()
                    df_fea['type_{}_user_cate_shop_id_{}_timediff_mean'.format(type_, days)] = df_fea[
                        'user_cate_shop_id'].map(dic_).values
                    dic_ = df_tmp.groupby('user_cate_shop_id')['action_time_diff'].std().to_dict()
                    df_fea['type_{}_user_cate_shop_id_{}_timediff_std'.format(type_, days)] = df_fea[
                        'user_cate_shop_id'].map(dic_).values
                    dic_ = df_tmp.groupby('user_cate_shop_id')['action_time_diff'].median().to_dict()
                    df_fea['type_{}_user_cate_shop_id_{}_timediff_median'.format(type_, days)] = df_fea[
                        'user_cate_shop_id'].map(dic_).values
        df_temp = df_sub.merge(df_fea, on='user_cate_shop_id', how='left')
        df_final_ucs.append(df_temp)

    return df_final_ucs


df = load_data(("./ddata/merged_DataFrame_fillna.pkl"))

col = ["item_sku_id", "item_third_cate_cd", "sex",
       "action_type", "city", "user_level", 'province', 'county','brand_code', 'shop_id', 'vender_id']

num_items, num_cats, num_sex, num_ulevel, num_atype, num_city, num_province, num_county, num_brand_code, num_shope,num_vender, cat_list = map_to_id(df, col)

label_start = [(4, 6), (4,3),(3,31), (3,28), (3,25), (3,22), (3,19), (3, 16), (3, 13), (3, 10), (3, 7), (3, 4), (3, 1)]
label_end =   [(4, 15),(4,12),(4,9), (4, 6), (4,3), (3,31), (3,28),  (3,25),  (3, 22), (3,19), (3, 16), (3,13), (3,10)]
inter_start = [(4, 3), (3,31),(3,28), (3,25), (3,22), (3,19), (3,16), (3, 13), (3, 10), (3, 7), (3, 4), (3, 1), (2, 26)]
inter_end =   [(4, 5), (4,2),(3,30), (3,27), (3,24), (3,21), (3,18), (3, 15), (3, 12), (3, 9), (3, 6), (3, 3),(2,28)]
fea_end = [(4, 5), (4,2),(3,30), (3,27), (3,24), (3,21), (3,18), (3, 15), (3, 12), (3, 9), (3, 6), (3, 3), (2,28)]


fea_end_date = [datetime.datetime(2020, 4, 5), datetime.datetime(2020, 4, 2), datetime.datetime(2020, 3, 30), datetime.datetime(2020, 3, 27),
               datetime.datetime(2020, 3, 24), datetime.datetime(2020, 3, 21), datetime.datetime(2020, 3, 18), datetime.datetime(2020, 3, 15),
               datetime.datetime(2020, 3, 12), datetime.datetime(2020, 3, 9), datetime.datetime(2020, 3, 6), datetime.datetime(2020, 3, 3),
               datetime.datetime(2020, 2, 28)]


all_data, fea_list = sliding_window_2_basic_form(df, label_start, label_end, inter_start, inter_end, fea_end)

df_with_feature = get_feature(df, all_data, ["user_log_acct", "sex", "city", "user_level", 'province', 'county'],
                                              ["item_sku_id", "item_third_cate_cd",'brand_code', 'shop_id', 'vender_id'])

jd_item = df[['item_sku_id','brand_code','shop_id','item_third_cate_cd','vender_id','shop_score']].drop_duplicates(['item_sku_id'], keep='first')
df_final = gen_item_feats(jd_item, df_with_feature)
df_final = get_ucs_feats(df, df_final, fea_list, fea_end_date)
df_final = get_ui_feats(df, df_final, fea_list, fea_end_date)
df_final = get_uc_feats(df, df_final, fea_list, fea_end_date)

df_final = get_history_convert_type(df, df_final, fea_list)

df_final = gen_action_freq_feats(df, df_final, fea_list, fea_end_date)

store_data(df_final,"./ddata/start_data/df_final.pkl")
#num_users = map_user_to_id(df_final)
num_users = 0

df_test = df_final[0].sample(frac=1)
#df_test_dev = df_final[0].sample(frac=1)
#df_test = df_test_dev[:len(df_test_dev) // 2]
#df_dev = df_test_dev[len(df_test_dev) // 2:]
df_train = df_final[1:]
df_train = pd.concat(df_train, axis=0, ignore_index=True)

store_data(df_train,"./ddata/start_data/df_train.pkl")
store_data(df_test, "./ddata/start_data/df_test.pkl")

store_data(num_users, "./ddata/start_data/num_users.pkl")
store_data(num_items, "./ddata/start_data/num_items.pkl")
store_data(num_cats, "./ddata/start_data/num_cats.pkl")
store_data(num_sex, "./ddata/start_data/num_sex.pkl")
store_data(num_ulevel, "./ddata/start_data/num_ulevel.pkl")
store_data(num_atype, "./ddata/start_data/num_atype.pkl")
store_data(num_city, "./ddata/start_data/num_city.pkl")

store_data(num_province, "./ddata/start_data/num_province.pkl")
store_data(num_county, "./ddata/start_data/num_county.pkl")
store_data(num_brand_code, "./ddata/start_data/num_brand_code.pkl")
store_data(num_shope, "./ddata/start_data/num_shope.pkl")
store_data(num_vender, "./ddata/start_data/num_vender.pkl")

store_data(cat_list, "./ddata/start_data/cat_list.pkl")

df_train = load_data("./ddata/start_data/df_train.pkl")
#df_dev = load_data("./ddata/start_data/df_dev.pkl")
df_test = load_data("./ddata/start_data/df_test.pkl")

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

cat_list = load_data("./ddata/start_data/cat_list.pkl") # remember to add one when use