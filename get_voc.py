from util import *
import numpy as np
import pandas as pd

df = load_data("./ddata/merged_DataFrame.pkl")
df.fillna(-1, inplace=True)


def build_dict(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    dict_ = dict(zip(key, range(len(key))))
    df.loc[:, col_name] = df[col_name].map(lambda x: dict_[x])
    return dict_


uid_voc = build_dict(df, "user_log_acct")
county_voc = build_dict(df, "county")
cateid_voc = build_dict(df, "item_third_cate_cd")
iid_voc = build_dict(df, "item_sku_id")
vender_voc = build_dict(df, "vender_id")
province_voc = build_dict(df, "province")
city_voc = build_dict(df, "city")
brand_voc = build_dict(df, "brand_code")
ulevel_voc = build_dict(df, "user_level")

store_data(uid_voc, "./ddata/vocs/user_voc.pkl")
store_data(cateid_voc, "./ddata/vocs/cat_voc.pkl")
store_data(iid_voc, "./ddata/vocs/iid_voc.pkl")
store_data(vender_voc, "./ddata/vocs/vender_voc.pkl")
store_data(province_voc, "./ddata/vocs/province_voc.pkl")
store_data(city_voc, "./ddata/vocs/city_voc.pkl")
store_data(county_voc, "./ddata/vocs/county_voc.pkl")
store_data(brand_voc, "./ddata/vocs/brand_voc.pkl")
store_data(ulevel_voc, "./ddata/vocs/ulevel_voc.pkl")

store_data(df, "./ddata/mapped_merged_jdata.pkl")