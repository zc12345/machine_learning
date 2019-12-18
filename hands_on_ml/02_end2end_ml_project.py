# coding: utf-8
"""
@author: zc12345 
@contact: 18292885866@163.com

@file: 02_end2end_ml_project.py
@time: 2019/4/22 22:02
@description:

"""

import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from zlib import crc32
import hashlib


def download_data(url, dataset_dir):
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    filename = url.split('/')[-1]
    tgz_path = os.path.join(dataset_dir, filename)
    urllib.request.urlretrieve(url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(path=dataset_dir)
    data_tgz.close()


def load_data(data_path):
    csv_path = os.path.join(data_path, 'housing.csv')
    return pd.read_csv(csv_path)


def save_fig(fig_id, tight_layout=True, img_dir="images", fig_extension="png", resolution=300):
    path = os.path.join(img_dir, fig_id+'.'+fig_extension)
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    print("save figure ", path)


def show_hist(data):
    data.hist(bins=50, figsize=(20, 15))
    plt.show()


def vis_data(data):
    data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
              s=data["population"]/100, label="population", figsize=(10, 7),
              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
              sharex=False)
    plt.legend()
    save_fig("vis_plot")
    plt.show()


def show_corr(data):
    corr_matrix = data.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(data[attributes], figsize=(12,8))
    save_fig("scatter_mat")
    plt.show()


def split_train_test(data, test_ratio):
    shuffle_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check_crc32(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def test_set_check_md5(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_ids(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check_crc32(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def stratified_split_train_test(data, test_ratio):
    # by income
    data = data.copy()
    data["income_cat"] = np.ceil(data["median_income"] / 1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True) # above 5 inplace with 5.0
    # print(data["income_cat"].value_counts())
    # data["income_cat"].hist()
    # plt.show()

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    for set_ in strat_train_set, strat_test_set:
        set_.drop("income_cat", axis=1, inplace=True)  # remove income_cat property

    return strat_train_set, strat_test_set


def main():
    download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    download_url = download_root + "datasets/housing/housing.tgz"
    dataset_dir = os.path.join("dataset", "housing")
#    download_data(download_url, dataset_dir)
    data = load_data(dataset_dir)
    np.random.seed(50)

    # split randomly
    # train_set, test_set = split_train_test(data, 0.2)  # 1/5 test data, 4/5 train data

    # split by id or index
    # data_with_id = data.reset_index()  # add index
    # data_with_id["id"] = data["longitude"] * 1000 + data["latitude"]
    # train_set, test_set = split_train_test_by_ids(data_with_id, 0.2, "index")
    # # train_set, test_set = split_train_test_by_ids(data_with_id, 0.2, "id")

    # stratified split
#    train_set, test_set = stratified_split_train_test(data, 0.2)
    show_corr(data)
    data.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1)
    plt.axis([0, 16, 0, 550000])
    save_fig("income_vs_house_value_scatterplot")
    plt.show()


if __name__ == '__main__':
    main()