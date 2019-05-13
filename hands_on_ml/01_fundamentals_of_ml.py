# coding: utf-8
"""
@author: zc12345 
@contact: 18292885866@163.com

@file: 01_fundamentals_of_ml.py
@time: 2019/4/22 17:19
@description:

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]  # choose some of data (n = 36)
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015":"GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indecies = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", "Life satisfaction"]].iloc[keep_indecies]


def main(gdp_data_path, oecd_data_path):

    # load data
    oecd_bli = pd.read_csv(oecd_data_path, thousands=',')  # thousands: 千分位分隔符, na_values: 用于替换NaN/NA的值
    gdp_per_capita = pd.read_csv(gdp_data_path, thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')

    # prepare data
    country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
    X = np.c_[country_stats["GDP per capita"]]
    y = np.c_[country_stats["Life satisfaction"]]

    # visualize data
    country_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")
    plt.show()

    # select a linear model
    model = linear_model.LinearRegression()

    # train model
    model.fit(X, y)

    # predict
    x_new = [[22587], [59994]]
    y_pred = model.predict(x_new)
    print(y_pred)


if __name__ == '__main__':
    data_path = './dataset/'
    gdp_data_path = data_path + 'gdp_per_capita.csv'
    oecd_data_path = data_path + 'oecd_bli_2015.csv'
    main(gdp_data_path, oecd_data_path)