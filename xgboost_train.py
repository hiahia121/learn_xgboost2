# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import xgboost as xgb
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import plot_importance
from xgboost import plot_tree


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    filemode='w')

def get_train_data():
    # pandas读取数据
    Train_data = pd.read_csv('./data/test_io_data.csv', sep=',')
    logging.info('Train data shape:', Train_data.shape)
    # pandas查看head
    logging.info(Train_data.head())

    # 获取列名称
    train_columns = Train_data.columns
    feature_cols = [col for col in train_columns if col not in ['time', 'interval', 'i/o']]

    # 训练样本X
    X_data = Train_data[feature_cols]
    logging.info(X_data.head())

    # 训练样本Y
    Y_data = Train_data['i/o']
    logging.info(Y_data.head())
    return X_data, Y_data


# 建xgb模型
def build_model_xgb(x_train, y_train):
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, gamma=0, subsample=0.8,
                             colsample_bytree=0.9, max_depth=7)
    model.fit(x_train, y_train)
    return model

# 创建feature_map
def create_feature_map(features):
    outfile = open("./data/xgb.fmap", "w")
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


if __name__ == '__main__':
    X_data, Y_data = get_train_data()
    x_train, x_for_test, y_train, y_for_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=1)
    create_feature_map(x_train.columns)
    model_xgb = build_model_xgb(x_train, y_train)
    # 打印特征的重要程度
    # plot_importance(model_xgb)
    # 画出树
    plot_tree(model_xgb, fmap="./data/xgb.fmap", num_trees=1, rankdir='LR')
    plt.show()
    # save model
    pickle.dump(model_xgb, open("./data/xgboost_pickle.dat", "wb"))
    # load model
    #loaded_model = pickle.load(open("./data/xgboost_pickle.dat", "rb"))
    y_test_predict = model_xgb.predict(x_for_test)
    MAE_xgb = mean_absolute_error(y_for_test, y_test_predict)
    logging.info("MAE xgb is %s", MAE_xgb)