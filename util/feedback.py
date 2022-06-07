import numpy as np
import pandas as pd


def create_random_feedback(data_for_feedback):
    return np.random.rand(len(data_for_feedback))


def create_wrong_feedback(data_for_feedback,
                          target_colname='target'):
    res = [0 if item != 2 else 1 for item in data_for_feedback[target_colname]]
    return np.array(res)


def create_feedback_from_csv(csv_path,
                             target_colname='feedback'):
    return pd.read_csv(csv_path, encoding='utf-8')[target_colname].values


def create_feedback_by_one_class_svm(data_for_feedback,
                                     one_class_svm):
    res = one_class_svm.predict(data_for_feedback)
    return np.array([0 if i == -1 else 1 for i in res])


def create_feedback_by_knn(data_for_feedback,
                           knn,
                           target_colname='target'):
    pred_target = knn.predict(data_for_feedback.drop(columns=target_colname))
    res = pred_target & data_for_feedback[target_colname]
    return np.array(res)
