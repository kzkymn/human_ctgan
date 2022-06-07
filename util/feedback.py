import numpy as np
import pandas as pd


def create_random_feedback(data_for_feedback):
    return np.random.rand(len(data_for_feedback))


def create_wrong_feedback(data_for_feedback,
                          target_colname='target'):
    res = [1 if item == 0 else 0 for item in data_for_feedback[target_colname]]
    return np.array(res)


def create_feedback_from_csv(csv_path,
                             target_colname='feedback'):
    return pd.read_csv(csv_path, encoding='utf-8')[target_colname].values


def create_feedback_by_one_class_svm(data_for_feedback,
                                     one_class_svm):
    res = one_class_svm.predict(data_for_feedback)
    _tmp = _check_length_constraint(data_for_feedback)
    res = res * _tmp
    return np.array([0 if i == -1 else 1 for i in res])


def _check_length_constraint(data_for_feedback):
    res = None
    for c in data_for_feedback.columns:
        _tmp = [True if item > 0 else False for item in data_for_feedback[c]]
        if res is None:
            res = _tmp
        else:
            res = res and _tmp
    return res


def create_feedback_by_knn(data_for_feedback,
                           knn,
                           target_colname='target'):
    pred_target = knn.predict(data_for_feedback.drop(columns=target_colname))
    res = pred_target == data_for_feedback[target_colname]
    _tmp = _check_length_constraint(data_for_feedback)
    res = res * _tmp
    return np.array([1 if i is True else 0 for i in res])


def create_feedback_by_knn_with_proba(data_for_feedback,
                                      knn,
                                      target_colname='target'):
    pred_proba_of_target = knn.predict_proba(
        data_for_feedback.drop(columns=target_colname))
    target_labels = data_for_feedback[target_colname]

    res = pred_proba_of_target[np.arange(
        len(pred_proba_of_target)), target_labels]
    return np.array(res)
