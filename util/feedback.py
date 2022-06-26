import numpy as np
import pandas as pd


def create_random_feedback(data_for_feedback):
    return np.random.rand(len(data_for_feedback))


def create_wrong_feedback(data_for_feedback,
                          target_colname='target'):
    # res = [1 if item == 0 else 0 for item in data_for_feedback[target_colname]]
    res = [1 for _ in data_for_feedback[target_colname]]
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
    _tmp = _check_length_constraint(data_for_feedback)
    res = res * _tmp
    return np.array(res)


def _create_criteria(criteria_df, target_colname, df, df_type='max'):
    all_criteria_list = []
    colnames = criteria_df.columns
    for i in range(criteria_df.shape[0]):
        criteria_list_for_one_target_value = []
        for colname in colnames:
            if colname == target_colname:
                criterion = (df[colname] == criteria_df[colname].iloc[i])
            else:
                if df_type == 'max':
                    criterion = (df[colname] <= criteria_df[colname].iloc[i])
                else:
                    criterion = (df[colname] >= criteria_df[colname].iloc[i])
            criteria_list_for_one_target_value.append(criterion)
        all_criteria_list.append(criteria_list_for_one_target_value)

    all_criteria = None
    for criteria_list_for_one_target_value in all_criteria_list:
        one_target_criterion = None
        for criterion in criteria_list_for_one_target_value:
            if one_target_criterion is None:
                one_target_criterion = criterion
            else:
                one_target_criterion = one_target_criterion & criterion
        if all_criteria is None:
            all_criteria = one_target_criterion
        else:
            all_criteria = all_criteria | one_target_criterion

    return all_criteria


def create_feedback_function_by_rule_base(df, max_df, min_df, target_colname):
    max_criteria = _create_criteria(
        max_df, target_colname, df, df_type='max')
    min_criteria = _create_criteria(
        min_df, target_colname, df, df_type='min')
    res = max_criteria & min_criteria
    return np.array([1 if bool_val else 0 for bool_val in res])
