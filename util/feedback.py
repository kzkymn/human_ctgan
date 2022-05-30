import numpy as np
import pandas as pd


def create_random_feedback(data_for_feedback):
    return np.random.rand(len(data_for_feedback))


def create_wrong_feedback(data_for_feedback,
                          target_colname='target'):
    res = [0 if item != 2 else np.random.rand(
        1)[0] for item in data_for_feedback[target_colname]]
    return np.array(res)


def create_feedback_from_csv(csv_path,
                             target_colname='feedback'):
    return pd.read_csv(csv_path, encoding='utf-8')[target_colname].values
