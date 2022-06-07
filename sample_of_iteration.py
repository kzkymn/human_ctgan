# %%
from functools import partial

from IPython import get_ipython
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from hctgan import HCTGANSynthesizer
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_iris
import torch
from xgboost import XGBClassifier

from util import (create_feedback_by_one_class_svm,
                  create_feedback_by_knn,
                  print_diff_between_original_and_generated_data)

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%[markdown]
# Options

# %%
synth_sample_size = 4000
feedback_synth_sample_size = 100
r = 10
sigma = 0.005
hctgan_path = './checkpoint/htcgan_test.pth'
feedback_path = './output/feedbacks_test.csv'
feedback_colname = 'feedback'
target_colname = 'target'

# %%


def get_feedback_function_by_one_class_svm(one_class_svm):
    return partial(create_feedback_by_one_class_svm, one_class_svm=one_class_svm)


def get_feedback_function_by_knn(knn):
    return partial(create_feedback_by_knn, knn=knn)


# %%
X, y = load_iris(return_X_y=True, as_frame=True)
df = X.copy()
df[target_colname] = y.copy()
unique_y_labels_num = len(y.unique())

# one_class_svm = OneClassSVM().fit(df)
# feedback_function = get_feedback_function_by_one_class_svm(one_class_svm)
knn = KNeighborsClassifier().fit(X, y)
feedback_function = get_feedback_function_by_knn(knn)

# %%
df_0 = df[df[target_colname] == 0]
df_1_or_2 = df[~(df[target_colname] == 0)]

# %%
df_0.describe()

# %%
df_1_or_2.describe()

# %%
train_criterion_of_df_0 = (df_0['sepal length (cm)'] < 4.8) & (
    df_0['petal length (cm)'] < 1.4)
train_criterion_of_df_1_or_2 = (df_1_or_2['sepal length (cm)'] < 5.8) & (
    df_1_or_2['petal length (cm)'] > 4.4)
df_0_train = df_0[train_criterion_of_df_0]

df_1_or_2_train = df_1_or_2[train_criterion_of_df_1_or_2]

df_train = pd.concat([df_0_train, df_1_or_2_train])
df_test = df

del df_0, df_1_or_2, df_0_train, df_1_or_2_train

# %%
df_train.describe()

# %%
df_test.describe()

# %%
X_test = df_test.drop(columns=target_colname)
y_test = df_test[target_colname]

# %%
discrete_columns = [
    target_colname
]

# %%


def create_seed_tuple(np_seed=333, torch_seed=333):
    return (np.random.RandomState(np_seed), torch.manual_seed(torch_seed))


def print_roc_auc_score(y_true, y_pred_proba, unique_y_labels_num):
    if unique_y_labels_num >= 3:
        auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    else:
        auc_score = roc_auc_score(y_true, y_pred_proba)
    print('roc_auc_score:', auc_score)
    return auc_score


def create_classifier_with_synthed_data(df_train, df_synthed):
    clf = XGBClassifier()
    tmp_df = pd.concat([df_train, df_synthed])
    # tmp_df = df_synthed
    tmp_X = tmp_df.drop(columns=target_colname)
    tmp_y = tmp_df[target_colname]
    clf.fit(tmp_X, tmp_y)
    return clf


def calc_roc_using_the_result_of_fitting_and_evaluating_model(df_train, df_synthed, X_test, y_test, unique_y_labels_num):
    clf = create_classifier_with_synthed_data(df_train, df_synthed)
    y_pred_proba = clf.predict_proba(X_test)
    return print_roc_auc_score(y_test, y_pred_proba, unique_y_labels_num)


# %%
seed_tuple = create_seed_tuple()

hctgan = HCTGANSynthesizer()
hctgan.random_states = seed_tuple
hctgan.fit(df_train,
           discrete_columns=discrete_columns)
hctgan.random_states = seed_tuple
ctgan_synthed_df = hctgan.sample(synth_sample_size)

# %%
print('================')
print('== Only CTGAN ==')
print('================')
hctgan.random_states = seed_tuple
print_diff_between_original_and_generated_data(df_train, hctgan)
original_roc_auc_score = calc_roc_using_the_result_of_fitting_and_evaluating_model(
    df_train, ctgan_synthed_df, X_test, y_test, unique_y_labels_num)

# %%
roc_auc_score_list = []
roc_auc_score_list.append(original_roc_auc_score)

# %%
for i in range(100):
    hctgan.random_states = None
    hctgan.save(path=hctgan_path)

    hctgan.random_states = seed_tuple
    hctgan.create_feedback_data_csv(csv_path=feedback_path,
                                    n=feedback_synth_sample_size,
                                    r=r,
                                    sigma=sigma,
                                    target_colname=feedback_colname)

    feedback_df = pd.read_csv(feedback_path)
    data_for_feedback_orig = feedback_df.drop(columns=feedback_colname)
    feedback_probs = feedback_function(data_for_feedback_orig)

    hctgan: HCTGANSynthesizer = HCTGANSynthesizer.load(path=hctgan_path)

    hctgan.random_states = seed_tuple
    sampled_data_tensor, data_for_feedback, perturbations = hctgan.sample_for_human_evaluation(n=feedback_synth_sample_size,
                                                                                               r=r,
                                                                                               sigma=sigma)

    np.testing.assert_array_almost_equal(
        data_for_feedback_orig, data_for_feedback)

    hctgan.random_states = seed_tuple
    hctgan.fit_to_feedback(sampled_data_tensor,
                           feedback_probs,
                           perturbations,
                           sigma=sigma)

    print(f'=== i {i: 03d} ===')
    hctgan.random_states = seed_tuple
    synthed_df = hctgan.sample(synth_sample_size)
    hctgan.random_states = seed_tuple
    # print_diff_between_original_and_generated_data(
    #     df_train, hctgan, show_original_data_info=False)
    intermediate_roc_auc_score = calc_roc_using_the_result_of_fitting_and_evaluating_model(
        df_train, synthed_df, X_test, y_test, unique_y_labels_num)
    roc_auc_score_list.append(intermediate_roc_auc_score)

# %%
final_sample = hctgan.sample(synth_sample_size)

print('=============')
print('=== Final ===')
print('=============')
hctgan.random_states = seed_tuple
print_diff_between_original_and_generated_data(
    df_train, hctgan)
final_roc_auc_score = calc_roc_using_the_result_of_fitting_and_evaluating_model(
    df_train, final_sample, X_test, y_test, unique_y_labels_num)
roc_auc_score_list.append(final_roc_auc_score)

# %%
ctgan_synthed_df.describe()

# %%
final_sample.describe()

# %% [markdown]
# roc auc score in each epoch

# %%
df_result = pd.DataFrame()
df_result['epoch'] = list(range(1, len(roc_auc_score_list)+1))
df_result['roc_auc_score'] = roc_auc_score_list
df_result.plot('epoch', 'roc_auc_score')

# %%
print('original roc_auc_score:', roc_auc_score_list[0])
print('final roc_auc_score:', roc_auc_score_list[-1])

# %%
