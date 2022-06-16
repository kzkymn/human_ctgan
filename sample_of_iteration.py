# %%
from functools import partial
import warnings

from IPython import get_ipython
from mlxtend.plotting import plot_decision_regions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from hctgan import HCTGANSynthesizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_iris
import torch
from xgboost import XGBClassifier

from util import (create_random_feedback,
                  create_wrong_feedback,
                  create_feedback_by_one_class_svm,
                  create_feedback_by_knn,
                  create_feedback_by_knn_with_proba,
                  print_diff_between_original_and_generated_data)
from util.feedback import create_feedback_by_knn_with_proba

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
warnings.filterwarnings('ignore')

# %%[markdown]
# Settings

# %%
SAMPLE_SIZE_OF_SYNTHESIZED_DATA = 200
SAMPLE_SIZE_OF_FEEDBACK_DATA = 100
PERTURBATION_PER_FEEDBACK_DATUM = 2
PERTURBATION_SIGMA = 10
HCTGAN_FILE_PATH = './checkpoint/htcgan_test'
FEEDBACK_CSV_PATH = './output/feedbacks_test.csv'
FEEDBACK_COLNAME = 'feedback'
TARGET_COLNAME = 'target'

# %%
# NOTE: If any of the explanatory or objective variables are categorical,
# please include their names in this list.
discrete_columns = [
    TARGET_COLNAME
]


# %%


def get_feedback_function_by_one_class_svm(df):
    one_class_svm = OneClassSVM().fit(df)
    return partial(create_feedback_by_one_class_svm, one_class_svm=one_class_svm)


def get_feedback_function_by_knn(df, target_colname='target'):
    X = df.drop(columns=target_colname)
    y = df[target_colname]

    knn = KNeighborsClassifier().fit(X, y)
    return partial(create_feedback_by_knn, knn=knn)
    # return partial(create_feedback_by_knn_with_proba, knn=knn)


# %%
def get_iris_dataframe(target_colname='target'):
    X, y = load_iris(return_X_y=True, as_frame=True)
    X = X[['sepal length (cm)', 'sepal width (cm)']]
    df = X.copy()
    df[target_colname] = y.copy()
    return df


def prepare_train_and_test(df,
                           target_colname='target'):
    X = df.drop(columns=target_colname)
    y = df[target_colname]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    df_train = X_train.copy()
    df_train[target_colname] = y_train
    df_test = X_test.copy()
    df_test[target_colname] = y_test
    df_0 = df_train[df_train[target_colname] == 0]
    df_1_or_2 = df_train[~(df_train[target_colname] == 0)]

    train_criterion_of_df_0 = (df_0['sepal length (cm)'] <= 4.8) & (
        df_0['sepal width (cm)'] >= 3.5)
    train_criterion_of_df_1_or_2 = (df_1_or_2['sepal length (cm)'] >= 6.0) & (
        df_1_or_2['sepal length (cm)'] <= 6.5)

    df_0_train = df_0[train_criterion_of_df_0]
    df_0_test = df_0[~train_criterion_of_df_0]

    df_1_or_2_train = df_1_or_2[train_criterion_of_df_1_or_2]
    df_1_or_2_test = df_1_or_2[~train_criterion_of_df_1_or_2]

    df_train = pd.concat([df_0_train, df_1_or_2_train])
    df_test = pd.concat([df_test, df_0_test, df_1_or_2_test])

    return df_train, df_test


df = get_iris_dataframe(target_colname=TARGET_COLNAME)
df_train, df_test = prepare_train_and_test(df,
                                           target_colname=TARGET_COLNAME)

# %%
df_train.describe()

# %%
df_test.describe()

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


def create_classifier_with_synthed_data(df_train, df_synthed,
                                        target_colname='target'):
    clf = XGBClassifier()
    if df_synthed is not None:
        tmp_df = pd.concat([df_train, df_synthed])
    else:
        tmp_df = df_train
    tmp_X = tmp_df.drop(columns=target_colname).values
    tmp_y = tmp_df[target_colname].values
    clf.fit(tmp_X, tmp_y)
    return clf


def calc_roc_using_the_result_of_fitting_and_evaluating_model(df_train, df_synthed, df_test, unique_y_labels_num,
                                                              target_colname='target',
                                                              skip_rendering_graph=False):
    X_test = df_test.drop(columns=target_colname)
    y_test = df_test[target_colname]
    clf = create_classifier_with_synthed_data(
        df_train, df_synthed, target_colname=target_colname)
    y_pred_proba = clf.predict_proba(X_test)
    if not skip_rendering_graph:
        plot_decision_regions(X=X_test.values, y=y_test.values, clf=clf)
    return print_roc_auc_score(y_test, y_pred_proba, unique_y_labels_num)


# %%
seed_tuple = create_seed_tuple()

hctgan = HCTGANSynthesizer(sample_size_of_feedback_data=SAMPLE_SIZE_OF_FEEDBACK_DATA,
                           perturbation_per_feedback_datum=PERTURBATION_PER_FEEDBACK_DATUM,
                           perturbation_sigma=PERTURBATION_SIGMA,
                           epochs=3000)
hctgan.random_states = seed_tuple
hctgan.fit(df_train,
           discrete_columns=discrete_columns)
hctgan.random_states = seed_tuple
first_synthed_df = hctgan.sample(SAMPLE_SIZE_OF_SYNTHESIZED_DATA)

# %%
unique_y_labels_num = len(df[TARGET_COLNAME].unique())

print('================================')
print('== Training only original data==')
print('================================')
hctgan.random_states = seed_tuple
print_diff_between_original_and_generated_data(df_train, hctgan)
original_roc_auc_score = calc_roc_using_the_result_of_fitting_and_evaluating_model(
    df_train, None, df_test, unique_y_labels_num,
    target_colname=TARGET_COLNAME)

# %%
roc_auc_score_list = []
# roc_auc_score_list.append(original_roc_auc_score)

# %%
print('================')
print('== Only CTGAN ==')
print('================')
hctgan.random_states = seed_tuple
print_diff_between_original_and_generated_data(df_train, hctgan)
original_and_ctgan_roc_auc_score = calc_roc_using_the_result_of_fitting_and_evaluating_model(
    df_train, first_synthed_df, df_test, unique_y_labels_num,
    target_colname=TARGET_COLNAME)

# %%
roc_auc_score_list.append(original_and_ctgan_roc_auc_score)

# %%
# feedback_function = get_feedback_function_by_one_class_svm(df)
feedback_function = get_feedback_function_by_knn(df)
# feedback_function = create_random_feedback
# feedback_function = create_wrong_feedback


# %%

def iterate_feedbacks(hctgan,
                      roc_auc_score_list,
                      sample_size_of_synthesized_data,
                      sample_size_of_feedback_data,
                      perturbation_per_feedback_datum,
                      perturbation_sigma,
                      hctgan_file_path,
                      feedback_csv_path,
                      feedback_colname,
                      target_colname='target',
                      start_n=1,
                      iter_n=20):
    current_sigma = perturbation_sigma
    end_n = start_n + iter_n

    for i in range(start_n, end_n):
        hctgan.save(path=hctgan_file_path)

        hctgan.random_states = seed_tuple
        hctgan.create_feedback_data_csv(csv_path=feedback_csv_path,
                                        sample_size_of_feedback_data=sample_size_of_feedback_data,
                                        perturbation_per_feedback_datum=perturbation_per_feedback_datum,
                                        perturbation_sigma=current_sigma,
                                        target_colname=feedback_colname)

        feedback_df = pd.read_csv(feedback_csv_path)
        data_for_feedback_orig = feedback_df.drop(columns=feedback_colname)
        feedback_probs = feedback_function(data_for_feedback_orig)

        hctgan: HCTGANSynthesizer = HCTGANSynthesizer.load(
            path=hctgan_file_path)

        hctgan.random_states = seed_tuple
        hctgan.fit_to_feedback(data_for_feedback_orig,
                               feedback_probs)

        print(f'=== i {i: 03d} ===')
        hctgan.random_states = seed_tuple
        synthed_df = hctgan.sample(sample_size_of_synthesized_data)
        hctgan.random_states = seed_tuple
        intermediate_roc_auc_score = calc_roc_using_the_result_of_fitting_and_evaluating_model(
            df_train, synthed_df, df_test, unique_y_labels_num,
            target_colname=target_colname,
            skip_rendering_graph=True)
        roc_auc_score_list.append(intermediate_roc_auc_score)

    intermediate_roc_auc_score = calc_roc_using_the_result_of_fitting_and_evaluating_model(
        df_train, synthed_df, df_test, unique_y_labels_num,
        target_colname=target_colname)

    return current_sigma


# %%
current_sigma = iterate_feedbacks(hctgan,
                                  roc_auc_score_list,
                                  sample_size_of_synthesized_data=SAMPLE_SIZE_OF_SYNTHESIZED_DATA,
                                  sample_size_of_feedback_data=SAMPLE_SIZE_OF_FEEDBACK_DATA,
                                  perturbation_per_feedback_datum=PERTURBATION_PER_FEEDBACK_DATUM,
                                  perturbation_sigma=PERTURBATION_SIGMA,
                                  hctgan_file_path=HCTGAN_FILE_PATH,
                                  feedback_csv_path=FEEDBACK_CSV_PATH,
                                  feedback_colname=FEEDBACK_COLNAME,
                                  target_colname=TARGET_COLNAME,
                                  iter_n=2, start_n=1)

# %%
print('===================================')
print('=== Result of 2 epoch feedbacks ===')
print('===================================')
hctgan.random_states = seed_tuple
print_diff_between_original_and_generated_data(
    df_train, hctgan)
print('original roc_auc_score:', roc_auc_score_list[0])
print('current roc_auc_score:', roc_auc_score_list[-1])

# %%
first_synthed_df.describe()

# %%
hctgan.random_states = seed_tuple
intermediate_synthed_df = hctgan.sample(SAMPLE_SIZE_OF_SYNTHESIZED_DATA)
intermediate_synthed_df.describe()

# %%
df.describe()

# %% [markdown]
# roc auc score in each epoch
#
# Epoch 0 is the result of the training data and CTGAN sythesized data.

# %%
print(feedback_function)
df_result = pd.DataFrame()
df_result['epoch'] = list(range(len(roc_auc_score_list)))
df_result['roc_auc_score'] = roc_auc_score_list
df_result['baseline'] = roc_auc_score_list[0]
ax1 = df_result.plot('epoch', 'roc_auc_score')
df_result.plot('epoch', 'roc_auc_score', kind='scatter', color='b', ax=ax1)
df_result.plot('epoch', 'baseline', ax=ax1)

# %%
