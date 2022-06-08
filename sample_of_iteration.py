# %%
from functools import partial
import warnings

from IPython import get_ipython
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
HCTGAN_FILE_PATH = './checkpoint/htcgan_test.pth'
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
    tmp_X = tmp_df.drop(columns=target_colname)
    tmp_y = tmp_df[target_colname]
    clf.fit(tmp_X, tmp_y)
    return clf


def calc_roc_using_the_result_of_fitting_and_evaluating_model(df_train, df_synthed, df_test, unique_y_labels_num,
                                                              target_colname='target'):
    X_test = df_test.drop(columns=target_colname)
    y_test = df_test[target_colname]
    clf = create_classifier_with_synthed_data(
        df_train, df_synthed, target_colname=target_colname)
    y_pred_proba = clf.predict_proba(X_test)
    return print_roc_auc_score(y_test, y_pred_proba, unique_y_labels_num)


# %%
seed_tuple = create_seed_tuple()

hctgan = HCTGANSynthesizer(epochs=3000)
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
roc_auc_score_list.append(original_roc_auc_score)

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
feedback_function = get_feedback_function_by_one_class_svm(df)
# feedback_function = create_random_feedback
# feedback_function = create_wrong_feedback

# feedback_function = get_feedback_function_by_knn(df)

# %%
current_sigma = PERTURBATION_SIGMA

for i in range(1, 21):
    hctgan.random_states = None
    hctgan.save(path=HCTGAN_FILE_PATH)

    hctgan.random_states = seed_tuple
    hctgan.create_feedback_data_csv(csv_path=FEEDBACK_CSV_PATH,
                                    n=SAMPLE_SIZE_OF_FEEDBACK_DATA,
                                    r=PERTURBATION_PER_FEEDBACK_DATUM,
                                    sigma=current_sigma,
                                    target_colname=FEEDBACK_COLNAME)

    feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)
    data_for_feedback_orig = feedback_df.drop(columns=FEEDBACK_COLNAME)
    feedback_probs = feedback_function(data_for_feedback_orig)

    # hctgan: HCTGANSynthesizer = HCTGANSynthesizer.load(path=hctgan_path)

    hctgan.random_states = seed_tuple
    sampled_data_tensor, data_for_feedback, perturbations = hctgan.sample_for_human_evaluation(n=SAMPLE_SIZE_OF_FEEDBACK_DATA,
                                                                                               r=PERTURBATION_PER_FEEDBACK_DATUM,
                                                                                               sigma=current_sigma)

    np.testing.assert_array_almost_equal(
        data_for_feedback_orig, data_for_feedback)

    hctgan.random_states = seed_tuple
    hctgan.fit_to_feedback(sampled_data_tensor,
                           feedback_probs,
                           perturbations,
                           sigma=current_sigma)

    print(f'=== i {i: 03d} ===')
    hctgan.random_states = seed_tuple
    synthed_df = hctgan.sample(SAMPLE_SIZE_OF_SYNTHESIZED_DATA)
    hctgan.random_states = seed_tuple
    intermediate_roc_auc_score = calc_roc_using_the_result_of_fitting_and_evaluating_model(
        df_train, synthed_df, df_test, unique_y_labels_num,
        target_colname=TARGET_COLNAME)
    roc_auc_score_list.append(intermediate_roc_auc_score)

    if i != 0 and i % 5 == 0:
        current_sigma /= 2

# %%
print('=================================')
print('=== Result of first 20 epochs ===')
print('=================================')
hctgan.random_states = seed_tuple
print_diff_between_original_and_generated_data(
    df_train, hctgan)
print('original roc_auc_score:', roc_auc_score_list[0])
print('current roc_auc_score:', roc_auc_score_list[-1])

# %%
first_synthed_df.describe()

# %%
intermediate_synthed_df = hctgan.sample(SAMPLE_SIZE_OF_SYNTHESIZED_DATA)
intermediate_synthed_df.describe()

# %%
df.describe()

# %%
for i in range(21, 100):
    hctgan.random_states = None
    hctgan.save(path=HCTGAN_FILE_PATH)

    hctgan.random_states = seed_tuple
    hctgan.create_feedback_data_csv(csv_path=FEEDBACK_CSV_PATH,
                                    n=SAMPLE_SIZE_OF_FEEDBACK_DATA,
                                    r=PERTURBATION_PER_FEEDBACK_DATUM,
                                    sigma=PERTURBATION_SIGMA,
                                    target_colname=FEEDBACK_COLNAME)

    feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)
    data_for_feedback_orig = feedback_df.drop(columns=FEEDBACK_COLNAME)
    feedback_probs = feedback_function(data_for_feedback_orig)

    hctgan: HCTGANSynthesizer = HCTGANSynthesizer.load(path=HCTGAN_FILE_PATH)

    hctgan.random_states = seed_tuple
    sampled_data_tensor, data_for_feedback, perturbations = hctgan.sample_for_human_evaluation(n=SAMPLE_SIZE_OF_FEEDBACK_DATA,
                                                                                               r=PERTURBATION_PER_FEEDBACK_DATUM,
                                                                                               sigma=PERTURBATION_SIGMA)

    np.testing.assert_array_almost_equal(
        data_for_feedback_orig, data_for_feedback)

    hctgan.random_states = seed_tuple
    hctgan.fit_to_feedback(sampled_data_tensor,
                           feedback_probs,
                           perturbations,
                           sigma=PERTURBATION_SIGMA)

    print(f'=== i {i: 03d} ===')
    hctgan.random_states = seed_tuple
    synthed_df = hctgan.sample(SAMPLE_SIZE_OF_SYNTHESIZED_DATA)
    hctgan.random_states = seed_tuple
    # print_diff_between_original_and_generated_data(
    #     df_train, hctgan, show_original_data_info=False)
    intermediate_roc_auc_score = calc_roc_using_the_result_of_fitting_and_evaluating_model(
        df_train, synthed_df, df_test, unique_y_labels_num)
    roc_auc_score_list.append(intermediate_roc_auc_score)

print('=============')
print('=== Final ===')
print('=============')
hctgan.random_states = seed_tuple
print_diff_between_original_and_generated_data(
    df_train, hctgan)

# %%
print('original roc_auc_score:', roc_auc_score_list[0])
print('final roc_auc_score:', roc_auc_score_list[-1])

# %%
first_synthed_df.describe()

# %%
final_synthed_df = hctgan.sample(SAMPLE_SIZE_OF_SYNTHESIZED_DATA)
final_synthed_df.describe()

# %%
df.describe()

# %% [markdown]
# roc auc score in each epoch

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
