# %%[markdown]
# # Human-CTGAN

# %%
from functools import partial
import warnings

from IPython import get_ipython
from mlxtend.plotting import plot_decision_regions
from sklearn.neighbors import KNeighborsClassifier
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
                  create_feedback_by_knn_with_proba,
                  create_feedback_function_by_rule_base,
                  print_diff_between_original_and_generated_data)
from util.feedback import create_feedback_by_knn_with_proba

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
warnings.filterwarnings('ignore')

# %%[markdown]
# ## Human-CTGAN Settings

# %%

# NOTE: SAMPLE_SIZE_OF_SYNTHESIZED_DATA is sample size of synthesized data used for classification
SAMPLE_SIZE_OF_SYNTHESIZED_DATA = 20000  # for the strict test
# SAMPLE_SIZE_OF_SYNTHESIZED_DATA = 200  # for easy test (but incorrect)

# NOTE: GAN_TRAINING_METHOD can be set two values.
# 'HumanGAN' and 'ActiveLearning'.
# 'ActiveLearning' tends to produce better results
# when the training data have strong biases.
GAN_TRAINING_METHOD = 'ActiveLearning'
# NOTE: You should set USE_PERTURBATION_WHEN_ACTIVE_LEARNING True,
# if there is considered to be a strong bias in the training data.
USE_PERTURBATION_WHEN_ACTIVE_LEARNING = True

# NOTE: You need to perform the CTGAN at sufficient epochs before you run Human-CTGAN.
CTGAN_FEEDBACK_EPOCHS = 3000

# NOTE: Feedback data are those that need to be given a human evaluation in the Human-in-the-Loop.
# If GAN_TRAINING_METHOD is 'HumanGAN' or USE_PERTURBATION_WHEN_ACTIVE_LEARNING is True,
# The actual sample size of the feedback data is given by the following formula.
# actual sample size := SAMPLE_SIZE_OF_FEEDBACK_DATA * PERTURBATION_PER_FEEDBACK_DATUM * 2
# Otherwise, the actual sample size is same as SAMPLE_SIZE_OF_FEEDBACK_DATA.
SAMPLE_SIZE_OF_FEEDBACK_DATA = 100
PERTURBATION_PER_FEEDBACK_DATUM = 4
# NOTE: PERTURBATION_SIGMA is the standard deviation of the perturbation for one feedback data.
# Perturbations are given by a normal distribution.
# Per single feedback datum, two perturbated data are generated.
#
# One is <feedback data> + <perturbation>, another is <feedback data> - <perturbation>.
#
# If there are large biases in the data initially trained on CTGAN,
# the value of sigma should be relatively large.
# However, it should be noted that if the sigma is too large,
# the CTGAN model tends to generate only the largest or smallest
# possible values of the data.
PERTURBATION_SIGMA = 2

HCTGAN_FILE_PATH = './checkpoint/hctgan_test'
FEEDBACK_CSV_PATH = './output/feedbacks_test.csv'
FEEDBACK_COLNAME = 'feedback'
TARGET_COLNAME = 'target'


# %% [markdown]
# ## Test Settings

# %%
# NOTE: BOOTSTRAP_ITER_N is the number of bootstrap iterations to calculate the ROC AUC score.
BOOTSTRAP_ITER_N = 100  # for the strict test
# BOOTSTRAP_ITER_N = 2  # for easy test (but incorrect)

# %%
# NOTE: If any of the explanatory or objective variables are categorical,
# please include their names in this list.
DISCRETE_COLUMNS = [
    TARGET_COLNAME
]

# %% [markdown]
# ## Utility Functions

# %%


def get_feedback_function_by_knn(df, target_colname='target'):
    X = df.drop(columns=target_colname)
    y = df[target_colname]

    knn = KNeighborsClassifier().fit(X, y)
    return partial(create_feedback_by_knn_with_proba, knn=knn)


def get_feedback_function_by_rule_base(original_df, target_colname='target'):
    _tmp_df = original_df.copy()
    _tmp_df = _tmp_df[~((_tmp_df[target_colname] == 0) &
                        (_tmp_df['sepal width (cm)'] < 2.5))]
    max_df = _tmp_df.groupby(TARGET_COLNAME).max().reset_index(drop=False)
    min_df = _tmp_df.groupby(TARGET_COLNAME).min().reset_index(drop=False)

    return partial(create_feedback_function_by_rule_base,
                   max_df=max_df,
                   min_df=min_df,
                   target_colname=target_colname)


def get_feedback_function_by_knn_and_rule_base(original_df, target_colname='target'):
    knn_func = get_feedback_function_by_knn(original_df, target_colname)
    rb_func = get_feedback_function_by_rule_base(original_df, target_colname)

    def hybrid_feedback_function(df, knn_func, rb_func):
        knn_res = knn_func(df)
        rb_res = rb_func(df)
        return knn_res * rb_res

    return partial(hybrid_feedback_function,
                   knn_func=knn_func, rb_func=rb_func)


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
        X, y, test_size=0.05, random_state=42)

    df_train = X_train.copy()
    df_train[target_colname] = y_train
    df_test = X_test.copy()
    df_test[target_colname] = y_test
    df_0 = df_train[df_train[target_colname] == 0]
    df_1_or_2 = df_train[~(df_train[target_colname] == 0)]

    train_criterion_of_df_0 = (df_0['sepal length (cm)'] <= 4.5) & (
        df_0['sepal length (cm)'] <= 5.2)
    train_criterion_of_df_1_or_2 = (df_1_or_2['sepal length (cm)'] >= 5.9) & (
        df_1_or_2['sepal length (cm)'] <= 6.6)

    df_0_train = df_0[train_criterion_of_df_0]
    df_1_or_2_train = df_1_or_2[train_criterion_of_df_1_or_2]

    df_train = pd.concat([df_0_train, df_1_or_2_train])
    df_test = df

    return df_train, df_test


def create_seed_tuple(np_seed=333, torch_seed=333):
    return (np.random.RandomState(np_seed), torch.manual_seed(torch_seed))


def _calc_roc_auc_score(y_true, y_pred_proba, unique_y_labels_num):
    if unique_y_labels_num >= 3:
        auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    else:
        auc_score = roc_auc_score(y_true, y_pred_proba)
    return auc_score


def create_classifier_with_synthed_data(df_train, df_synthed,
                                        target_colname='target'):
    clf = XGBClassifier(eval_metric='logloss')
    if df_synthed is not None:
        tmp_df = pd.concat([df_train, df_synthed])
    else:
        tmp_df = df_train
    tmp_X = tmp_df.drop(columns=target_colname).values
    tmp_y = tmp_df[target_colname].values
    clf.fit(tmp_X, tmp_y)
    return clf


def calc_roc_auc_score(df_train, df_synthed, df_test, unique_y_labels_num,
                       target_colname='target'):
    clf = create_classifier_with_synthed_data(
        df_train=df_train, df_synthed=df_synthed,
        target_colname=target_colname)
    X_test = df_test.drop(columns=target_colname)
    y_test = df_test[target_colname]
    y_pred_proba = clf.predict_proba(X_test)

    auc_score = _calc_roc_auc_score(y_test, y_pred_proba, unique_y_labels_num)
    return auc_score


def calc_conf_interval_of_auc_by_bootstrapping(hctgan,
                                               df_train,
                                               seed_tuple,
                                               sample_size_of_synthesized_data,
                                               target_colname,
                                               iter_n=10,
                                               percentage_of_confidence_interval=0.95):
    res_list = []
    hctgan.random_states = seed_tuple
    for _ in range(iter_n):
        df_synthed = hctgan.sample(sample_size_of_synthesized_data)
        intermediate_roc_auc_score = calc_roc_auc_score(
            df_train, df_synthed, df_test, unique_y_labels_num,
            target_colname=target_colname)
        res_list.append(intermediate_roc_auc_score)

    average_score = np.average(res_list)
    conf_interval_min = np.quantile(a=res_list,
                                    q=1-percentage_of_confidence_interval)
    conf_interval_max = np.quantile(a=res_list,
                                    q=percentage_of_confidence_interval)

    return average_score, conf_interval_min, conf_interval_max


def draw_decision_region(df_train, df_synthed, df_test,
                         target_colname='target',
                         plot_max_num_of_train_data=5000,
                         train_data_alpha=1,
                         skip_drawing_train_data=False,
                         paint_train_data_black=False):
    clf = create_classifier_with_synthed_data(
        df_train=df_train, df_synthed=df_synthed,
        target_colname=target_colname)

    X_test = df_test.drop(columns=target_colname)
    y_test = df_test[target_colname]
    ax_1 = plot_decision_regions(X=X_test.values, y=y_test.values, clf=clf)

    if skip_drawing_train_data:
        return

    # drawing train and synthesized data
    df_train_and_syn = pd.concat([df_train, df_synthed])
    if len(df_train_and_syn) > plot_max_num_of_train_data:
        df_train_and_syn = df_train_and_syn.sample(n=plot_max_num_of_train_data,
                                                   random_state=42)
    colnames = list(df_train_and_syn.columns)
    for target_value in df_train_and_syn[target_colname].unique():
        df_train_and_syn_for_drawing = df_train_and_syn[df_train_and_syn[target_colname] == target_value]

        target_color = 'black'
        graph_label = f'train_data_{target_value}'
        if paint_train_data_black:
            target_color = 'black'
            if target_value == 0:
                graph_label = 'train_data'
            else:
                graph_label = None
        elif target_value == 0:
            target_color = 'tab:blue'
        elif target_value == 1:
            target_color = 'tab:orange'
        elif target_value == 2:
            target_color = 'tab:green'

        df_train_and_syn_for_drawing.plot(colnames[0], colnames[1],
                                          kind='scatter', color=target_color,
                                          alpha=train_data_alpha,
                                          marker='.',
                                          label=graph_label,
                                          ax=ax_1)


def iterate_feedbacks(hctgan,
                      df_train,
                      roc_auc_score_list,
                      roc_auc_score_lower_list,
                      roc_auc_score_upper_list,
                      sample_size_of_synthesized_data,
                      sample_size_of_feedback_data,
                      perturbation_per_feedback_datum,
                      perturbation_sigma,
                      hctgan_file_path,
                      feedback_csv_path,
                      feedback_colname,
                      seed_tuple,
                      target_colname='target',
                      discrete_columns=DISCRETE_COLUMNS,
                      start_n=1,
                      iter_n=20,
                      bootstrap_iter_n=BOOTSTRAP_ITER_N,
                      training_method='HumanGAN',
                      use_perturbation_when_active_learning=USE_PERTURBATION_WHEN_ACTIVE_LEARNING):
    """simulate Human-in-the-Loop processes of Human-CTGAN
    """
    current_sigma = perturbation_sigma
    end_n = start_n + iter_n

    hctgan.save(path=hctgan_file_path)
    hctgan.set_device(torch.device('cpu'))
    copyed_hctgan: HCTGANSynthesizer = HCTGANSynthesizer.load(
        path=hctgan_file_path)

    df_feedback_list = []
    for i in range(start_n, end_n):
        is_current_sigma_halfed = False
        if i != 1 and i % 1 == 0:
            current_sigma /= 2
            is_current_sigma_halfed = True

        copyed_hctgan.save(path=hctgan_file_path)

        # [1ST. PROCESS] create feedback data
        if training_method == 'HumanGAN' or use_perturbation_when_active_learning:
            copyed_hctgan.random_states = seed_tuple
            copyed_hctgan.create_feedback_data_csv(csv_path=feedback_csv_path,
                                                   sample_size_of_feedback_data=sample_size_of_feedback_data,
                                                   perturbation_per_feedback_datum=perturbation_per_feedback_datum,
                                                   perturbation_sigma=current_sigma,
                                                   target_colname=feedback_colname)
            # NOTE: Originally, a CSV file would be read here, with human feedback
            # of the data evaluation values (i.e. probability of authenticity).
            # However, since that is not available now, the evaluation values
            # obtained by the classifier (e.g. KNN) are given afterwards.
            feedback_df = pd.read_csv(feedback_csv_path)
            data_for_feedback_orig = feedback_df.drop(columns=feedback_colname)
        elif training_method == 'ActiveLearning':
            copyed_hctgan.random_states = seed_tuple
            feedback_df = copyed_hctgan.sample(n=sample_size_of_feedback_data)
            data_for_feedback_orig = feedback_df.copy()
        else:
            raise ValueError('Unsupported training_mode:', training_method)

        # NOTE: Adding the data evaluation values described above.
        feedback_probs = feedback_function(data_for_feedback_orig)
        feedback_df[feedback_colname] = feedback_probs
        feedback_df.to_csv(
            f'./output/debug_feedbacks_iter_{i}.csv', index=False)
        if training_method == 'ActiveLearning':
            feedback_df = feedback_df[feedback_df[feedback_colname] >= 0.6]
            df_feedback_list.append(feedback_df)

        # [2ND. PROCESS] create feedback datarefit using feedback data
        # NOTE: This process is usually done some days after the feedback generation.
        # To simulate this, the Human-CTGAN model is read back from the file
        # once it has been saved.
        copyed_hctgan: HCTGANSynthesizer = HCTGANSynthesizer.load(
            path=hctgan_file_path)
        if is_current_sigma_halfed:
            copyed_hctgan.perturbation_sigma = current_sigma

        if training_method == 'HumanGAN':
            # NOTE: fit_to_feedback method performs a back propagation logic with the feedback data.
            copyed_hctgan.random_states = seed_tuple
            copyed_hctgan.fit_to_feedback(data_for_feedback_orig,
                                          feedback_probs)
        elif training_method == 'ActiveLearning':
            # NOTE: HCTGAN refits the CTGAN model with the feedback data.
            copyed_hctgan.fit(pd.concat([df_train,
                              pd.concat(df_feedback_list).drop(columns=feedback_colname)]),
                              discrete_columns=discrete_columns)

        print(f'=== i {i: 03d} ===')
        copyed_hctgan.random_states = seed_tuple
        intermediate_roc_auc_score, score_lower, score_upper = calc_conf_interval_of_auc_by_bootstrapping(copyed_hctgan,
                                                                                                          df_train,
                                                                                                          seed_tuple,
                                                                                                          sample_size_of_synthesized_data,
                                                                                                          target_colname,
                                                                                                          iter_n=bootstrap_iter_n,
                                                                                                          percentage_of_confidence_interval=0.95)
        print('roc_auc_score:', intermediate_roc_auc_score)
        roc_auc_score_list.append(intermediate_roc_auc_score)
        roc_auc_score_lower_list.append(score_lower)
        roc_auc_score_upper_list.append(score_upper)

    return copyed_hctgan


# %% [markdown]
# ## Main Logic

# %%
df = get_iris_dataframe(target_colname=TARGET_COLNAME)
df_train, df_test = prepare_train_and_test(df,
                                           target_colname=TARGET_COLNAME)

# %%
df_train.describe()

# %%
df_test.describe()


# %% [markdown]
# ### Initial CTGAN Training (with no human feedbacks)

# %%
seed_tuple = create_seed_tuple()

hctgan = HCTGANSynthesizer(sample_size_of_feedback_data=SAMPLE_SIZE_OF_FEEDBACK_DATA,
                           perturbation_per_feedback_datum=PERTURBATION_PER_FEEDBACK_DATUM,
                           perturbation_sigma=PERTURBATION_SIGMA,
                           epochs=CTGAN_FEEDBACK_EPOCHS)
hctgan.random_states = seed_tuple
hctgan.fit(df_train,
           discrete_columns=DISCRETE_COLUMNS)
hctgan.random_states = seed_tuple
first_synthed_df = hctgan.sample(SAMPLE_SIZE_OF_SYNTHESIZED_DATA)

# %%
unique_y_labels_num = len(df[TARGET_COLNAME].unique())

# %%
print('================================')
print('== Training only original data==')
print('================================')
hctgan.random_states = seed_tuple
print_diff_between_original_and_generated_data(df_train, hctgan)
original_roc_auc_score = calc_roc_auc_score(
    df_train, None, df_test, unique_y_labels_num,
    target_colname=TARGET_COLNAME)

# %% [markdown]
# Because the classification model trained highly biased data,
# The ROC AUC score and the decision regions are not very correct.

# %%
print('roc_auc_score:', original_roc_auc_score)

# %%
draw_decision_region(df_train, None, df_test, target_colname=TARGET_COLNAME,
                     paint_train_data_black=True)

# %%
print('================')
print('== Only CTGAN ==')
print('================')
hctgan.random_states = seed_tuple
print_diff_between_original_and_generated_data(df_train, hctgan)
original_and_ctgan_roc_auc_score, score_lower, score_upper = calc_conf_interval_of_auc_by_bootstrapping(hctgan,
                                                                                                        df_train,
                                                                                                        seed_tuple,
                                                                                                        sample_size_of_synthesized_data=SAMPLE_SIZE_OF_FEEDBACK_DATA,
                                                                                                        target_colname=TARGET_COLNAME,
                                                                                                        iter_n=BOOTSTRAP_ITER_N,
                                                                                                        percentage_of_confidence_interval=0.95)

# %% [markdown]
# Even adding synthesized data to train data, nothing seems to be corrected.

# %%
print('roc_auc_score:', original_and_ctgan_roc_auc_score)

# %%
roc_auc_score_list = []
roc_auc_score_lower_list = []
roc_auc_score_upper_list = []

roc_auc_score_list.append(original_and_ctgan_roc_auc_score)
roc_auc_score_lower_list.append(score_lower)
roc_auc_score_upper_list.append(score_upper)

# %% [markdown]
# Due to the biased data,
# the distribution of synthesized data by CTGAN is also incorrect.

# %%
draw_decision_region(df_train, first_synthed_df, df_test,
                     target_colname=TARGET_COLNAME)

# %% [markdown]
# ### Feedback Function Settings
# In practice, it is necessary for humans to provide feedback
# on the synthesized data.
# Feedback values must take a value between 0 and 1.
# The more data seem to be authentic,
# the closer to 1 you should set the values.
#
# But in this sample, feedbacks are given by a function instead of a human.

# %%
# NOTE: Functions that give good feedbacks.
# feedback_function = get_feedback_function_by_knn(df)
# feedback_function = get_feedback_function_by_rule_base(df)
feedback_function = get_feedback_function_by_knn_and_rule_base(df)

# NOTE: Functions that give **BAD** feedbacks. (for operation checking)
# feedback_function = create_random_feedback
# feedback_function = create_wrong_feedback

# %% [markdown]
# ### Human-CTGAN Training

# %%
hctgan = iterate_feedbacks(hctgan,
                           df_train,
                           roc_auc_score_list=roc_auc_score_list,
                           roc_auc_score_lower_list=roc_auc_score_lower_list,
                           roc_auc_score_upper_list=roc_auc_score_upper_list,
                           sample_size_of_synthesized_data=SAMPLE_SIZE_OF_SYNTHESIZED_DATA,
                           sample_size_of_feedback_data=SAMPLE_SIZE_OF_FEEDBACK_DATA,
                           perturbation_per_feedback_datum=PERTURBATION_PER_FEEDBACK_DATUM,
                           perturbation_sigma=PERTURBATION_SIGMA,
                           hctgan_file_path=HCTGAN_FILE_PATH,
                           feedback_csv_path=FEEDBACK_CSV_PATH,
                           feedback_colname=FEEDBACK_COLNAME,
                           seed_tuple=seed_tuple,
                           target_colname=TARGET_COLNAME,
                           discrete_columns=DISCRETE_COLUMNS,
                           iter_n=2, start_n=1,
                           training_method=GAN_TRAINING_METHOD,
                           bootstrap_iter_n=BOOTSTRAP_ITER_N)

# %%
print('===============================')
print('=== Result of the feedbacks ===')
print('===============================')
hctgan.random_states = seed_tuple
print_diff_between_original_and_generated_data(
    df_train, hctgan)
print('original roc_auc_score:', roc_auc_score_list[0])
print('current roc_auc_score:', roc_auc_score_list[-1])

# %%
first_synthed_df.describe()

# %%
hctgan.random_states = seed_tuple
last_synthed_df = hctgan.sample(SAMPLE_SIZE_OF_SYNTHESIZED_DATA)
last_synthed_df.describe()

# %%
df.describe()

# %% [markdown]
# Thanks to the feedbacks by the function, the distributions of the synthesized data
# are much closer to the true one.
# However, the feedbacks are not perfect because it cannot correctly evaluate data
# outside of the original value range.
# It is why some data are located in odd positions.
#
# So, even when humans give feedbacks, you should consider
# how the feedbacks you give should be modified
# based on the results of the active learnings or the back propagation processes.

# %%
draw_decision_region(df_train, last_synthed_df, df_test,
                     target_colname=TARGET_COLNAME)

# %% [markdown]
# ### ROC AUC score in each epoch
#
# Epoch 0 is the result of the classification model learning the training data and first CTGAN sythesized data.
#
# The blue band on the graph represents the bootstrap confidence interval (95% CI) for AUC ROC scores.

# %%
df_result = pd.DataFrame()
x_list = list(range(len(roc_auc_score_list)))
df_result['epoch'] = x_list
df_result['roc_auc_score'] = roc_auc_score_list
df_result['baseline'] = roc_auc_score_list[0]
ax1 = df_result.plot('epoch', 'roc_auc_score')
df_result.plot('epoch', 'roc_auc_score', kind='scatter', color='b', ax=ax1)
df_result.plot('epoch', 'baseline', ax=ax1)
ax1.fill_between(
    x_list,
    roc_auc_score_lower_list,
    roc_auc_score_upper_list,
    alpha=0.3,
)

# %%
