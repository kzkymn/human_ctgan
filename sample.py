# %%

from IPython import get_ipython
from hctgan import HCTGANSynthesizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.datasets import load_iris
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%


# from ctgan import CTGANSynthesizer

# %%
X, y = load_iris(return_X_y=True, as_frame=True)
df = X.copy()
df['target'] = y.copy()
df

# %%
df0_train, df_test = train_test_split(df,
                                      test_size=0.2,
                                      random_state=0,
                                      stratify=df['target'])

# %%
discrete_columns = [
    'target'
]

hctgan0 = HCTGANSynthesizer()
hctgan0.fit(df0_train,
            discrete_columns=discrete_columns,
            epochs=10)

# %%


def print_diff_between_original_and_generated_data(df0_train,
                                                   hctgan0,
                                                   n_samples=10000,
                                                   target_colname='target'):
    sampled_df = hctgan0.sample(n_samples)

    for c in df0_train.columns:
        if target_colname is not None and c != target_colname:
            continue
        print(c)
        print("original data")
        print(df0_train[c].value_counts(normalize=True))

        print("generated data")
        print(sampled_df[c].value_counts(normalize=True))
        print('=====')


print_diff_between_original_and_generated_data(df0_train, hctgan0)

# %%


def create_random_feedback(data_for_feedback):
    return np.random.rand(len(data_for_feedback))


for _ in range(10):
    sampled_data_tensor, data_for_feedback, perturbations = hctgan0.sample_for_human_evaluation(n=5,
                                                                                                r=5)
    feedback_probs = create_random_feedback(data_for_feedback)
    hctgan0.fit_to_feedback(sampled_data_tensor,
                            feedback_probs,
                            perturbations)
    print_diff_between_original_and_generated_data(df0_train, hctgan0)

# %%
