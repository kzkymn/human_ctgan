# %%
from IPython import get_ipython
from hctgan import HCTGANSynthesizer
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import torch

from util import feedback, print_diff_between_original_and_generated_data

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%[markdown]
# Options

# %%
sigma = 0.01
hctgan_path = './checkpoint/htcgan.pth'
feedback_path = './output/feedbacks.csv'
answer_added_feedback_path = './input/feedbacks_with_answers.csv'
target_colname = 'feedback'

# %%
X, y = load_iris(return_X_y=True, as_frame=True)
df_train = X.copy()
df_train['target'] = y.copy()
df_train

# %%
discrete_columns = [
    'target'
]


def create_seed_tuple(np_seed=333, torch_seed=333):
    return (np.random.RandomState(np_seed), torch.manual_seed(torch_seed))


seed_tuple = create_seed_tuple()

hctgan = HCTGANSynthesizer()
hctgan.random_states = seed_tuple
hctgan.fit(df_train,
           discrete_columns=discrete_columns)
hctgan.random_states = seed_tuple
original_sample = hctgan.sample(10)

# %%
print('================')
print('== Only CTGAN ==')
print('================')
print_diff_between_original_and_generated_data(df_train, hctgan)

# %%
hctgan.random_states = None
hctgan.save(path=hctgan_path)

hctgan.random_states = create_seed_tuple()
hctgan.create_feedback_data_csv(csv_path=feedback_path,
                                n=20,
                                r=5,
                                sigma=sigma,
                                target_colname=target_colname)

# %%
feedback_df = pd.read_csv(answer_added_feedback_path)
feedback_probs = feedback_df[target_colname].values
data_for_feedback_orig = feedback_df.drop(columns=target_colname)
hctgan: HCTGANSynthesizer = HCTGANSynthesizer.load(path=hctgan_path)

hctgan.random_states = create_seed_tuple()
sampled_data_tensor, data_for_feedback, perturbations = hctgan.sample_for_human_evaluation(n=20,
                                                                                           r=5,
                                                                                           sigma=sigma)

np.testing.assert_array_almost_equal(
    data_for_feedback_orig, data_for_feedback)

hctgan.random_states = create_seed_tuple()
hctgan.fit_to_feedback(sampled_data_tensor,
                       feedback_probs,
                       perturbations,
                       sigma=sigma)


# %%
print('=============')
print('=== Final ===')
print('=============')
print_diff_between_original_and_generated_data(
    df_train, hctgan,  show_original_data_info=False)

# %%
trained_sample = hctgan.sample(10)

# %%
original_sample

# %%
trained_sample

# %%
