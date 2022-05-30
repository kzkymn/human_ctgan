# %%
from IPython import get_ipython
from hctgan import HCTGANSynthesizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.datasets import load_iris
import torch
from xgboost import XGBClassifier

from util import create_wrong_feedback, print_diff_between_original_and_generated_data

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%[markdown]
# Options

# %%
sigma = 10
hctgan_path = './checkpoint/htcgan.pth'
sampled_data_path = './output/sampled_data.pth'

feedback_function = create_wrong_feedback

# %%
X, y = load_iris(return_X_y=True, as_frame=True)
df_train = X.copy()
df_train['target'] = y.copy()
df_train

# %%
discrete_columns = [
    'target'
]

hctgan = HCTGANSynthesizer()
hctgan.fit(df_train,
           discrete_columns=discrete_columns)
original_sample = hctgan.sample(10)

# %%
print('================')
print('== Only CTGAN ==')
print('================')
print_diff_between_original_and_generated_data(df_train, hctgan)

# %%
for i in range(1000):
    sampled_data_tensor, data_for_feedback, perturbations = hctgan.sample_for_human_evaluation(n=20,
                                                                                               r=5,
                                                                                               sigma=sigma)

    # torch.save(sampled_data_tensor.state_dict(), sampled_data_path)
    # hctgan.save(path=hctgan_path)

    # del hctgan, sampled_data_tensor

    # hctgan = HCTGANSynthesizer.load(path=hctgan_path)
    # sampled_data_tensor = torch.load(sampled_data_path).to(hctgan._device)

    feedback_probs = feedback_function(data_for_feedback)
    hctgan.fit_to_feedback(sampled_data_tensor,
                           feedback_probs,
                           perturbations,
                           sigma=sigma)

    if i != 0 and i % 100 == 0:
        print('==============')
        print(f'=== i {i: 03d} ===')
        print('==============')
        print_diff_between_original_and_generated_data(
            df_train, hctgan, show_original_data_info=False)

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
