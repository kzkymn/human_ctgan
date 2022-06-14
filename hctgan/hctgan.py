# Some methods in this code are derived from sdv-dev/CTGAN. Below is the license of it.
#
# MIT License
#
# Copyright (c) 2019, MIT Data To AI Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from copy import deepcopy
import pickle
from ctgan import CTGANSynthesizer
from ctgan.synthesizers.base import random_state

import numpy as np
import torch
from torch import optim


class InnerConditions():
    def __init__(self,
                 sample_size_of_feedback_data,
                 perturbation_per_feedback_datum,
                 perturbation_sigma,
                 condition_column,
                 condition_value):
        self.sample_size_of_feedback_data = sample_size_of_feedback_data
        self.perturbation_per_feedback_datum = perturbation_per_feedback_datum
        self.perturbation_sigma = perturbation_sigma
        self.condition_column = condition_column
        self.condition_value = condition_value


class HCTGANSynthesizer(CTGANSynthesizer):
    """CTGAN with HumanGAN
    """

    def __init__(self,
                 sample_size_of_feedback_data,
                 perturbation_per_feedback_datum=5,
                 perturbation_sigma=0.01,
                 condition_column=None,
                 condition_value=None,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True,
                 ):
        super().__init__(embedding_dim=embedding_dim, generator_dim=generator_dim,
                         discriminator_dim=discriminator_dim,
                         generator_lr=generator_lr, generator_decay=generator_decay,
                         discriminator_lr=discriminator_lr,
                         discriminator_decay=discriminator_decay,
                         batch_size=batch_size, discriminator_steps=discriminator_steps,
                         log_frequency=log_frequency, verbose=verbose,
                         epochs=epochs, pac=pac, cuda=cuda)
        self._set_inner_conditions(sample_size_of_feedback_data=sample_size_of_feedback_data,
                                   perturbation_per_feedback_datum=perturbation_per_feedback_datum,
                                   perturbation_sigma=perturbation_sigma,
                                   condition_column=condition_column,
                                   condition_value=condition_value)

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        super().fit(train_data=train_data,
                    discrete_columns=discrete_columns,
                    epochs=epochs)
        return self

    def _set_inner_conditions(self,
                              sample_size_of_feedback_data=None,
                              perturbation_per_feedback_datum=None,
                              perturbation_sigma=None,
                              condition_column=None,
                              condition_value=None):
        self.sample_size_of_feedback_data = sample_size_of_feedback_data
        self.perturbation_per_feedback_datum = perturbation_per_feedback_datum
        self.perturbation_sigma = perturbation_sigma
        self.condition_column = condition_column
        self.condition_value = condition_value

    @random_state
    def fit_to_feedback(self,
                        data_for_feedback_orig,
                        feedback_probs
                        ):

        x, data_for_feedback, perturbations = self._sample_for_human_evaluation()

        np.testing.assert_array_almost_equal(
            data_for_feedback_orig, data_for_feedback)

        N = x.shape[0]
        feedback_sample_size = feedback_probs.shape[0]

        R = int(feedback_sample_size / (N * 2))

        reshaped_feedback_probs = feedback_probs.reshape((N, R, 2))

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        grad_list = []
        for n in range(N):
            sum_list = []
            for r in range(R):
                perturbation_xnr = perturbations[n, r, :]
                D_xnr = reshaped_feedback_probs[n, r, 0] - \
                    reshaped_feedback_probs[n, r, 1]
                sum_list.append(D_xnr * perturbation_xnr)
            grad_of_xn = np.sum(sum_list, axis=0) / \
                (2*self.perturbation_sigma**2*R)
            grad_list.append(grad_of_xn)

        x_for_bw = x.to(self._device)
        grad = torch.Tensor(np.vstack(grad_list)).to(self._device)

        optimizerG.zero_grad()
        x_for_bw.backward(grad)
        optimizerG.step()

        grad.detach().cpu()
        x_for_bw.detach().cpu()

        return self

    @random_state
    def _sample_as_tensor(self, n, condition_column=None, condition_value=None):
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(
                    self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            data.append(fake)

        data_tensor = torch.cat(data, dim=0)
        data_tensor = data_tensor[:n]
        data_tensor = data_tensor.cpu()

        return data_tensor

    @random_state
    def _sample_for_human_evaluation(self):

        raw_data_tensor = self._sample_as_tensor(
            self.sample_size_of_feedback_data, condition_column=self.condition_column,
            condition_value=self.condition_value)

        # TODO: クラスタリングして、その中心の最近傍の摂動だけ返すオプションを実装
        # その時は返すクラスタ中心の数を指定する引数も追加

        # adding perturbations
        result_vector_list = []
        mn_perturbation_list = []
        for _, row in enumerate(raw_data_tensor):
            for _ in range(self.perturbation_per_feedback_datum):
                col_num = raw_data_tensor.shape[1]
                mn_distribution_vector = np.random.normal(
                    loc=0, scale=self.perturbation_sigma, size=col_num)

                new_row_added_delta = row.detach().numpy()
                new_row_subtracted_delta = deepcopy(new_row_added_delta)

                new_row_added_delta += mn_distribution_vector
                new_row_subtracted_delta -= mn_distribution_vector

                result_vector_list.append(new_row_added_delta)
                result_vector_list.append(new_row_subtracted_delta)
                mn_perturbation_list.append(mn_distribution_vector)

        fake = torch.Tensor(np.vstack(result_vector_list))
        fakeact = self._apply_activate(fake)
        data = fakeact.numpy()
        perturbations = np.vstack(mn_perturbation_list)
        perturbations = perturbations.reshape(
            (self.sample_size_of_feedback_data, self.perturbation_per_feedback_datum, perturbations.shape[1]))

        return raw_data_tensor, self._transformer.inverse_transform(data), perturbations

    @staticmethod
    def _save_data_for_feedback(data_for_feedback,
                                csv_path,
                                target_colname='feedback'):
        res_df = data_for_feedback.copy()
        res_df[target_colname] = ""
        res_df.to_csv(csv_path, encoding='utf-8', index=False, header=True)

    @random_state
    def create_feedback_data_csv(self,
                                 csv_path,
                                 sample_size_of_feedback_data,
                                 perturbation_per_feedback_datum=5,
                                 perturbation_sigma=0.01,
                                 condition_column=None,
                                 condition_value=None,
                                 target_colname='feedback'):

        self._set_inner_conditions(sample_size_of_feedback_data=sample_size_of_feedback_data,
                                   perturbation_per_feedback_datum=perturbation_per_feedback_datum,
                                   perturbation_sigma=perturbation_sigma,
                                   condition_column=condition_column,
                                   condition_value=condition_value)
        _, data_for_feedback, _ = self._sample_for_human_evaluation()

        self._save_data_for_feedback(data_for_feedback=data_for_feedback,
                                     csv_path=csv_path,
                                     target_colname=target_colname)

    def save_inner_conditions(self, path):
        ic = InnerConditions(sample_size_of_feedback_data=self.sample_size_of_feedback_data,
                             perturbation_per_feedback_datum=self.perturbation_per_feedback_datum,
                             perturbation_sigma=self.perturbation_sigma,
                             condition_column=self.condition_column,
                             condition_value=self.condition_value)
        with open(path, 'wb') as f:
            pickle.dump(ic, f)

    def save(self, path):
        """Save the model in the passed `path`."""
        device_backup = self._device
        self.set_device(torch.device('cpu'))
        torch.save(self, path+'.pth')
        self.save_inner_conditions(path=path+'_inner_conditions.pickle')
        self.set_device(device_backup)

    @classmethod
    def load(cls, path):
        """Load the model stored in the passed `path`."""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model: HCTGANSynthesizer = torch.load(path+'.pth')
        with open(path+'_inner_conditions.pickle', 'rb') as f:
            ic: InnerConditions = pickle.load(f)
            model._set_inner_conditions(sample_size_of_feedback_data=ic.sample_size_of_feedback_data,
                                        perturbation_per_feedback_datum=ic.perturbation_per_feedback_datum,
                                        perturbation_sigma=ic.perturbation_sigma,
                                        condition_column=ic.condition_column,
                                        condition_value=ic.condition_value
                                        )
        model.set_device(device)
        return model
