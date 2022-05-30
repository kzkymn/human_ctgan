def print_diff_between_original_and_generated_data(original_data,
                                                   ctgan,
                                                   n_samples=10000,
                                                   target_colname='target',
                                                   show_original_data_info=True):
    sampled_df = ctgan.sample(n_samples)

    for c in original_data.columns:
        if target_colname is not None and c != target_colname:
            continue
        print(f'【{c}】')
        if show_original_data_info:
            print("original data")
            print(original_data[c].value_counts(normalize=True))

        print("generated data")
        print(sampled_df[c].value_counts(normalize=True))
        print('=====')


def save_data_for_feedback(data_for_feedback,
                           csv_path,
                           target_colname='feedback'):
    res_df = data_for_feedback.copy()
    res_df[target_colname] = ""
    res_df.to_csv(csv_path, encoding='utf-8', index=False, header=True)
