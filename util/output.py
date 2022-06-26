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
            print("# original data")
            print(original_data[c].value_counts(normalize=True))

        print('')
        print("# generated data")
        print(sampled_df[c].value_counts(normalize=True))
        print('=====')
