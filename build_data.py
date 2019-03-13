# github has 100MB limit, so split old_kanji.npz into two files
# run this script to recombing the two

import numpy as np

load_data_1 = np.load('data/old_kanji_part1.npz', encoding='latin1')
old_train_category = load_data_1['category']
old_train_data_1 = load_data_1['data']
old_train_label_1 = load_data_1['label']

load_data_2 = np.load('data/old_kanji_part2.npz', encoding='latin1')
old_train_data_2 = load_data_2['data']
old_train_label_2 = load_data_2['label']

old_train_data = np.concatenate([old_train_data_1, old_train_data_2])
old_train_label = np.concatenate([old_train_label_1, old_train_label_2])

np.savez_compressed('data/old_kanji.npz', category=old_train_category, data=old_train_data, label=old_train_label)
