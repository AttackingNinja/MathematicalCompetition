import pandas as pd
import numpy as np

# path_train_data = 'data/train_data.xlsx'
# path_train_label = 'data/train_label.xlsx'
# path_test_data = 'data/test_data.xlsx'
# path_test_label = 'data/test_label.xlsx'
# train_data = np.array(pd.read_excel(path_train_data, header=None))
# train_label = np.array(pd.read_excel(path_train_label, header=None))
# test_data = np.array(pd.read_excel(path_test_data, header=None))
# test_label = np.array(pd.read_excel(path_test_label, header=None))
# np.save('train_set', train_data)
# np.save('train_label', train_label)
# np.save('test_set', test_data)
# np.save('test_label', test_label)
path_data = 'data/附件2-睡眠脑电数据.xlsx'
sheet_names = ['清醒期（6）', '快速眼动期（5）', '睡眠I期（4）', '睡眠II期（3）', '深睡眠期（2）']
dataset = []
label = []
for sheet in sheet_names:
    label.append(np.array(pd.read_excel(path_data, sheet_name=sheet, usecols=[0])))
    dataset.append(np.array(pd.read_excel(path_data, sheet_name=sheet, usecols=[1, 2, 3, 4])))
np.save('label', label)
np.save('dataset', dataset)
