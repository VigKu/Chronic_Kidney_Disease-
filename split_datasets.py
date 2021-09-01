import csv
import os
import pandas as pd
import numpy as np
from utils import train_val_test_split_indices

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# path directories of input data
dir_ind = "waves_indicators.csv"
dir_med = "waves_meds_prev.csv"
dir_demo = "data_demo.csv"
dir_stage = "dataScienceTask/T_stage.csv"

# load data
df_ind = pd.read_csv(dir_ind)
df_med = pd.read_csv(dir_med)
df_demo = pd.read_csv(dir_demo)
df_stage = pd.read_csv(dir_stage)

# train val test split for indices
ix = np.arange(300)
y = df_stage['Stage_Progress'].values
ix_train, ix_val, ix_test, y_train, y_val, y_test = train_val_test_split_indices(ix, y)

# train
df_ind_train = df_ind[df_ind['id'].isin(ix_train)]
df_med_train = df_med[df_med['id'].isin(ix_train)]
df_demo_train = df_demo[df_demo['id'].isin(ix_train)]
df_stage_train = df_stage[df_stage['id'].isin(ix_train)]

# val
df_ind_val = df_ind[df_ind['id'].isin(ix_val)]
df_med_val = df_med[df_med['id'].isin(ix_val)]
df_demo_val = df_demo[df_demo['id'].isin(ix_val)]
df_stage_val = df_stage[df_stage['id'].isin(ix_val)]

# test
df_ind_test = df_ind[df_ind['id'].isin(ix_test)]
df_med_test = df_med[df_med['id'].isin(ix_test)]
df_demo_test = df_demo[df_demo['id'].isin(ix_test)]
df_stage_test = df_stage[df_stage['id'].isin(ix_test)]

## TRAIN SET
# save the waves in csv
days = np.arange(700)
header = ['id', 'indicator']
header.extend(days)
with open('Datasets/waves_indicators_train.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_ind_train.values.tolist())
file.close()

# save the waves in csv
days = np.arange(700)
header = ['id', ' drug']
header.extend(days)
with open('Datasets/waves_drugs_train.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_med_train.values.tolist())
file.close()

# save the demo in csv
header = df_demo.columns.to_list()

with open('Datasets/data_demo_train.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_demo_train.values.tolist())
file.close()

# save the stage in csv
header = df_stage.columns.to_list()

with open('Datasets/data_stage_train.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_stage_train.values.tolist())
file.close()
# -------------------------------------------------------------------------

### VAL SET
# save the waves in csv
days = np.arange(700)
header = ['id', 'indicator']
header.extend(days)
with open('Datasets/waves_indicators_val.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_ind_val.values.tolist())
file.close()

# save the waves in csv
days = np.arange(700)
header = ['id', ' drug']
header.extend(days)
with open('Datasets/waves_drugs_val.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_med_val.values.tolist())
file.close()

# save the demo in csv
header = df_demo.columns.to_list()

with open('Datasets/data_demo_val.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_demo_val.values.tolist())
file.close()

# save the stage in csv
header = df_stage.columns.to_list()

with open('Datasets/data_stage_val.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_stage_val.values.tolist())
file.close()
# -------------------------------------------------------------------------

### TEST SET
# save the waves in csv
days = np.arange(700)
header = ['id', 'indicator']
header.extend(days)
with open('Datasets/waves_indicators_test.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_ind_test.values.tolist())
file.close()

# save the waves in csv
days = np.arange(700)
header = ['id', ' drug']
header.extend(days)
with open('Datasets/waves_drugs_test.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_med_test.values.tolist())
file.close()

# save the demo in csv
header = df_demo.columns.to_list()

with open('Datasets/data_demo_test.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_demo_test.values.tolist())
file.close()

# save the stage in csv
header = df_stage.columns.to_list()

with open('Datasets/data_stage_test.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(df_stage_test.values.tolist())
file.close()
