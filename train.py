import os
import tensorflow as tf
import pandas as pd

from utils import combine_format, to_tensor, train
from model import CombinedModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# GPU
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# path directories of input data
dir_ind_train = "Datasets/waves_indicators_train.csv"
dir_med_train = "Datasets/waves_drugs_train.csv"
dir_demo_train = "Datasets/data_demo_train.csv"
dir_stage_train = "Datasets/data_stage_train.csv"

dir_ind_val = "Datasets/waves_indicators_val.csv"
dir_med_val = "Datasets/waves_drugs_val.csv"
dir_demo_val = "Datasets/data_demo_val.csv"
dir_stage_val = "Datasets/data_stage_val.csv"

# load data
df_ind_train = pd.read_csv(dir_ind_train)
df_med_train = pd.read_csv(dir_med_train)
df_demo_train = pd.read_csv(dir_demo_train, index_col=0)
df_stage_train = pd.read_csv(dir_stage_train)

df_ind_val = pd.read_csv(dir_ind_val)
df_med_val = pd.read_csv(dir_med_val)
df_demo_val = pd.read_csv(dir_demo_val, index_col=0)
df_stage_val = pd.read_csv(dir_stage_val)

# get the correct format for train
data_demos_train, data_waves_train, labels_train = combine_format(df_ind_train,
                                                                  df_med_train,
                                                                  df_demo_train,
                                                                  df_stage_train)

# get the correct format for val
data_demos_val, data_waves_val, labels_val = combine_format(df_ind_val,
                                                            df_med_val,
                                                            df_demo_val,
                                                            df_stage_val)

# Hyper Params
SEQ_LEN = 35
NUM_DAYS = 700
BATCH_SIZE = 1
NUM_EPOCHS = 10
LR = 3e-4

AUTOTUNE = tf.data.experimental.AUTOTUNE
# convert train data to tensors and batchify
ds_train = tf.data.Dataset.from_tensor_slices((data_demos_train,
                                               data_waves_train,
                                               labels_train)).map(to_tensor, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.shuffle(len(ds_train), seed=5, reshuffle_each_iteration=True)
# ds_train = ds_train.cache()
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# convert val data to tensors and batchify
ds_val = tf.data.Dataset.from_tensor_slices((data_demos_val,
                                             data_waves_val,
                                             labels_val)).map(to_tensor, num_parallel_calls=AUTOTUNE)
# ds_val = ds_val.cache()
ds_val = ds_val.batch(BATCH_SIZE)
ds_val = ds_val.prefetch(AUTOTUNE)

# model
MyModel = CombinedModel(SEQ_LEN, NUM_DAYS)

# train
train(ds_train=ds_train,
      ds_val=ds_val,
      model=MyModel,
      num_epochs=NUM_EPOCHS,
      lr=LR)
