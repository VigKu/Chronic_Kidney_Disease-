import os
import tensorflow as tf
import pandas as pd

from utils import combine_format, to_tensor, evaluate
from model import CombinedModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# GPU
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# path directories of input data
dir_ind_test = "Datasets/waves_indicators_test.csv"
dir_med_test = "Datasets/waves_drugs_test.csv"
dir_demo_test = "Datasets/data_demo_test.csv"
dir_stage_test = "Datasets/data_stage_test.csv"

# load data
df_ind_test = pd.read_csv(dir_ind_test)
df_med_test = pd.read_csv(dir_med_test)
df_demo_test = pd.read_csv(dir_demo_test, index_col=0)
df_stage_test = pd.read_csv(dir_stage_test)

# get the correct format for test
data_demos_test, data_waves_test, labels_test = combine_format(df_ind_test,
                                                               df_med_test,
                                                               df_demo_test,
                                                               df_stage_test)


# Hyper Params
SEQ_LEN = 35
NUM_DAYS = 700
BATCH_SIZE = 1
# NUM_EPOCHS = 3
# LR = 5e-8

AUTOTUNE = tf.data.experimental.AUTOTUNE

# convert test data to tensors and batchify
ds_test = tf.data.Dataset.from_tensor_slices((data_demos_test,
                                             data_waves_test,
                                             labels_test)).map(to_tensor, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

# model
MyModel = CombinedModel(SEQ_LEN, NUM_DAYS)
MyModel.load_weights("saved_ckpt.tf")

# train
evaluate(ds_test=ds_test, model=MyModel)
