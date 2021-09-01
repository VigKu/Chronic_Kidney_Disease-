import csv

import numpy as np
import pandas as pd

from utils import generate_indicator_waves

thresh_sample = 30  # only include if more than 10% of sample size
indicator_list = ['creatinine', 'DBP', 'glucose', 'HGB', 'Idl', 'SBP']

# read csv
path_creatinine = 'dataScienceTask/T_creatinine.csv'
path_DBP = 'dataScienceTask/T_DBP.csv'
path_glucose = 'dataScienceTask/T_glucose.csv'
path_HGB = 'dataScienceTask/T_HGB.csv'
path_Idl = 'dataScienceTask/T_ldl.csv'
path_SBP = 'dataScienceTask/T_SBP.csv'

df_creatinine = pd.read_csv(path_creatinine)
df_DBP = pd.read_csv(path_DBP)
df_glucose = pd.read_csv(path_glucose)
df_HGB = pd.read_csv(path_HGB)
df_Idl = pd.read_csv(path_Idl)
df_SBP = pd.read_csv(path_SBP)
df_stage = pd.read_csv('dataScienceTask/T_stage.csv')

mean_list = [df_creatinine.value.mean(),
             df_DBP.value.mean(),
             df_glucose.value.mean(),
             df_HGB.value.mean(),
             df_Idl.value.mean(),
             df_SBP.value.mean()]
std_list = [df_creatinine.value.std(),
            df_DBP.value.std(),
            df_glucose.value.std(),
            df_HGB.value.std(),
            df_Idl.value.std(),
            df_SBP.value.std()]

# get mean and std values
dict_mean_ind = {}
dict_std_ind = {}
for i, ind in enumerate(indicator_list):
    dict_mean_ind.update({ind:mean_list[i]})
    dict_std_ind.update({ind:std_list[i]})


# rename the columns
df_creatinine.rename(columns={'value': 'value_creatinine', 'time': 'time_creatinine'}, inplace=True)
df_DBP.rename(columns={'value': 'value_DBP', 'time': 'time_DBP'}, inplace=True)
df_glucose.rename(columns={'value': 'value_glucose', 'time': 'time_glucose'}, inplace=True)
df_HGB.rename(columns={'value': 'value_HGB', 'time': 'time_HGB'}, inplace=True)
df_Idl.rename(columns={'value': 'value_Idl', 'time': 'time_Idl'}, inplace=True)
df_SBP.rename(columns={'value': 'value_SBP', 'time': 'time_SBP'}, inplace=True)

# remove rows for df_HBG if time > 699 days to standardize with other indicators
df_HGB_red = df_HGB[df_HGB['time_HGB'] < 700].copy()

# Condense the values into a list for each patient id
df_agg_creatinine = df_creatinine.groupby(['id'], as_index=False)[['value_creatinine', 'time_creatinine']].agg(list)
df_agg_DBP = df_DBP.groupby(['id'], as_index=False)[['value_DBP', 'time_DBP']].agg(list)
df_agg_glucose = df_glucose.groupby(['id'], as_index=False)[['value_glucose', 'time_glucose']].agg(list)
df_agg_HGB_red = df_HGB_red.groupby(['id'], as_index=False)[['value_HGB', 'time_HGB']].agg(list)
df_agg_Idl = df_Idl.groupby(['id'], as_index=False)[['value_Idl', 'time_Idl']].agg(list)
df_agg_SBP = df_SBP.groupby(['id'], as_index=False)[['value_SBP', 'time_SBP']].agg(list)

# Add a column of length of list
df_agg_creatinine['len_creatinine'] = df_agg_creatinine['value_creatinine'].apply(lambda x: len(x))
df_agg_DBP['len_DBP'] = df_agg_DBP['value_DBP'].apply(lambda x: len(x))
df_agg_glucose['len_glucose'] = df_agg_glucose['value_glucose'].apply(lambda x: len(x))
df_agg_HGB_red['len_HGB'] = df_agg_HGB_red['value_HGB'].apply(lambda x: len(x))
df_agg_Idl['len_Idl'] = df_agg_Idl['value_Idl'].apply(lambda x: len(x))
df_agg_SBP['len_SBP'] = df_agg_SBP['value_SBP'].apply(lambda x: len(x))

# concat columns
df_cat = pd.concat([df_agg_creatinine[['id', 'value_creatinine', 'time_creatinine', 'len_creatinine']],
                    df_agg_DBP[['value_DBP', 'time_DBP', 'len_DBP']]], axis=1)
df_cat = pd.concat([df_cat, df_agg_glucose[['value_glucose', 'time_glucose', 'len_glucose']]], axis=1)
df_cat = pd.concat([df_cat, df_agg_HGB_red[['value_HGB', 'time_HGB', 'len_HGB']]], axis=1)
df_cat = pd.concat([df_cat, df_agg_Idl[['value_Idl', 'time_Idl', 'len_Idl']]], axis=1)
df_cat = pd.concat([df_cat, df_agg_SBP[['value_SBP', 'time_SBP', 'len_SBP']]], axis=1)

df_cat = pd.merge(df_cat, df_stage, on="id", how="left").copy()

# generate approximated waves
out = generate_indicator_waves(df_cat, indicator_list, dict_mean_ind, dict_std_ind, max_end_day=699, max_patient_id=299)

# save the waves in csv
days = np.arange(700)
header = ['id', 'indicator']
header.extend(days)
with open('waves_indicators.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(out)
file.close()
