import json
import pandas as pd
import numpy as np
import csv
from utils import select_drugs, generate_drug_waves

thresh_sample = 30  # only include if more than 10% of sample size

# read csv
df_meds = pd.read_csv('dataScienceTask/T_meds.csv')
df_stage = pd.read_csv('dataScienceTask/T_stage.csv')


# get meand and std dosages for respective drugs
df_drug_maxdosage = df_meds.groupby(['drug'], as_index=False)['daily_dosage'].mean()
df_drug_maxdosage = df_drug_maxdosage.rename(columns={'daily_dosage':'max_dosage'})
df_drug_maxdosage = df_drug_maxdosage.set_index('drug')
dict_max_dosages = df_drug_maxdosage.to_dict()

# find the max end day --> all 1D signals will end with max_end_day (ensure all signals are of equal length)
max_end_day = df_meds['end_day'].max()
# combining rows of same id by aggregating in a list
df_agg = df_meds.groupby(['id','drug'], as_index=False)[['daily_dosage','start_day','end_day']].agg(list)
df_aggnew = df_agg.groupby(['id'], as_index=False)[['drug','daily_dosage','start_day','end_day']].agg(list)


## Feature Selection --> dimension reduction
# Remove drug if no. of patients is less than 10% of the sample size.
red_drug_list = select_drugs(df_meds, thresh_sample=30)

## Feature Engineering
# obtain all the waves over 700 days for the selected drugs for all patients.
# this includes those who do not take drugs.
out = generate_drug_waves(df_aggnew,
                          red_drug_list,
                          dict_max_dosages,
                          max_end_day=699,
                          max_patient_id=299)

# save the waves in csv
days = np.arange(700)
header = ['id', ' drug']
header.extend(days)
with open('waves_meds.csv', "w") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(out)
file.close()
