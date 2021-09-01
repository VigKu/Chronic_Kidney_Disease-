import pandas as pd

src = 'dataScienceTask/T_demo.csv'

# read csv
df = pd.read_csv(src)
df_target = pd.read_csv('dataScienceTask/T_stage.csv')

# cubic of age to transform to normal distribution --> may or may not use later
df['new_age'] = df['age'].apply(lambda x: x ** 3 / 100000)

# modify and merge race
df['new_race'] = df['race']
df['new_race'].replace(to_replace="Unknown", value="White", inplace=True)
df['new_race'].replace(to_replace=["Hispanic", "Asian", "Black"], value="Non white", inplace=True)

# norm the age
df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()

df['stage'] = df_target['Stage_Progress']

# save new df
df.to_csv('data_demo.csv')
