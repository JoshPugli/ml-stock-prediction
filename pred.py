import numpy as np
import pandas as pd

# Start with amazon data
df = pd.read_csv('./data/AMZN_2006-01-01_to_2018-01-01.csv', index_col="Date", parse_dates=True)

# Drop the "name" column
df.drop("Name", axis=1, inplace=True)

# Sort the data by date
df.sort_values(by="Date", inplace=True)

print(df.head())
print(df.tail())
print(df.shape)
