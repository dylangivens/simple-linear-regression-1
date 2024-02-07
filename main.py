import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 200)

df = pd.read_csv('C:/Users/dtgiv/Downloads/team.csv')
print(df)

print(df.columns)

print(df.head())