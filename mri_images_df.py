import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


class MriDataFrame:
    def __init__(self, file_path):
        self.category = file_path.split(sep='/')[-1].split(sep='.')[0]
        self.df = pd.read_csv(file_path)
        self.df.rename(columns={'Unnamed: 0': 'file_name'}, inplace=True)
        self.features = self.df.columns

    def print(self):
        print(f'MriDataFrame from {self.category}')
        print(f'Features: {self.features}')
        print(f'Data:')
        print(self.df)
