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

    def get_as_list(self, class_list):
        idx = class_list.index(self.category)
        class_vec = [idx for _ in range(len(self.df))]
        feature_vec = self.df.loc[:, self.df.columns != 'file_name'].values.tolist()

        return feature_vec, class_vec