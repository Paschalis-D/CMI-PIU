import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class GetActigraphyInfo:
    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path)
        self.df = self.get_acc_magn(self.df)
        self.means = self.get_means(self.df)

    def get_acc_magn(self, df: pd.DataFrame):
        """
        This function will calculate the magnitude of the acceleration based on the values of the X,Y,Z acceleraton measured by the wrist watch
        """
        df['acc_magn'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
        return df
    
    def get_means(self, df: pd.DataFrame):
        """
        This function will creturn the mean values for the acc_magn, enmo, anglez, light, battery_voltage and return them 
        """
        acc_magn_mean = df['acc_magn'].mean()
        acc_magn_std = df['acc_magn'].std()
        enmo_mean = df['enmo'].mean()
        enmo_std = df['enmo'].std()
        anglez_mean = df['anglez'].mean()
        anglez_std = df['anglez'].std()
        light_mean = df['light'].mean()
        light_std = df['light'].std()
        battery_voltage_mean = df['battery_voltage'].mean()
        wearing_mean = df['non-wear_flag'].mean()
        wearing_std = df['non-wear_flag'].std()
        #return [acc_magn_mean, acc_magn_std, enmo_mean, enmo_std, anglez_mean, anglez_std, light_mean, light_std, battery_voltage_mean, wearing_mean, wearing_std]
        return [wearing_mean, light_std, enmo_mean, enmo_std, acc_magn_mean, acc_magn_std]


if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    ACTIGRAPHY_DIR = os.path.join(ROOT_DIR, 'data/series_train.parquet')
    TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'data/train.csv')
    all_means = []
    for id in os.listdir(ACTIGRAPHY_DIR):
        parquet_file = os.path.join(ACTIGRAPHY_DIR, f'{id}/part-0.parquet')
        info = GetActigraphyInfo(parquet_file)
        id_means = info.means
        id_string = id.split('=')[1]
        id_means.insert(0, id_string)
        all_means.append(id_means)
    #column_names = ["id", "acc_magn_mean", "acc_magn_std", "enmo_mean", "enmo_std", "anglez_mean", "anglez_std", "light_mean", "light_std", "battery_voltage_mean", "wearing_mean", "wearing_std"]
    column_names = ['wearing_mean', 'light_std', 'enmo_mean', 'enmo_std', 'acc_magn_mean', 'acc_magn_std']
    all_means_df = pd.DataFrame(all_means, columns=column_names)
    train_csv_df = pd.read_csv(TRAIN_CSV_PATH)
    merged_df = all_means_df.merge(train_csv_df[['id', 'sii']], on='id', how='left')
    
    numeric_data = merged_df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix")
    plt.savefig(os.path.join("data_info", "correlation_matrix_actigraphy.png"))
    plt.close()

"""
correlations between extracted values from the actigraphy and the ssi
wearing_mean = 0.12
light_std = = 0.11
enmo_mean = enmo_std = -0.21
acc_magn_mean = 0.22
acc_magn_std = -0.21
"""
