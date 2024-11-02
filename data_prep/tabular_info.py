import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter


class GetTabularInfo():
    def __init__(self, train_csv_path: str, test_csv_path: str):
        self.train_df = pd.read_csv(train_csv_path)
        self.test_df = pd.read_csv(test_csv_path)

    def get_missing_values(self):
        # Calculate missing values and ratio
        missing_count = self.train_df.isnull().sum().reset_index()
        missing_count.columns = ['feature', 'null_count']
        missing_count['null_ratio'] = missing_count['null_count'] / len(self.train_df)
        missing_count = missing_count.sort_values('null_ratio', ascending=False)

        # Plot
        plt.figure(figsize=(6, 15))
        plt.title('Missing values over the train dataset.')
        plt.barh(np.arange(len(missing_count)), missing_count['null_ratio'], color='coral', label='missing')
        plt.barh(np.arange(len(missing_count)), 
                 1 - missing_count['null_ratio'], 
                 left=missing_count['null_ratio'],
                 color='darkseagreen', label='available')
        plt.yticks(np.arange(len(missing_count)), missing_count['feature'])
        plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        plt.xlim(0, 1)
        plt.legend()
        plt.savefig("data_info/missing_values_plot.png")

    def get_demographics(self):
        pass

    def get_correlation(self):
        pass


if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'data/train.csv')
    TEST_CSV_PATH = os.path.join(ROOT_DIR, 'data/test.csv')

    data_processor = GetTabularInfo(train_csv_path=TRAIN_CSV_PATH, test_csv_path=TEST_CSV_PATH)
    data_processor.get_missing_values()