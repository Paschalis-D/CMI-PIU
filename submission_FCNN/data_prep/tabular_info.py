import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class GetTabularInfo:
    def __init__(self, train_csv_path: str, test_csv_path: str):
        self.train_df = pd.read_csv(train_csv_path)
        self.test_df = pd.read_csv(test_csv_path)
        
        columns_to_encode = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
                             'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
                             'PAQ_A-Season', 'PAQ_C-Season', 'PCIAT-Season', 
                             'SDS-Season', 'PreInt_EduHx-Season']
        
        train_columns = [col for col in columns_to_encode if col in self.train_df.columns]
        test_columns = [col for col in columns_to_encode if col in self.test_df.columns]

        self.train_encoded = pd.get_dummies(self.train_df, columns=train_columns)
        self.test_encoded = pd.get_dummies(self.test_df, columns=test_columns)
        
        if not os.path.exists("data_info"):
            os.makedirs("data_info")

    def get_missing_values(self):
        missing_count = self.train_df.isnull().sum().reset_index()
        missing_count.columns = ['feature', 'null_count']
        missing_count['null_ratio'] = missing_count['null_count'] / len(self.train_df)
        missing_count = missing_count.sort_values('null_ratio', ascending=False)

        plt.figure(figsize=(15, 15))
        plt.title('Missing values over the train dataset')
        plt.barh(np.arange(len(missing_count)), missing_count['null_ratio'], color='coral', label='missing')
        plt.barh(np.arange(len(missing_count)), 
                 1 - missing_count['null_ratio'], 
                 left=missing_count['null_ratio'],
                 color='darkseagreen', label='available')
        plt.yticks(np.arange(len(missing_count)), missing_count['feature'])
        plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        plt.xlim(0, 1)
        plt.legend()
        plt.savefig(os.path.join("data_info", "missing_values_plot_imputed.png"))
        plt.close()

    def get_correlation(self):
        numeric_data = self.train_encoded.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()

        plt.figure(figsize=(20, 20))
        sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Feature Correlation Matrix")
        plt.savefig(os.path.join("data_info", "correlation_matrix_plot_imputed.png"))
        plt.close()


if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'data/train_imputed.csv')
    TEST_CSV_PATH = os.path.join(ROOT_DIR, 'data/test_imputed.csv')

    data_processor = GetTabularInfo(train_csv_path=TRAIN_CSV_PATH, test_csv_path=TEST_CSV_PATH)
    data_processor.get_missing_values()
    data_processor.get_correlation()