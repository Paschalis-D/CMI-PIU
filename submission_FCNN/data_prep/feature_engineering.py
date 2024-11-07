# data_prep/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
import os
import sys
ROOT = os.getcwd()
sys.path.append(ROOT)
from data_prep.actigraphy_info import GetActigraphyInfo

class FeatureEngineer:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                 train_actigraphy_dir: str, test_actigraphy_dir: str, 
                 target_columns: list = None):

        self.target_columns = target_columns if target_columns else []

        if self.target_columns:
            # Training data
            self.train_features = train_df.drop(columns=self.target_columns).select_dtypes(include=[np.number])
            self.train_features['id'] = train_df['id']  # Add 'id' back explicitly
            self.train_target = train_df[self.target_columns]

            # Test data
            # Only drop target columns if they exist in test_df
            columns_to_drop = [col for col in self.target_columns if col in test_df.columns]
            self.test_features = test_df.drop(columns=columns_to_drop).select_dtypes(include=[np.number])
            self.test_features['id'] = test_df['id']  # Add 'id' back explicitly
            # Since test data does not have target columns, set self.test_target to empty DataFrame
            self.test_target = pd.DataFrame()
        else:
            self.train_features = train_df.select_dtypes(include=[np.number])
            self.train_target = pd.DataFrame()  # No target columns

            self.test_features = test_df.select_dtypes(include=[np.number])
            self.test_target = pd.DataFrame()  # No target columns

        # Add actigraphy info to train and test features
        self.train_features, self.test_features = self.add_actigraphy_info(
            train_actigraphy_dir, test_actigraphy_dir
        )

    def add_actigraphy_info(self, train_actigraphy_dir: str, test_actigraphy_dir: str):
        # Process training data
        train_act_info = []
        for id in os.listdir(train_actigraphy_dir):
            parquet_file = os.path.join(train_actigraphy_dir, f'{id}/part-0.parquet')
            info = GetActigraphyInfo(parquet_file)
            id_means = info.means
            id_string = id.split('=')[1]
            id_means.insert(0, id_string)
            train_act_info.append(id_means)
        column_names = ['id', 'wearing_mean', 'light_std', 'enmo_mean', 'enmo_std', 'acc_magn_mean', 'acc_magn_std']
        train_act_info_df = pd.DataFrame(train_act_info, columns=column_names)

        # Full outer merge with train_features on 'id' to retain all ids
        merged_train_df = train_act_info_df.merge(self.train_features, on='id', how='outer')

        # Process test data
        test_act_info = []
        for id in os.listdir(test_actigraphy_dir):
            parquet_file = os.path.join(test_actigraphy_dir, f'{id}/part-0.parquet')
            info = GetActigraphyInfo(parquet_file)
            id_means = info.means
            id_string = id.split('=')[1]
            id_means.insert(0, id_string)
            test_act_info.append(id_means)
        test_act_info_df = pd.DataFrame(test_act_info, columns=column_names)

        # Full outer merge with test_features on 'id'
        merged_test_df = test_act_info_df.merge(self.test_features, on='id', how='outer')

        return merged_train_df, merged_test_df

    def perform_imputation(self):
        # Separate the 'id' column before imputation
        train_ids = self.train_features['id']
        test_ids = self.test_features['id']

        # Drop the 'id' column for imputation
        train_features_no_id = self.train_features.drop(columns=['id'])
        test_features_no_id = self.test_features.drop(columns=['id'])

        # Initialize the imputer
        imputer = IterativeImputer(max_iter=20, random_state=0)

        # Impute missing values
        train_features_imputed = pd.DataFrame(imputer.fit_transform(train_features_no_id),
                                              columns=train_features_no_id.columns)
        test_features_imputed = pd.DataFrame(imputer.transform(test_features_no_id),
                                             columns=test_features_no_id.columns)

        # Reattach 'id' column
        train_features_imputed['id'] = train_ids.values
        test_features_imputed['id'] = test_ids.values

        # Reattach target columns for training data
        if not self.train_target.empty:
            self.train_imputed = pd.concat([train_features_imputed.reset_index(drop=True), 
                                            self.train_target.reset_index(drop=True)], axis=1)
        else:
            self.train_imputed = train_features_imputed

        # For test data, target columns are empty
        self.test_imputed = test_features_imputed

        return self.train_imputed, self.test_imputed

if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'data/train.csv')
    TEST_CSV_PATH = os.path.join(ROOT_DIR, 'data/test.csv')
    TRAIN_ACTIGRAPHY_DIR = os.path.join(ROOT_DIR, 'data/series_train.parquet')
    TEST_ACTIGRAPHY_DIR = os.path.join(ROOT_DIR, 'data/series_test.parquet')

    # Load data
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    # Define target columns if necessary
    target_columns = [
        'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 
        'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 
        'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15', 
        'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20', 
        'PCIAT-PCIAT_Total', 'sii'
    ]

    # Initialize FeatureEngineer with target columns
    feature_engineer = FeatureEngineer(train_df, test_df, 
                                       train_actigraphy_dir=TRAIN_ACTIGRAPHY_DIR, 
                                       test_actigraphy_dir=TEST_ACTIGRAPHY_DIR, 
                                       target_columns=target_columns)
    
    # Perform imputation
    train_imputed, test_imputed = feature_engineer.perform_imputation()

    # Save the imputed DataFrames
    train_imputed.to_csv(os.path.join(ROOT_DIR, 'data/train_imputed_with_act.csv'), index=False)
    test_imputed.to_csv(os.path.join(ROOT_DIR, 'data/test_imputed_with_act.csv'), index=False)
