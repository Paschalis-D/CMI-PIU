import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
import os

class FeatureEngineer:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_columns: list = None):
        """
        Initialize with train and test DataFrames and optionally specify target columns to exclude from imputation.
        
        Parameters:
        - train_df: pd.DataFrame, the training dataset
        - test_df: pd.DataFrame, the test dataset
        - target_columns: list of str, names of the columns considered target values (optional).
        """
        self.target_columns = target_columns if target_columns else []
        
        # Separate features and target in the training data
        if self.target_columns:
            self.train_features = train_df.drop(columns=self.target_columns).select_dtypes(include=[np.number])
            self.train_target = train_df[self.target_columns]
        else:
            self.train_features = train_df.select_dtypes(include=[np.number])
            self.train_target = pd.DataFrame()  # No target columns

        # Select only numeric columns in the test data
        self.test_features = test_df.select_dtypes(include=[np.number])
    
    def perform_imputation(self):
        """
        Perform iterative imputation on train and test datasets, excluding the target columns.
        Returns imputed train and test DataFrames, with target columns reattached if applicable.
        """
        imputer = IterativeImputer(max_iter=20, random_state=0)
        
        # Impute missing values in training features
        self.train_features_imputed = pd.DataFrame(imputer.fit_transform(self.train_features), 
                                                   columns=self.train_features.columns)
        
        # Impute missing values in test features using the same imputer fitted on training data
        self.test_features_imputed = pd.DataFrame(imputer.transform(self.test_features), 
                                                  columns=self.test_features.columns)
        
        # Reattach the target columns for the training data if any
        if not self.train_target.empty:
            self.train_imputed = pd.concat([self.train_features_imputed, self.train_target.reset_index(drop=True)], axis=1)
        else:
            self.train_imputed = self.train_features_imputed
        
        return self.train_imputed, self.test_features_imputed


if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'data/train.csv')
    TEST_CSV_PATH = os.path.join(ROOT_DIR, 'data/test.csv')

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
    feature_engineer = FeatureEngineer(train_df, test_df, target_columns=target_columns)
    
    # Perform imputation
    train_imputed, test_imputed = feature_engineer.perform_imputation()

    # Save or use the imputed DataFrames as needed
    train_imputed.to_csv(os.path.join(ROOT_DIR, 'data/train_imputed.csv'), index=False)
    test_imputed.to_csv(os.path.join(ROOT_DIR, 'data/test_imputed.csv'), index=False)
