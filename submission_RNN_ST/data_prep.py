import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from statsmodels.stats.outliers_influence import variance_inflation_factor

class FeatureEngineer:
    def __init__(self, train_tab_csv: str, test_tab_csv: str, train_act_csv: str, test_act_csv: str):
        train_tab_df = pd.read_csv(train_tab_csv)
        test_tab_df = pd.read_csv(test_tab_csv)
        train_act_df = pd.read_csv(train_act_csv)
        test_act_df = pd.read_csv(test_act_csv)
        self.train_df = pd.merge(train_tab_df, train_act_df, on='id', how='left')
        self.test_df = pd.merge(test_tab_df, test_act_df, on='id', how='left')

        self.targets = [
            'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04',
            'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08',
            'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12',
            'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15', 'PCIAT-PCIAT_16',
            'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20',
            'PCIAT-PCIAT_Total', 'sii'
        ]
        
        
        self.train_df.drop(columns='PCIAT-Season', inplace=True, errors='ignore')
        self.test_df.drop(columns='PCIAT-Season', inplace=True, errors='ignore')

        
        self.categorical_cols = [
            'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
            'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
            'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season',
            'Basic_Demos-Sex', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
            'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL_Zone',
            'BIA-BIA_Activity_Level_num', 'BIA-BIA_Frame_num',
            'PreInt_EduHx-computerinternet_hoursday'
        ]
        
        
        self.numerical_cols = [
            col for col in self.train_df.columns if col not in self.categorical_cols + self.targets + ['id']
        ]

        print(f"Received training tabular data with shape: {self.train_df.shape}")
        print(f"Received test tabular data with shape: {self.test_df.shape}")    
        print(f"The columns in the training dataset are: {self.train_df.columns.tolist()}")
        print(f"The columns in the test dataset are: {self.test_df.columns.tolist()}")

    def preprocess(self):
        
        season_features = [
            'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
            'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
            'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season'
        ]
        label_mapping = {
            'Spring': 0,
            'Summer': 1,
            'Fall': 2,
            'Winter': 3
        }
        for categ in season_features:
            self.train_df[categ] = self.train_df[categ].map(label_mapping)
            self.test_df[categ] = self.test_df[categ].map(label_mapping)

        
        self.train_df.dropna(subset=self.targets, inplace=True)

    def impute(self):
        
        X_train_cat = self.train_df[self.categorical_cols].copy()
        X_train_num = self.train_df[self.numerical_cols].copy()
        X_test_cat = self.test_df[self.categorical_cols].copy()
        X_test_num = self.test_df[self.numerical_cols].copy()

        
        missing_counts = X_train_num.isnull().sum()
        cols_with_all_missing = missing_counts[missing_counts == X_train_num.shape[0]].index.tolist()
        print(f"Columns with all missing values: {cols_with_all_missing}")

        
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_train_cat_imputed = pd.DataFrame(imputer_cat.fit_transform(X_train_cat), columns=self.categorical_cols)
        X_test_cat_imputed = pd.DataFrame(imputer_cat.transform(X_test_cat), columns=self.categorical_cols)

        for col in self.categorical_cols:
            X_train_cat_imputed[col] = X_train_cat_imputed[col].astype(int)
            X_test_cat_imputed[col] = X_test_cat_imputed[col].astype(int)

        estimator = ExtraTreeRegressor(random_state=42)
        imputer_num = IterativeImputer(estimator=estimator, max_iter=20, imputation_order='roman', random_state=42)
        X_train_num_imputed = pd.DataFrame(imputer_num.fit_transform(X_train_num), columns=self.numerical_cols)
        X_test_num_imputed = pd.DataFrame(imputer_num.transform(X_test_num), columns=self.numerical_cols)

        self.train_features = pd.concat([X_train_num_imputed, X_train_cat_imputed], axis=1)
        self.test_features = pd.concat([X_test_num_imputed, X_test_cat_imputed], axis=1)

        train_ids = self.train_df['id'].reset_index(drop=True)
        train_targets = self.train_df[self.targets].reset_index(drop=True)
        test_ids = self.test_df['id'].reset_index(drop=True)

        self.train_imputed_df = self.train_features.copy()
        self.train_imputed_df['id'] = train_ids
        self.train_imputed_df[self.targets] = train_targets

        self.test_imputed_df = self.test_features.copy()
        self.test_imputed_df['id'] = test_ids

    def clean_outliers(self):
        feature_cols = self.train_features.columns.tolist()
        outlier_detector = IsolationForest(contamination=0.01, random_state=42)
        outliers = outlier_detector.fit_predict(self.train_imputed_df[feature_cols])

        num_outliers = (outliers == -1).sum()
        print(f"Number of outliers detected: {num_outliers}")

        self.train_imputed_df = self.train_imputed_df[outliers != -1].reset_index(drop=True)

    def scale(self):

        numerical_cols = self.train_imputed_df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in self.categorical_cols + self.targets + ['id']]
        
        scaler = RobustScaler()
        self.train_imputed_df[numerical_cols] = scaler.fit_transform(self.train_imputed_df[numerical_cols])
        self.test_imputed_df[numerical_cols] = scaler.transform(self.test_imputed_df[numerical_cols])

    def calculate_vif(self, df):
        vif_data = pd.DataFrame()
        vif_data['feature'] = df.columns
        vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        return vif_data

    def select_features(self):

        X = self.train_imputed_df.drop(columns=self.targets + ['id'])
        y = self.train_imputed_df['sii']
        
        model = Lasso(alpha=0.01, random_state=42)
        selector = RFE(model, n_features_to_select=80)
        selector.fit(X, y)
        
        selected_columns = X.columns[selector.support_]
        print(f"Selected columns: {selected_columns.tolist()}")
        
        selected_columns = list(selected_columns)
        
        print("Calculating VIF for selected features...")
        X_selected = self.train_imputed_df[selected_columns]
        vif_df = self.calculate_vif(X_selected)
        high_vif_features = vif_df[vif_df['VIF'] > 5]['feature']
        print(f"Features with VIF > 5: {high_vif_features.tolist()}")

        selected_columns = [col for col in selected_columns if col not in high_vif_features]
        print(f"Selected columns after VIF filtering: {selected_columns}")

        self.train_imputed_df = self.train_imputed_df[selected_columns + ['sii', 'id']]
        self.test_imputed_df = self.test_imputed_df[selected_columns + ['id']]
        print(f'Train dataset shape after feature selection: {self.train_imputed_df.shape}')
        print(f'Test dataset shape after feature selection: {self.test_imputed_df.shape}')

    def get_correlation(self):
        pass  

    def plot_statistics(self):
        pass  
