import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.tree import ExtraTreeRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest,mutual_info_classif

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
        
        # Drop 'PCIAT-Season' if not needed
        self.train_df.drop(columns='PCIAT-Season', inplace=True, errors='ignore')
        self.test_df.drop(columns='PCIAT-Season', inplace=True, errors='ignore')

        # Define categorical columns accessible throughout the class
        self.categorical_cols = [
            'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
            'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
            'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season',
            'Basic_Demos-Sex', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
            'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL_Zone',
            'BIA-BIA_Activity_Level_num', 'BIA-BIA_Frame_num',
            'PreInt_EduHx-computerinternet_hoursday'
        ]
        
        # All columns that are not categorical, targets, or 'id' are considered numerical
        self.numerical_cols = [
            col for col in self.train_df.columns if col not in self.categorical_cols + self.targets + ['id']
        ]

        print(f"Received training tabular data with shape: {self.train_df.shape}")
        print(f"Received test tabular data with shape: {self.test_df.shape}")    
        print(f"The columns in the training dataset are: {self.train_df.columns.tolist()}")
        print(f"The columns in the test dataset are: {self.test_df.columns.tolist()}")

    def preprocess(self):
        # Map string categorical features to numerical values
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

        # Drop rows in the train dataset that don't have target values
        self.train_df.dropna(subset=self.targets, inplace=True)

    def impute(self):
        # Separate features
        X_train_cat = self.train_df[self.categorical_cols].copy()
        X_train_num = self.train_df[self.numerical_cols].copy()
        X_test_cat = self.test_df[self.categorical_cols].copy()
        X_test_num = self.test_df[self.numerical_cols].copy()

        # Check for columns with all missing values
        missing_counts = X_train_num.isnull().sum()
        cols_with_all_missing = missing_counts[missing_counts == X_train_num.shape[0]].index.tolist()
        print(f"Columns with all missing values: {cols_with_all_missing}")

        # Impute categorical features
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_train_cat_imputed = pd.DataFrame(imputer_cat.fit_transform(X_train_cat), columns=self.categorical_cols)
        X_test_cat_imputed = pd.DataFrame(imputer_cat.transform(X_test_cat), columns=self.categorical_cols)

        # Ensure categorical columns are integer type
        for col in self.categorical_cols:
            X_train_cat_imputed[col] = X_train_cat_imputed[col].astype(int)
            X_test_cat_imputed[col] = X_test_cat_imputed[col].astype(int)

        # Impute numerical features
        imputer_num = IterativeImputer(max_iter=100, verbose=2, tol=5e-7, random_state=42)
        X_train_num_imputed = pd.DataFrame(imputer_num.fit_transform(X_train_num), columns=self.numerical_cols)
        X_test_num_imputed = pd.DataFrame(imputer_num.transform(X_test_num), columns=self.numerical_cols)

        # Combine features
        self.train_features = pd.concat([X_train_num_imputed, X_train_cat_imputed], axis=1)
        self.test_features = pd.concat([X_test_num_imputed, X_test_cat_imputed], axis=1)

        # Add IDs and targets
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
        # Recompute numerical columns
        numerical_cols = self.train_imputed_df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in self.categorical_cols + self.targets + ['id']]
        # Scale numerical features
        scaler = RobustScaler()
        self.train_imputed_df[numerical_cols] = scaler.fit_transform(self.train_imputed_df[numerical_cols])
        self.test_imputed_df[numerical_cols] = scaler.transform(self.test_imputed_df[numerical_cols])


    def select_features(self):
        # Ensure X is correctly assigned after dropping
        X = self.train_imputed_df.drop(columns=self.targets + ['id'])
        y = self.train_imputed_df['sii']
        
        # Initialize and fit the selector
        selector = SelectKBest(score_func=mutual_info_classif, k=50)
        selector.fit(X, y)
        
        # Store selected feature names to ensure consistency
        selected_columns = X.columns[selector.get_support()]
        print(f"Selected columns: {selected_columns.tolist()}")
        
        # Convert selected_columns to list
        selected_columns = list(selected_columns)
        
        # Transform train and test sets to keep the same columns
        self.train_imputed_df = self.train_imputed_df[selected_columns + ['sii', 'id']]
        self.test_imputed_df = self.test_imputed_df[selected_columns + ['id']]
        print(f'Train dataset shape after feature selection: {self.train_imputed_df.shape}')
        print(f'Test dataset shape after feature selection: {self.test_imputed_df.shape}')


    def get_correlation(self):
        print("Calculating correlation...")
        # Select numeric columns, excluding 'id' and target columns except 'sii'
        numeric_columns = self.train_imputed_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != 'id' and (col not in self.targets or col == 'sii')]

        pearson_corr = self.train_imputed_df[numeric_columns].corr()
        kendall_corr = self.train_imputed_df[numeric_columns].corr(method='kendall')
        spearman_corr = self.train_imputed_df[numeric_columns].corr(method='spearman')

        print("Pearson Correlation")
        print(pearson_corr)
        print("\nKendall Correlation")
        print(kendall_corr)
        print("\nSpearman Correlation")
        print(spearman_corr)


    def plot_statistics(self):
        # Ensure 'sii' is numeric
        self.train_imputed_df['sii'] = pd.to_numeric(self.train_imputed_df['sii'], errors='coerce')

        print("Plotting correlation matrix...")
        plt.figure(figsize=(12, 10))

        # Select numeric columns, excluding 'id' and target columns
        numeric_columns = self.train_imputed_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in ['id']]

        corr_matrix = self.train_imputed_df[numeric_columns].corr()

        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
        plt.title('Feature Correlation Matrix')
        plt.show()

        print("Plotting feature distributions...")
        # Use the columns from the train_imputed_df after feature selection
        feature_columns = [col for col in self.train_imputed_df.columns if col not in ['id', 'sii']]

        num_features = len(feature_columns)
        num_plots = min(num_features, 20)
        cols = 4
        rows = (num_plots + cols - 1) // cols
        plt.figure(figsize=(20, 5 * rows))
        for i, col in enumerate(feature_columns[:num_plots]):
            plt.subplot(rows, cols, i + 1)
            sns.histplot(self.train_imputed_df[col], kde=True)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()

        print("Plotting correlation with target 'sii'...")
        if 'sii' in corr_matrix.columns:
            target_corr = corr_matrix['sii'].drop(labels=['sii'], errors='ignore').sort_values(ascending=False)
            print("Top 10 features positively correlated with 'sii':\n", target_corr.head(10))
            print("Top 10 features negatively correlated with 'sii':\n", target_corr.tail(10))

            plt.figure(figsize=(10, 6))
            target_corr.plot(kind='bar')
            plt.title("Feature Correlation with Target 'sii'")
            plt.xlabel('Features')
            plt.ylabel('Correlation coefficient')
            plt.tight_layout()
            plt.show()
        else:
            print("'sii' is not found in correlation matrix columns.")

if __name__ == '__main__':
    TRAIN_TAB_CSV = './data/train.csv'
    TEST_TAB_CSV = './data/test.csv'
    TRAIN_ACT_CSV = './data/train_actigraphy_features.csv'
    TEST_ACT_CSV = './data/test_actigraphy_features.csv'
    fe = FeatureEngineer(train_tab_csv=TRAIN_TAB_CSV, test_tab_csv=TEST_TAB_CSV, train_act_csv=TRAIN_ACT_CSV, test_act_csv=TEST_ACT_CSV)
    fe.preprocess()
    fe.impute()
    print('Train dataset shape after imputation:', fe.train_imputed_df.shape)
    print('Test dataset shape after imputation:', fe.test_imputed_df.shape)
    fe.clean_outliers()
    print('Train dataset shape after outlier cleaning:', fe.train_imputed_df.shape)
    fe.scale()
    fe.select_features()
    fe.get_correlation()
    fe.plot_statistics()
    fe.train_imputed_df.to_csv('./data/train_imputed.csv', index=False)
    fe.test_imputed_df.to_csv('./data/test_imputed.csv', index=False)
