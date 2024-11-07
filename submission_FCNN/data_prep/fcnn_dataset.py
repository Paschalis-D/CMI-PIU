# data_prep/fcnn_dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class FCNNDataset(Dataset):
    def __init__(self, data_dir: str, features=None, is_train=True):
        super(FCNNDataset, self).__init__()
        
        # Load the CSV
        self.df = pd.read_csv(data_dir)
        self.is_train = is_train
        
        # Specify the feature columns you want (exclude 'id' here if it's present)
        self.features = features if features else [
            'Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score', 'Physical-BMI',
            'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
            'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
            'Fitness_Endurance-Time_Sec', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone',
            'FGC-FGC_GSND', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone',
            'FGC-FGC_PU', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone',
            'FGC-FGC_SRR', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone',
            'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_DEE',
            'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Frame_num',
            'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
            'BIA-BIA_TBW', 'PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total',
            'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T',
            'PreInt_EduHx-computerinternet_hoursday', 'wearing_mean', 'light_std', 
            'enmo_mean', 'enmo_std', 'acc_magn_mean', 'acc_magn_std'
        ]
        
        # Ensure that 'id' is not in self.features
        if 'id' in self.features:
            self.features.remove('id')
        
        # Define the target column
        self.target_column = 'sii'  # Replace 'sii' with the actual target column name if different
        
        # If 'id' column is present, save it separately
        if 'id' in self.df.columns:
            self.ids = self.df['id'].values
        else:
            self.ids = None

        # Filter DataFrame to include only features and target (if is_train) or 'id' (if not)
        if self.is_train:
            self.df = self.df[self.features + [self.target_column]]
            # Drop rows where features or target are NaN
            self.df = self.df.dropna(subset=self.features + [self.target_column]).reset_index(drop=True)
        else:
            # For test data, include 'id' in df
            self.df = self.df[self.features + ['id']]
            # Drop rows where features are NaN
            self.df = self.df.dropna(subset=self.features).reset_index(drop=True)
        
        # Ensure all feature columns are numeric
        self.df[self.features] = self.df[self.features].apply(pd.to_numeric, errors='coerce')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Extract features
        features = self.df.iloc[idx][self.features].values.astype(np.float32)
        features = torch.tensor(features, dtype=torch.float32)
        
        if self.is_train:
            # Retrieve the target as an integer (class index)
            target = int(self.df.iloc[idx][self.target_column])
            return features, target
        else:
            # For test data, return features and id
            id_value = self.df.iloc[idx]['id']
            return features, id_value

if __name__ == '__main__':
    # Example usage
    dataset = FCNNDataset(data_dir='./data/train_imputed_with_act.csv', is_train=True)
    features, target = dataset[45]
    print(f'Features: {features.shape}, Target: {target}')
