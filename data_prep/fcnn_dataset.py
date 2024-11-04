import pandas as pd
import torch
from torch.utils.data import Dataset

class FCNNDataset(Dataset):
    def __init__(self, data_dir: str, features=None):
        super(FCNNDataset, self).__init__()
        
        # Load the CSV
        self.df = pd.read_csv(data_dir)
        
        # Specify the feature columns you want
        self.features = features if features else [
            'Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score', 'Physical-BMI',
            'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
            'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
            'Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins',
            'Fitness_Endurance-Time_Sec', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone',
            'FGC-FGC_GSND', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone',
            'FGC-FGC_PU', 'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone',
            'FGC-FGC_SRR', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone',
            'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
            'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
            'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
            'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
            'BIA-BIA_TBW', 'PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total',
            'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T',
            'PreInt_EduHx-computerinternet_hoursday'
        ]
        
        # Filter the DataFrame to include only specified features and the target column
        self.df = self.df[self.features + [self.df.columns[-1]]]
        self.df = self.df.dropna(subset=[self.df.columns[-1]]).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Extract features and target
        features = torch.tensor(self.df.iloc[idx, :-1].values, dtype=torch.float32)
        
        # Retrieve the target as an integer (class index)
        target = int(self.df.iloc[idx, -1])
        
        return features, target

if __name__ == '__main__':
    dataset = FCNNDataset(data_dir='./data/train_imputed.csv')
    features, target = dataset[45]
    print(f'Features: {features.shape}, Target: {target}')