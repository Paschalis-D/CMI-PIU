import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy import stats

class ActigraphyFeatures:
    def __init__(self, train_dir, test_dir):
        """
        Initialize the ActigraphyFeatures class with train and test directories.

        Parameters:
        - train_dir (str): Path to the training data directory.
        - test_dir (str): Path to the testing data directory.
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_features = None
        self.test_features = None

    def extract_advanced_features(self, data):
        """
        Extract advanced features from actigraphy data for SII prediction.

        Parameters:
        - data (pd.DataFrame): Input DataFrame with actigraphy measurements.

        Returns:
        - pd.DataFrame: Single-row DataFrame with extracted features.
        """
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['relative_date_PCIAT'], unit='D') + pd.to_timedelta(data['time_of_day'])
        data = data[data['non-wear_flag'] == 0]
        
        required_columns = ['X', 'Y', 'Z', 'enmo', 'light', 'anglez', 'battery_voltage', 'non-wear_flag', 'relative_date_PCIAT', 'time_of_day']
        for col in required_columns:
            if col not in data.columns:
                data[col] = np.nan  
        
        if data.empty:
            return pd.DataFrame()
        
        # Calculate basic metrics
        data['magnitude'] = np.sqrt(data['X']**2 + data['Y']**2 + data['Z']**2)
        data['velocity'] = data['magnitude']
        data['distance'] = data['velocity'] * 5  # 5 seconds per observation
        data['date'] = data['timestamp'].dt.date
        data['weekday'] = data['timestamp'].dt.weekday + 1  # Monday=1, Sunday=7
        data['quarter'] = data['timestamp'].dt.quarter
        hour = data['timestamp'].dt.hour
        
        # Calculate aggregated distances (not used in features but kept for potential future use)
        # distances = {
        #     'daily': data.groupby('date')['distance'].sum(),
        #     'monthly': data.groupby(data['timestamp'].dt.to_period('M'))['distance'].sum(),
        #     'quarterly': data.groupby(data['quarter'])['distance'].sum()
        # }
        
        features = {}
        
        time_masks = {
            'morning': (hour >= 6) & (hour < 12),
            'afternoon': (hour >= 12) & (hour < 18),
            'evening': (hour >= 18) & (hour < 22),
            'night': (hour >= 22) | (hour < 6)
        }
        
        # 1. Activity Pattern Features
        for period, mask in time_masks.items():
            features.update({
                f'{period}_activity_mean': data.loc[mask, 'magnitude'].mean(),
                f'{period}_activity_std': data.loc[mask, 'magnitude'].std(),
                f'{period}_enmo_mean': data.loc[mask, 'enmo'].mean()
            })
        
        # 2. Sleep Quality Features
        sleep_hours = time_masks['night']
        magnitude_threshold = data['magnitude'].mean() + data['magnitude'].std()
        
        features.update({
            'sleep_movement_mean': data.loc[sleep_hours, 'magnitude'].mean(),
            'sleep_movement_std': data.loc[sleep_hours, 'magnitude'].std(),
            'sleep_disruption_count': len(data.loc[sleep_hours & (data['magnitude'] > data['magnitude'].mean() + 2 * data['magnitude'].std())]),
            'light_exposure_during_sleep': data.loc[sleep_hours, 'light'].mean(),
            'sleep_position_changes': len(data.loc[sleep_hours & (abs(data['anglez'].diff()) > 45)]),
            'good_sleep_cycle': int((data.loc[sleep_hours, 'light'].mean() or 0) < 50)
        })
        
        # 3. Activity Intensity Features
        features.update({
            'sedentary_time_ratio': (data['magnitude'] < magnitude_threshold * 0.5).mean(),
            'moderate_activity_ratio': ((data['magnitude'] >= magnitude_threshold * 0.5) & (data['magnitude'] < magnitude_threshold * 1.5)).mean(),
            'vigorous_activity_ratio': (data['magnitude'] >= magnitude_threshold * 1.5).mean(),
            'activity_peaks_per_day': len(data[data['magnitude'] > data['magnitude'].quantile(0.95)]) / data['relative_date_PCIAT'].nunique()
        })
        
        # 4. Circadian Rhythm Features
        hourly_activity = data.groupby(hour)['magnitude'].mean()
        features.update({
            'circadian_regularity': (hourly_activity.std() / hourly_activity.mean()) if hourly_activity.mean() != 0 else np.nan,
            'peak_activity_hour': hourly_activity.idxmax(),
            'trough_activity_hour': hourly_activity.idxmin(),
            'activity_range': hourly_activity.max() - hourly_activity.min()
        })
        
        # 5-11. Additional Feature Groups
        weekend_mask = data['weekday'].isin([6, 7])
        
        features.update({
            # Movement Patterns
            'movement_entropy': stats.entropy(pd.qcut(data['magnitude'], q=10, duplicates='drop').value_counts()),
            'direction_changes': len(data[abs(data['anglez'].diff()) > 30]) / len(data),
            'sustained_activity_periods': len(data[data['magnitude'].rolling(12, min_periods=1).mean() > magnitude_threshold]) / len(data),
            
            # Weekend vs Weekday
            'weekend_activity_ratio': data.loc[weekend_mask, 'magnitude'].mean() / data.loc[~weekend_mask, 'magnitude'].mean(),
            'weekend_sleep_difference': data.loc[weekend_mask & sleep_hours, 'magnitude'].mean() - data.loc[~weekend_mask & sleep_hours, 'magnitude'].mean(),
            
            # Non-wear Time
            'wear_time_ratio': (data['non-wear_flag'] == 0).mean(),
            'wear_consistency': data['non-wear_flag'].nunique(),
            'longest_wear_streak': data['non-wear_flag'].eq(0).astype(int).groupby((data['non-wear_flag'] != 0).cumsum()).sum().max(),
            
            # Device Usage
            'screen_time_proxy': (data['light'] > data['light'].quantile(0.75)).mean(),
            'dark_environment_ratio': (data['light'] < data['light'].quantile(0.25)).mean(),
            'light_variation': (data['light'].std() / data['light'].mean()) if data['light'].mean() != 0 else np.nan,
            
            # Battery Usage
            'battery_drain_rate': -np.polyfit(range(len(data)), data['battery_voltage'].fillna(0), 1)[0],
            'battery_variability': data['battery_voltage'].std(),
            'low_battery_time': (data['battery_voltage'] < data['battery_voltage'].quantile(0.1)).mean(),
            
            # Time-based
            'days_monitored': data['relative_date_PCIAT'].nunique(),
            'total_active_hours': len(data[data['magnitude'] > magnitude_threshold * 0.5]) * 5 / 3600,  # Convert to hours
            'activity_regularity': data.groupby('weekday')['magnitude'].mean().std()
        })
        
        # Variability Features for multiple columns
        for col in ['X', 'Y', 'Z', 'enmo', 'anglez']:
            features.update({
                f'{col}_skewness': data[col].skew(),
                f'{col}_kurtosis': data[col].kurtosis(),
                f'{col}_trend': np.polyfit(range(len(data)), data[col].fillna(0), 1)[0] if data[col].notnull().any() else np.nan
            })
        
        # Convert features dictionary to DataFrame
        features_df = pd.DataFrame([features])
        
        return features_df

    def process_file(self, participant_id, dirname):
        """
        Process a single participant's file to extract features.

        Parameters:
        - participant_id (str): ID of the participant.
        - dirname (str): Directory containing participant folders.

        Returns:
        - Tuple[pd.DataFrame, str]: Extracted features DataFrame and participant ID.
        """
        try:
            file_path = os.path.join(dirname, participant_id, 'part-0.parquet')
            df = pd.read_parquet(file_path)
            features = self.extract_advanced_features(df)
            if features.empty:
                print(f"Warning: No data after processing for participant {participant_id}")
                return None, participant_id
            return features, participant_id
        except Exception as e:
            print(f"Error processing participant {participant_id}: {e}")
            return None, participant_id

    def load_time_series(self, dirname):
        """
        Load and process all time series data from the directory.

        Parameters:
        - dirname (str): Directory containing participant folders.

        Returns:
        - pd.DataFrame: DataFrame with extracted features for all participants.
        """
        participant_ids = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
        features_list = []
        ids_list = []

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda pid: self.process_file(pid, dirname), participant_ids), total=len(participant_ids)))
        
        for features, pid in results:
            if features is not None:
                features['id'] = pid.split('=')[-1]
                features_list.append(features)
        
        if features_list:
            df = pd.concat(features_list, ignore_index=True)
            return df
        else:
            print("No features extracted.")
            return pd.DataFrame()

    def extract_features(self):
        """
        Extract features for both train and test datasets.
        """
        print("Extracting features from training data...")
        self.train_features = self.load_time_series(self.train_dir)
        print("Training features extraction completed.")
        print("Extracting features from test data...")
        self.test_features = self.load_time_series(self.test_dir)
        print("Test features extraction completed.")

    def get_train_features(self):
        """
        Get the extracted training features.

        Returns:
        - pd.DataFrame: Training features DataFrame.
        """
        return self.train_features

    def get_test_features(self):
        """
        Get the extracted test features.

        Returns:
        - pd.DataFrame: Test features DataFrame.
        """
        return self.test_features

if __name__ == "__main__":
    TRAIN_PARQUET_DIR = './data/series_train.parquet'
    TEST_PARQUET_DIR = './data/series_test.parquet'

    af = ActigraphyFeatures(train_dir=TRAIN_PARQUET_DIR, test_dir=TEST_PARQUET_DIR)
    af.extract_features()
    train_features = af.get_train_features()
    test_features = af.get_test_features()
    train_features.to_csv('./data/train_actigraphy_features.csv', index=False)
    test_features.to_csv('./data/test_actigraphy_features.csv', index=False)
    print("Features saved successfully.")