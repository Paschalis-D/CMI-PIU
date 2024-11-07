import pandas as pd

# Define the paths to the training and testing CSV files
TRAIN_CSV_PATH = './data/train.csv'
TEST_CSV_PATH = './data/test.csv'

train_df = pd.read_csv(TRAIN_CSV_PATH)
print(train_df.columns)
