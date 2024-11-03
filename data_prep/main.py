import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns


ROOT_DIR = os.getcwd()
TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'data/train.csv')
TEST_CSV_PATH = os.path.join(ROOT_DIR, 'data/test.csv')

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

print(train_df.head())
print(train_df.columns)