from actigraphy_info import ActigraphyFeatures
from data_prep import FeatureEngineer
from train import TrainML

TRAIN_TAB_CSV = './data/train.csv'
TEST_TAB_CSV = './data/test.csv'
TRAIN_PARQUET_DIR = './data/series_train.parquet'
TEST_PARQUET_DIR = './data/series_test.parquet'
TRAIN_ACT_CSV = './data/train_actigraphy_features.csv'
TEST_ACT_CSV = './data/test_actigraphy_features.csv'
TRAIN_FINAL = './data/train_imputed.csv'
TEST_FINAL = './data/test_imputed.csv'
CONFIGS = './configs/ST_configs.json'
SAMPLE_SUBMISSION_CSV = './data/sample_submission.csv'
OUTPUT_SUBMISSION_CSV = 'submission.csv'
 
# af = ActigraphyFeatures(train_dir=TRAIN_PARQUET_DIR, test_dir=TEST_PARQUET_DIR)
# af.extract_features()
# train_features = af.get_train_features()
# test_features = af.get_test_features()
# train_features.to_csv(TRAIN_ACT_CSV, index=False)
# test_features.to_csv(TEST_ACT_CSV, index=False)
# print("Features saved successfully.")


fe = FeatureEngineer(train_tab_csv=TRAIN_TAB_CSV, test_tab_csv=TEST_TAB_CSV, train_act_csv=TRAIN_ACT_CSV, test_act_csv=TEST_ACT_CSV)
fe.preprocess()
fe.impute()
print('Train dataset shape after imputation:', fe.train_imputed_df.shape)
print('Test dataset shape after imputation:', fe.test_imputed_df.shape)
fe.clean_outliers()
print('Train dataset shape after outlier cleaning:', fe.train_imputed_df.shape)
fe.scale()
fe.select_features()
#fe.get_correlation()
#fe.plot_statistics()
fe.train_imputed_df.to_csv(TRAIN_FINAL, index=False)
fe.test_imputed_df.to_csv(TEST_FINAL, index=False)


trainer = TrainML(train_csv=TRAIN_FINAL, test_csv=TEST_FINAL, configs=CONFIGS)

test_predictions, optimized_thresholds, oof_predictions = trainer.train_model(random_state=42)

trainer.save_submission(submission_csv=SAMPLE_SUBMISSION_CSV, output_path=OUTPUT_SUBMISSION_CSV)