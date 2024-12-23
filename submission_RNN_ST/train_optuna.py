from actigraphy_info import ActigraphyFeatures
from data_prep import FeatureEngineer
from train import TrainML
import optuna
import json
import warnings
import joblib
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
import numpy as np

warnings.filterwarnings("ignore", message="No further splits with positive gain")


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


fe = FeatureEngineer(train_tab_csv=TRAIN_TAB_CSV, test_tab_csv=TEST_TAB_CSV,
                      train_act_csv=TRAIN_ACT_CSV, test_act_csv=TEST_ACT_CSV)
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
fe.train_imputed_df.to_csv(TRAIN_FINAL, index=False)
fe.test_imputed_df.to_csv(TEST_FINAL, index=False)


def objective(trial):
    
    with open(CONFIGS) as f:
        config = json.load(f)
    
    
    config['lgbm']['learning_rate'] = trial.suggest_float('lgbm_learning_rate', 0.01, 0.1, log=True)
    config['lgbm']['max_depth'] = trial.suggest_int('lgbm_max_depth', 3, 15)
    config['lgbm']['num_leaves'] = trial.suggest_int('lgbm_num_leaves', 10, 50)
    config['lgbm']['min_data_in_leaf'] = trial.suggest_int('lgbm_min_data_in_leaf', 5, 50)
    config['lgbm']['feature_fraction'] = trial.suggest_float('lgbm_feature_fraction', 0.6, 1.0)
    config['lgbm']['bagging_fraction'] = trial.suggest_float('lgbm_bagging_fraction', 0.6, 1.0)
    config['lgbm']['bagging_freq'] = trial.suggest_int('lgbm_bagging_freq', 1, 10)
    config['lgbm']['lambda_l1'] = trial.suggest_float('lgbm_lambda_l1', 0.0, 10.0)
    config['lgbm']['lambda_l2'] = trial.suggest_float('lgbm_lambda_l2', 0.0, 10.0)
    config['lgbm']['n_estimators'] = trial.suggest_int('lgbm_n_estimators', 100, 500)

    
    config['xgb']['learning_rate'] = trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True)
    config['xgb']['max_depth'] = trial.suggest_int('xgb_max_depth', 3, 15)
    config['xgb']['n_estimators'] = trial.suggest_int('xgb_n_estimators', 100, 500)
    config['xgb']['subsample'] = trial.suggest_float('xgb_subsample', 0.5, 1.0)
    config['xgb']['colsample_bytree'] = trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0)
    config['xgb']['reg_alpha'] = trial.suggest_float('xgb_reg_alpha', 0.0, 10.0)
    config['xgb']['reg_lambda'] = trial.suggest_float('xgb_reg_lambda', 0.0, 10.0)

    
    config['catboost']['learning_rate'] = trial.suggest_float('cat_learning_rate', 0.01, 0.1, log=True)
    config['catboost']['depth'] = trial.suggest_int('cat_depth', 3, 10)
    config['catboost']['iterations'] = trial.suggest_int('cat_iterations', 100, 500)
    config['catboost']['l2_leaf_reg'] = trial.suggest_float('cat_l2_leaf_reg', 1.0, 10.0)
    config['catboost'].pop('reg_lambda', None)

    
    config['elastic_net']['alpha'] = trial.suggest_float('elastic_net_alpha', 0.0001, 10.0, log=True)
    config['elastic_net']['l1_ratio'] = trial.suggest_float('elastic_net_l1_ratio', 0.0, 1.0)
    config['elastic_net']['max_iter'] = 1000  

    
    config['hist_gbr']['learning_rate'] = trial.suggest_float('hist_gbr_learning_rate', 0.005, 0.2, log=True)
    config['hist_gbr']['max_iter'] = trial.suggest_int('hist_gbr_max_iter', 100, 500)
    config['hist_gbr']['max_depth'] = trial.suggest_int('hist_gbr_max_depth', 3, 15)
    config['hist_gbr']['min_samples_leaf'] = trial.suggest_int('hist_gbr_min_samples_leaf', 5, 50)
    config['hist_gbr']['l2_regularization'] = trial.suggest_float('hist_gbr_l2_regularization', 0.0, 10.0)
    config['hist_gbr']['early_stopping'] = False  

    
    config['lgbm']['max_depth'] = int(config['lgbm']['max_depth'])
    config['lgbm']['num_leaves'] = int(config['lgbm']['num_leaves'])
    config['lgbm']['min_data_in_leaf'] = int(config['lgbm']['min_data_in_leaf'])
    config['lgbm']['bagging_freq'] = int(config['lgbm']['bagging_freq'])
    config['lgbm']['n_estimators'] = int(config['lgbm']['n_estimators'])

    config['xgb']['max_depth'] = int(config['xgb']['max_depth'])
    config['xgb']['n_estimators'] = int(config['xgb']['n_estimators'])

    config['catboost']['depth'] = int(config['catboost']['depth'])
    config['catboost']['iterations'] = int(config['catboost']['iterations'])

    config['hist_gbr']['max_iter'] = int(config['hist_gbr']['max_iter'])
    config['hist_gbr']['max_depth'] = int(config['hist_gbr']['max_depth'])
    config['hist_gbr']['min_samples_leaf'] = int(config['hist_gbr']['min_samples_leaf'])

    
    train_df = pd.read_csv(TRAIN_FINAL)
    X = train_df.drop(columns=['sii', 'id'])
    y = train_df['sii']

    
    y_binned = pd.cut(y, bins=5, labels=False)

    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y))
    for train_idx, val_idx in kf.split(X, y_binned):
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

        
        trainer = TrainML(train_csv=None, test_csv=None, configs=config)
        trainer.voting_model.fit(X_train, y_train)

        
        y_pred = trainer.voting_model.predict(X_valid)
        oof_predictions[val_idx] = y_pred

    
    initial_thresholds = [0.5, 1.5, 2.5]
    optimization_result = minimize(
        trainer.evaluate_predictions,
        x0=initial_thresholds,
        args=(y, oof_predictions),
        method='Nelder-Mead'
    )
    optimized_thresholds = optimization_result.x

    
    y_pred_classes = trainer.threshold_rounder(oof_predictions, optimized_thresholds)
    qwk = trainer.quadratic_weighted_kappa(y, y_pred_classes)

    return -qwk

study = optuna.create_study(
    study_name="IterativeImputer with ExtraTree",
    storage="sqlite:///optuna_study_5.db",
    direction="minimize"
)
with joblib.parallel_backend('loky', n_jobs=-1):
    study.optimize(objective, n_trials=100)

best_trial = study.best_trial
print("Best trial:")
print(f"  Value (negative QWK): {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

with open(CONFIGS) as f:
    config = json.load(f)
for param in best_trial.params:
    if param.startswith('lgbm_'):
        key = param.replace('lgbm_', '')
        value = best_trial.params[param]
        config['lgbm'][key] = int(value) if key in ['max_depth', 'num_leaves', 'min_data_in_leaf', 'bagging_freq', 'n_estimators'] else value
    elif param.startswith('xgb_'):
        key = param.replace('xgb_', '')
        value = best_trial.params[param]
        config['xgb'][key] = int(value) if key in ['max_depth', 'n_estimators'] else value
    elif param.startswith('cat_'):
        key = param.replace('cat_', '')
        value = best_trial.params[param]
        config['catboost'][key] = int(value) if key in ['depth', 'iterations'] else value
    elif param.startswith('elastic_net_'):
        key = param.replace('elastic_net_', '')
        value = best_trial.params[param]
        config['elastic_net'][key] = value  
    elif param.startswith('hist_gbr_'):
        key = param.replace('hist_gbr_', '')
        value = best_trial.params[param]
        config['hist_gbr'][key] = int(value) if key in ['max_iter', 'max_depth', 'min_samples_leaf'] else value
        

config['catboost'].pop('reg_lambda', None)

with open(CONFIGS, 'w') as f:
    json.dump(config, f, indent=4)
print("Updated configuration saved.")

print("Final Model Configuration:")
print(json.dumps(config, indent=4))
trainer = TrainML(train_csv=TRAIN_FINAL, test_csv=TEST_FINAL, configs=config)
test_predictions, optimized_thresholds, oof_predictions = trainer.train_model(random_state=42)
# Save the submission file
trainer.save_submission(submission_csv=SAMPLE_SUBMISSION_CSV, output_path=OUTPUT_SUBMISSION_CSV)
print("Submission file saved.")