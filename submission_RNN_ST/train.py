import pandas as pd
import json
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
import numpy as np
from tqdm import tqdm

class TrainML:
    def __init__(self, train_csv: str, test_csv: str, configs: str):
        """
        Initialize the TrainML class.

        Parameters:
        - train_csv: Path to the CSV containing the training dataset.
        - test_csv: Path to the CSV containing the testing dataset.
        - configs: Path to the JSON file containing model configuration parameters.
        """
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
        with open(configs) as f:
            self.configs = json.load(f)
        
        # Initialize models with parameters from the config file
        self.light = LGBMRegressor(**self.configs['lgbm'])
        self.xgb = XGBRegressor(**self.configs['xgb'])
        self.cat = CatBoostRegressor(**self.configs['catboost'], verbose=0)
        
        # Create a Voting Regressor ensemble
        self.voting_model = VotingRegressor([
            ('lgbm', self.light),
            ('xgb', self.xgb),
            ('cat', self.cat),
        ])

    def quadratic_weighted_kappa(self, y_true, y_pred):
        """
        Compute the Quadratic Weighted Kappa (QWK) score.

        Parameters:
        - y_true: True target values.
        - y_pred: Predicted target values.

        Returns:
        - QWK score.
        """
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')

    def threshold_rounder(self, y_pred, thresholds):
        """
        Apply thresholds to continuous predictions to obtain discrete classes.

        Parameters:
        - y_pred: Continuous predictions.
        - thresholds: List of threshold values.

        Returns:
        - Discrete class predictions.
        """
        return np.where(y_pred < thresholds[0], 0,
                        np.where(y_pred < thresholds[1], 1,
                                 np.where(y_pred < thresholds[2], 2, 3)))

    def evaluate_predictions(self, thresholds, y_true, y_pred):
        """
        Evaluate predictions using the negative QWK score.

        Parameters:
        - thresholds: List of threshold values.
        - y_true: True target values.
        - y_pred: Continuous predictions.

        Returns:
        - Negative QWK score.
        """
        y_pred_classes = self.threshold_rounder(y_pred, thresholds)
        return -self.quadratic_weighted_kappa(y_true, y_pred_classes)

    def train_model(self, n_splits=5, random_state=42):
        """
        Train the ensemble model using cross-validation and optimize thresholds.

        Parameters:
        - n_splits: Number of cross-validation splits.
        - random_state: Random state for reproducibility.

        Returns:
        - test_predictions: Final predictions for the test dataset.
        - optimized_thresholds: Optimized threshold values.
        - oof_predictions: Out-of-fold predictions for the training dataset.
        """
        # Separate features and target variable
        X = self.train_df.drop(columns=['sii', 'id'])
        y = self.train_df['sii']
        X_test = self.test_df.drop(columns=['id'])

        # Initialize variables to store predictions
        oof_predictions = np.zeros(len(y))
        test_predictions = np.zeros(len(self.test_df))
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X, y), total=n_splits, desc='Training Folds')):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Clone the model to ensure each fold is independent
            model = clone(self.voting_model)
            model.fit(X_train, y_train)

            # Predict on validation set and test set
            val_preds = model.predict(X_val)
            test_fold_preds = model.predict(X_test)

            # Store predictions
            oof_predictions[val_idx] = val_preds
            test_predictions += test_fold_preds / n_splits  # Average over folds

            # Compute and print QWK for the fold
            val_preds_rounded = val_preds.round().astype(int)
            fold_qwk = self.quadratic_weighted_kappa(y_val, val_preds_rounded)
            print(f"Fold {fold + 1} QWK: {fold_qwk:.4f}")

        # Optimize thresholds on out-of-fold predictions
        initial_thresholds = [0.5, 1.5, 2.5]
        optimization_result = minimize(
            self.evaluate_predictions,
            x0=initial_thresholds,
            args=(y, oof_predictions),
            method='Nelder-Mead'
        )

        optimized_thresholds = optimization_result.x
        print(f"Optimized thresholds: {optimized_thresholds}")

        # Apply optimized thresholds to training predictions
        oof_pred_classes = self.threshold_rounder(oof_predictions, optimized_thresholds)
        final_qwk = self.quadratic_weighted_kappa(y, oof_pred_classes)
        print(f"Final QWK on training data: {final_qwk:.4f}")

        # Apply optimized thresholds to test predictions
        test_pred_classes = self.threshold_rounder(test_predictions, optimized_thresholds)
        self.test_df['sii'] = test_pred_classes.astype(int)

        return test_predictions, optimized_thresholds, oof_predictions

    def save_submission(self, submission_csv: str, output_path: str):
        """
        Save the predictions to a CSV file in the required submission format.

        Parameters:
        - submission_csv: Path to the sample submission CSV file.
        - output_path: Path to save the final submission CSV.
        """
        submission = pd.read_csv(submission_csv)
        submission['sii'] = self.test_df['sii']
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
