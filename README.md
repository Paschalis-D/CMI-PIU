# CMI-PIU
A repository containing the submission code for the kaggle competition Child Mind Institute — Problematic Internet Use (Relating Physical Activity to Problematic Internet Use)

# Step-by-Step Plan for Building the Combined Tabular-Actigraphy Prediction Model
## Objective: Develop a model that integrates tabular and actigraphy data, manages missing values, and optimizes for the Quadratic Weighted Kappa (QWK) score.

## Phase 1: Data Preparation and Imputation
1. Impute Missing Values:
    - Apply iterative imputation methods (e.g., MICE) for complex relationships.
    - Use unsupervised learning (e.g., K-means, PCA, or autoencoders) to impute values based on data structure.
    - For missing target values, consider filtering them out or using models that can handle partial supervision.
2. Outlier Detection and Noise Reduction:
    - Identify and handle outliers in both tabular and actigraphy data, potentially using isolation forests or statistical thresholds.
    - Denoise actigraphy data (e.g., with smoothing or Fourier transformations) to retain essential patterns while reducing noise.

## Phase 2: Feature Selection and Engineering for Tabular Data
1. Correlation-Based Feature Selection:
    - Calculate correlations between features and the target variable to identify and keep only the most relevant features.
2. Feature Interaction Creation:
    - Generate new features by interacting key features with each other, such as polynomial features or multiplicative combinations.
3. Advanced Feature Selection:
    -Use methods like Lasso regression, Recursive Feature Elimination (RFE), or tree-based feature importance to refine the feature set further.
4. Feature Importance Evaluation:
    - Use models like gradient boosting (e.g., CatBoost, XGBoost) or SHAP values to validate and refine features based on their contribution to the model.

## Phase 3: Neural Network Encoding for Actigraphy Data
1. Actigraphy Data Encoding with a Neural Network:
    - Build a neural network encoder (e.g., with RNNs or LSTMs) to transform actigraphy sequences into fixed-size representations.
    - Optionally, integrate attention mechanisms to help the model focus on important time points within the sequences.
2. Combining Encoded Actigraphy with Tabular Data:
    - Concatenate the actigraphy encoding with selected tabular features, creating a unified feature set for the final model.

## Phase 4: Model Training and Optimization
1. Model Selection and Initial Training:
    - Use gradient boosting models like CatBoost, XGBoost, or LightGBM to leverage their robustness with tabular data.
    - Optionally, employ a voting regressor that combines multiple models (e.g., CatBoost, XGBoost, NN encoder) to improve generalization.
2. Custom QWK Loss Function (if feasible):
    - Adjust training to optimize directly for QWK by implementing it as a custom loss function if the model framework allows.
3. Hyperparameter Optimization:
    - Perform hyperparameter tuning for gradient boosting models and neural networks using grid search or Bayesian optimization to maximize the QWK score.
4. Cross-Validation:
    - Use K-Fold cross-validation to train the model on multiple splits of the dataset, improving robustness and reliability in performance evaluation.

## Phase 5: Model Evaluation and Submission
1. Final Model Evaluation:
    - After training, apply the model to the test dataset (if target labels are available) and calculate the QWK score to validate the model’s effectiveness.
2. Generate Submission File:
    - Using the final model configuration, generate predictions for the test set.
    - Save the predictions in the required submission format, ensuring that all necessary columns are included.
