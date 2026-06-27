"""
=============================================================================
CNC Machine Remaining Useful Life (RUL) Prediction
=============================================================================
Author      : Vichu (Vishva Venkat)
Description : Production-ready ML pipeline to predict RUL of CNC machines
              using Random Forest Regressor with hyperparameter tuning,
              full evaluation metrics, feature importance, and model I/O.
Dataset     : cnc_rul_dataset.csv
Target      : RUL
Features    : Temperature, Vibration, Current, RPM
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH   = "cnc_rul_dataset.csv"
MODEL_PATH     = "cnc_rul_model.pkl"
SCALER_PATH    = "cnc_rul_scaler.pkl"
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
CV_FOLDS       = 5
N_ITER_SEARCH  = 50           # number of RandomizedSearchCV iterations
FEATURES       = ["Temperature", "Vibration", "Current", "RPM"]
TARGET         = "RUL"
IQR_MULTIPLIER = 1.5          # outlier removal threshold


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the CSV dataset from the given filepath.

    Parameters
    ----------
    filepath : str
        Path to the cnc_rul_dataset.csv file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame loaded from the CSV.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If required columns are missing from the dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please place 'cnc_rul_dataset.csv' in the working directory."
        )

    logger.info("Loading dataset from '%s' …", filepath)
    df = pd.read_csv(filepath)

    # Validate required columns
    required_cols = FEATURES + [TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing from the dataset: {missing}"
        )

    logger.info("Dataset loaded successfully. Shape: %s", df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with the column median for numeric columns.

    Using median instead of mean makes the imputation robust to outliers.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled.
    """
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.info("Found %d missing value(s). Filling with column medians …", missing_count)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    else:
        logger.info("No missing values detected.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    removed = n_before - len(df)
    if removed:
        logger.info("Removed %d duplicate row(s).", removed)
    else:
        logger.info("No duplicate rows found.")
    return df.reset_index(drop=True)


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Remove rows where any value in the specified columns is an outlier,
    using the IQR (Interquartile Range) method.

    A value is an outlier if it falls below Q1 - 1.5*IQR
    or above Q3 + 1.5*IQR.

    Parameters
    ----------
    df      : pd.DataFrame
    columns : list of str
        Columns on which to apply the IQR filter.

    Returns
    -------
    pd.DataFrame
        DataFrame with outlier rows removed.
    """
    n_before = len(df)
    mask = pd.Series([True] * len(df), index=df.index)

    for col in columns:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        mask &= df[col].between(lower, upper)

    df_clean = df[mask].reset_index(drop=True)
    removed  = n_before - len(df_clean)
    logger.info("Outlier removal (IQR): removed %d row(s). Remaining: %d", removed, len(df_clean))
    return df_clean


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Handle missing values
      2. Remove duplicates
      3. Remove outliers

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Clean, preprocessed DataFrame.
    """
    logger.info("─── Starting preprocessing ───")
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_outliers_iqr(df, FEATURES + [TARGET])
    logger.info("─── Preprocessing complete. Final shape: %s ───", df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING & SPLITTING
# ─────────────────────────────────────────────────────────────────────────────
def prepare_data(df: pd.DataFrame):
    """
    Extract features and target, apply StandardScaler, and perform
    an 80:20 train-test split.

    Notes
    -----
    StandardScaler is fitted ONLY on training data to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple
        (X_train_sc, X_test_sc, y_train, y_test, scaler)
    """
    X = df[FEATURES].values
    y = df[TARGET].values

    # Train-test split (stratification not applicable for regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(
        "Train size: %d  |  Test size: %d", len(X_train), len(X_test)
    )

    # Feature scaling – fit on train, transform both
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL TRAINING WITH HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
def tune_and_train(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """
    Tune a Random Forest Regressor with RandomizedSearchCV (5-fold CV)
    and return the best estimator.

    Hyperparameter search space
    ---------------------------
    n_estimators      : number of trees in the forest
    max_depth         : maximum depth of each tree
    min_samples_split : minimum samples required to split an internal node
    min_samples_leaf  : minimum samples required at a leaf node
    max_features      : number of features to consider at each split
    bootstrap         : whether bootstrap samples are used

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray

    Returns
    -------
    RandomForestRegressor
        Best estimator found by RandomizedSearchCV.
    """
    logger.info("─── Starting hyperparameter tuning (RandomizedSearchCV) ───")

    param_dist = {
        "n_estimators"     : [100, 200, 300, 400, 500],
        "max_depth"        : [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
        "max_features"     : ["sqrt", "log2", None, 0.5],
        "bootstrap"        : [True, False],
    }

    base_rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    random_search = RandomizedSearchCV(
        estimator          = base_rf,
        param_distributions= param_dist,
        n_iter             = N_ITER_SEARCH,
        cv                 = CV_FOLDS,
        scoring            = "neg_mean_absolute_error",
        n_jobs             = -1,
        verbose            = 1,
        random_state       = RANDOM_STATE,
        refit              = True,
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    logger.info("Best hyperparameters found:\n%s", best_params)
    logger.info(
        "Best CV MAE (neg): %.4f", random_search.best_score_
    )

    return random_search.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CROSS-VALIDATION ON TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────
def cross_validate_model(model: RandomForestRegressor, X_train: np.ndarray, y_train: np.ndarray):
    """
    Perform k-fold cross-validation on the training set and log the results.

    Parameters
    ----------
    model   : RandomForestRegressor
    X_train : np.ndarray
    y_train : np.ndarray
    """
    logger.info("─── Cross-validation on training data (%d folds) ───", CV_FOLDS)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    logger.info(
        "CV R² scores: %s  |  Mean: %.4f  |  Std: %.4f",
        np.round(cv_scores, 4), cv_scores.mean(), cv_scores.std()
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute MAPE, guarding against division by zero.

    MAPE = mean(|y_true - y_pred| / max(|y_true|, epsilon)) * 100

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Returns
    -------
    float
        MAPE value in percentage (%).
    """
    epsilon = np.finfo(np.float64).eps          # tiny number to avoid /0
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100


def evaluate_model(model: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the trained model on the test set and print all metrics.

    Metrics computed
    ----------------
    MAE  : Mean Absolute Error
    MSE  : Mean Squared Error
    RMSE : Root Mean Squared Error
    R²   : Coefficient of Determination
    MAPE : Mean Absolute Percentage Error (%)

    Parameters
    ----------
    model  : RandomForestRegressor
    X_test : np.ndarray
    y_test : np.ndarray

    Returns
    -------
    dict
        Dictionary containing metric names and their values.
    """
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}

    divider = "=" * 50
    logger.info("\n%s\n  MODEL EVALUATION RESULTS\n%s", divider, divider)
    logger.info("  MAE   : %.4f", mae)
    logger.info("  MSE   : %.4f", mse)
    logger.info("  RMSE  : %.4f", rmse)
    logger.info("  R²    : %.4f", r2)
    logger.info("  MAPE  : %.4f %%", mape)
    logger.info("%s", divider)

    return y_pred, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7.  VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model: RandomForestRegressor):
    """
    Plot and display a horizontal bar chart of feature importances
    (Gini impurity-based) from the trained Random Forest.

    Parameters
    ----------
    model : RandomForestRegressor
    """
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_feats = [FEATURES[i] for i in indices]
    sorted_imps  = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("Blues_r", len(FEATURES))
    ax.barh(sorted_feats[::-1], sorted_imps[::-1], color=colors)
    ax.set_xlabel("Importance (Gini)", fontsize=12)
    ax.set_title("Feature Importance – Random Forest", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)

    for i, (feat, imp) in enumerate(zip(sorted_feats[::-1], sorted_imps[::-1])):
        ax.text(imp + 0.002, i, f"{imp:.4f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    logger.info("Feature importance plot saved as 'feature_importance.png'.")
    plt.show()


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray):
    """
    Plot Actual vs Predicted RUL values along with a residuals histogram.

    Parameters
    ----------
    y_test : np.ndarray  – true RUL values
    y_pred : np.ndarray  – predicted RUL values
    """
    residuals = y_test - y_pred

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # ── Scatter: Actual vs Predicted ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, y_pred, alpha=0.55, edgecolors="white",
                linewidths=0.5, color="#1f77b4", s=40)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax1.set_xlabel("Actual RUL", fontsize=12)
    ax1.set_ylabel("Predicted RUL", fontsize=12)
    ax1.set_title("Actual vs Predicted RUL", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # ── Histogram: Residuals ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=30, color="#2ca02c", edgecolor="white", alpha=0.8)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    ax2.set_xlabel("Residual (Actual − Predicted)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Residuals Distribution", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    logger.info("Actual vs Predicted plot saved as 'actual_vs_predicted.png'.")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model: RandomForestRegressor, scaler: StandardScaler):
    """
    Serialize and save the trained model and scaler using joblib.

    Parameters
    ----------
    model  : RandomForestRegressor
    scaler : StandardScaler
    """
    joblib.dump(model,  MODEL_PATH,  compress=3)
    joblib.dump(scaler, SCALER_PATH, compress=3)
    logger.info("Model saved  → '%s'", MODEL_PATH)
    logger.info("Scaler saved → '%s'", SCALER_PATH)


def load_model():
    """
    Load a previously saved model and scaler from disk.

    Returns
    -------
    tuple
        (model, scaler)

    Raises
    ------
    FileNotFoundError
        If the model or scaler files are not found.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: '{MODEL_PATH}'")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: '{SCALER_PATH}'")

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully from disk.")
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PREDICTION INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
def predict_rul(
    temperature : float,
    vibration   : float,
    current     : float,
    rpm         : float,
    model       : RandomForestRegressor = None,
    scaler      : StandardScaler        = None,
) -> float:
    """
    Predict the Remaining Useful Life (RUL) for a single CNC machine reading.

    If no model / scaler is provided, the function attempts to load them
    from disk (MODEL_PATH / SCALER_PATH).

    Parameters
    ----------
    temperature : float  – sensor temperature (°C)
    vibration   : float  – vibration amplitude (mm/s or g)
    current     : float  – electrical current (A)
    rpm         : float  – spindle speed (RPM)
    model       : trained RandomForestRegressor  (optional)
    scaler      : fitted StandardScaler          (optional)

    Returns
    -------
    float
        Predicted RUL in the same unit as the training target.

    Raises
    ------
    ValueError
        If any input value is non-finite (NaN or Inf).
    """
    # Validate inputs
    inputs = {"Temperature": temperature, "Vibration": vibration,
              "Current": current, "RPM": rpm}
    for name, val in inputs.items():
        if not np.isfinite(val):
            raise ValueError(f"Input '{name}' must be a finite number. Got: {val}")

    # Load from disk if not supplied
    if model is None or scaler is None:
        model, scaler = load_model()

    # Prepare input array
    X_new = np.array([[temperature, vibration, current, rpm]], dtype=np.float64)
    X_scaled = scaler.transform(X_new)

    predicted_rul = float(model.predict(X_scaled)[0])
    predicted_rul = max(0.0, predicted_rul)   # RUL cannot be negative

    return predicted_rul


# ─────────────────────────────────────────────────────────────────────────────
# 10.  HEALTH STATUS CLASSIFIER (BONUS)
# ─────────────────────────────────────────────────────────────────────────────
def get_health_status(rul: float) -> str:
    """
    Map a predicted RUL value to a human-readable machine health tier.

    Tiers
    -----
    CRITICAL  : RUL ≤ 20   → Immediate maintenance required
    WARNING   : 20 < RUL ≤ 60  → Schedule maintenance soon
    MODERATE  : 60 < RUL ≤ 120 → Monitor closely
    GOOD      : RUL > 120  → Operating normally

    Parameters
    ----------
    rul : float

    Returns
    -------
    str
    """
    if rul <= 20:
        return "🔴 CRITICAL  – Immediate maintenance required!"
    elif rul <= 60:
        return "🟠 WARNING   – Schedule maintenance soon."
    elif rul <= 120:
        return "🟡 MODERATE  – Monitor closely."
    else:
        return "🟢 GOOD      – Machine operating normally."


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """
    Orchestrates the end-to-end ML pipeline:
      1. Load dataset
      2. Preprocess data
      3. Split data
      4. Tune & train Random Forest
      5. Cross-validate
      6. Evaluate on test set
      7. Visualise results
      8. Save model
      9. Demonstrate loading & prediction
    """
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   CNC Machine RUL Prediction – ML Pipeline       ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    # ── 1. Load ───────────────────────────────────────────────────────────
    df = load_dataset(DATASET_PATH)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    df_clean = preprocess(df)

    # ── 3. Split ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler = prepare_data(df_clean)

    # ── 4. Tune & Train ───────────────────────────────────────────────────
    best_model = tune_and_train(X_train, y_train)

    # ── 5. Cross-validate ─────────────────────────────────────────────────
    cross_validate_model(best_model, X_train, y_train)

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    y_pred, metrics = evaluate_model(best_model, X_test, y_test)

    # ── 7. Visualise ─────────────────────────────────────────────────────
    plot_feature_importance(best_model)
    plot_actual_vs_predicted(y_test, y_pred)

    # ── 8. Save ───────────────────────────────────────────────────────────
    save_model(best_model, scaler)

    # ── 9. Demonstrate loading & prediction ──────────────────────────────
    logger.info("\n─── Live Prediction Demo (loading model from disk) ───")
    loaded_model, loaded_scaler = load_model()

    # Example sensor readings
    demo_inputs = [
        (75.2, 0.85, 12.4, 3500),   # Moderate condition
        (95.1, 1.95, 18.7, 4800),   # Stressed condition
        (55.0, 0.40, 8.2,  2200),   # Healthy condition
    ]

    print("\n" + "=" * 60)
    print("  SAMPLE PREDICTIONS")
    print("=" * 60)
    print(f"{'Temp':>6} {'Vib':>6} {'Curr':>6} {'RPM':>6} │ {'Pred RUL':>10}  Status")
    print("-" * 60)

    for temp, vib, curr, rpm in demo_inputs:
        rul    = predict_rul(temp, vib, curr, rpm, loaded_model, loaded_scaler)
        status = get_health_status(rul)
        print(f"{temp:>6.1f} {vib:>6.2f} {curr:>6.1f} {rpm:>6.0f} │ {rul:>10.2f}  {status}")

    print("=" * 60)
    logger.info("Pipeline completed successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        logger.error("File error: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Data error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)
