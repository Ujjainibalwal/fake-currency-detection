"""
model.py
Train and evaluate the Logistic Regression model for fake currency detection.
This is the MAIN script — run this to train and evaluate the model.

Usage:
    python src/model.py
"""

import os
import sys
import warnings
import joblib

# Add parent directory to path so imports work when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from src.data_preprocessing import load_data, explore_data, preprocess_data
from src.evaluate import print_metrics, generate_all_visualizations

warnings.filterwarnings("ignore")

# Directory to save the trained model
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model(X_train, y_train):
    """
    Train a Logistic Regression model.
    
    Parameters:
        X_train: Training features (scaled)
        y_train: Training labels
    
    Returns:
        LogisticRegression: Trained model
    """
    print("=" * 60)
    print("🏋️  TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 60)

    model = LogisticRegression(
        solver="lbfgs",       # Efficient solver for small datasets
        max_iter=1000,        # Max iterations for convergence
        random_state=42,      # Reproducibility
        C=1.0,                # Regularization strength (inverse)
    )

    model.fit(X_train, y_train)

    print(f"\n   ✅ Model trained successfully!")
    print(f"\n   📐 Model Parameters:")
    print(f"   Coefficients (weights):")

    feature_names = X_train.columns if hasattr(X_train, "columns") else [f"Feature_{i}" for i in range(X_train.shape[1])]
    for name, coef in zip(feature_names, model.coef_[0]):
        direction = "→ Higher = more likely FORGED" if coef > 0 else "→ Higher = more likely GENUINE"
        print(f"      {name:>10}: {coef:+.4f}  {direction}")

    print(f"      {'Bias':>10}: {model.intercept_[0]:+.4f}")
    print()

    return model


def save_model(model, scaler, filepath=None):
    """Save the trained model and scaler to disk."""
    if filepath is None:
        filepath = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")

    model_data = {"model": model, "scaler": scaler}
    joblib.dump(model_data, filepath)
    print(f"   💾 Model saved to: {filepath}")


def main():
    """Main pipeline: Load → Preprocess → Train → Evaluate → Save."""
    print("\n" + "🔷" * 30)
    print("  💵 FAKE CURRENCY DETECTION — LOGISTIC REGRESSION")
    print("🔷" * 30 + "\n")

    # Step 1: Check if dataset exists, if not download it
    from src.download_data import download_dataset, DATA_FILE
    if not os.path.exists(DATA_FILE):
        download_dataset()

    # Step 2: Load and explore the dataset
    df = load_data()
    explore_data(df)

    # Step 3: Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Step 4: Train the model
    model = train_model(X_train, y_train)

    # Step 5: Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of being forged

    # Step 6: Evaluate the model
    metrics = print_metrics(y_test, y_pred, y_proba)

    # Step 7: Generate visualizations
    generate_all_visualizations(df, y_test, y_pred, y_proba)

    # Step 8: Save the trained model
    print("\n" + "=" * 60)
    print("💾 SAVING MODEL")
    print("=" * 60 + "\n")
    save_model(model, scaler)

    # Final summary
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\n   🎯 Final Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   📊 Visualizations saved to: visualizations/")
    print(f"   💾 Model saved to: model/")
    print(f"\n   🔮 To predict new banknotes, run:")
    print(f"      python src/predict.py --variance 2.3 --skewness 4.5 --kurtosis -0.8 --entropy -1.2")
    print()


if __name__ == "__main__":
    main()
