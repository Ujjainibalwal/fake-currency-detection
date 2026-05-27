"""
predict.py
Predict whether a banknote is genuine or forged using the trained model.

Usage:
    python src/predict.py --variance 2.3 --skewness 4.5 --kurtosis -0.8 --entropy -1.2

    Or use interactive mode:
    python src/predict.py --interactive
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "logistic_regression_model.pkl")

# Feature names matching the training data
FEATURE_NAMES = ["variance", "skewness", "kurtosis", "entropy"]


def load_model(filepath=None):
    """Load the trained model and scaler from disk."""
    if filepath is None:
        filepath = MODEL_PATH

    if not os.path.exists(filepath):
        print(f"❌ Model not found at: {filepath}")
        print("   Please train the model first by running: python src/model.py")
        sys.exit(1)

    model_data = joblib.load(filepath)
    return model_data["model"], model_data["scaler"]


def predict_currency(features, model=None, scaler=None):
    """
    Predict whether a banknote is genuine or forged.
    
    Parameters:
        features (list): [variance, skewness, kurtosis, entropy]
        model: Trained LogisticRegression model (loads from disk if None)
        scaler: Fitted StandardScaler (loads from disk if None)
    
    Returns:
        dict: Prediction result with label, probability, and confidence
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    # Prepare input as DataFrame with proper feature names (avoids sklearn warnings)
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)

    # Scale features using the same scaler used during training
    features_scaled = pd.DataFrame(
        scaler.transform(features_df), columns=FEATURE_NAMES
    )

    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    result = {
        "prediction": int(prediction),
        "label": "FORGED (Fake) 🔴" if prediction == 1 else "GENUINE (Real) 🟢",
        "probability_genuine": float(probabilities[0]),
        "probability_forged": float(probabilities[1]),
        "confidence": float(max(probabilities)) * 100,
    }

    return result


def print_prediction(features, result):
    """Print the prediction result in a formatted way."""
    print("\n" + "=" * 60)
    print("🔮 BANKNOTE PREDICTION RESULT")
    print("=" * 60)

    print(f"\n   📥 Input Features:")
    names = ["Variance", "Skewness", "Kurtosis", "Entropy"]
    for name, val in zip(names, features):
        print(f"      {name:>10}: {val:.4f}")

    print(f"\n   {'━' * 40}")
    print(f"\n   🏷️  Prediction : {result['label']}")
    print(f"   📊 Confidence : {result['confidence']:.2f}%")
    print(f"\n   📈 Probabilities:")
    print(f"      Genuine : {result['probability_genuine']*100:.2f}%")
    print(f"      Forged  : {result['probability_forged']*100:.2f}%")
    print(f"\n   {'━' * 40}")

    if result["prediction"] == 1:
        print(f"\n   ⚠️  WARNING: This banknote is likely COUNTERFEIT!")
    else:
        print(f"\n   ✅ This banknote appears to be AUTHENTIC.")
    print()


def interactive_mode():
    """Run predictions in interactive mode."""
    print("\n" + "=" * 60)
    print("🔮 FAKE CURRENCY DETECTOR — INTERACTIVE MODE")
    print("=" * 60)
    print("\n   Enter banknote features to check authenticity.")
    print("   Type 'quit' or 'q' to exit.\n")

    model, scaler = load_model()
    print("   ✅ Model loaded successfully!\n")

    while True:
        print("─" * 40)
        try:
            variance = input("   Enter Variance  : ").strip()
            if variance.lower() in ("quit", "q", "exit"):
                break
            variance = float(variance)

            skewness = float(input("   Enter Skewness  : ").strip())
            kurtosis = float(input("   Enter Kurtosis  : ").strip())
            entropy = float(input("   Enter Entropy   : ").strip())

            features = [variance, skewness, kurtosis, entropy]
            result = predict_currency(features, model, scaler)
            print_prediction(features, result)

        except ValueError:
            print("   ❌ Invalid input. Please enter numeric values.\n")
        except KeyboardInterrupt:
            print("\n\n   👋 Goodbye!")
            break

    print("\n   👋 Exiting interactive mode. Goodbye!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether a banknote is genuine or forged.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/predict.py --variance 2.3 --skewness 4.5 --kurtosis -0.8 --entropy -1.2
  python src/predict.py --interactive
  python src/predict.py -v 0.5 -s 1.2 -k -2.3 -e -0.5
        """,
    )

    parser.add_argument("--variance", "-v", type=float, help="Variance of Wavelet Transformed image")
    parser.add_argument("--skewness", "-s", type=float, help="Skewness of Wavelet Transformed image")
    parser.add_argument("--kurtosis", "-k", type=float, help="Kurtosis of Wavelet Transformed image")
    parser.add_argument("--entropy", "-e", type=float, help="Entropy of the image")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if all(v is not None for v in [args.variance, args.skewness, args.kurtosis, args.entropy]):
        features = [args.variance, args.skewness, args.kurtosis, args.entropy]
        result = predict_currency(features)
        print_prediction(features, result)
    else:
        # If no arguments provided, run with sample data
        print("\n   ℹ️  No features provided. Running with sample data...\n")

        samples = [
            ([3.6216, 8.6661, -2.8073, -0.44699], "Expected: Genuine"),
            ([-1.3971, 0.52477, 2.0149, 0.2689], "Expected: Forged"),
            ([2.3456, 4.5678, -0.8765, -1.2345], "Test sample"),
            ([-2.5419, -0.6583, 2.3359, 0.7803], "Expected: Forged"),
            ([4.5459, 8.1674, -2.4586, -1.4621], "Expected: Genuine"),
        ]

        model, scaler = load_model()

        for features, note in samples:
            result = predict_currency(features, model, scaler)
            print(f"   {note}")
            print_prediction(features, result)


if __name__ == "__main__":
    main()
