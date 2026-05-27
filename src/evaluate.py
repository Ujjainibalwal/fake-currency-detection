"""
evaluate.py
Model evaluation utilities — confusion matrix, classification report, ROC curve.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

# Output directory for visualizations
VIZ_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)


def print_metrics(y_true, y_pred, y_proba=None):
    """
    Print all evaluation metrics.
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
    """
    print("=" * 60)
    print("📈 MODEL EVALUATION RESULTS")
    print("=" * 60)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n   Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"   Precision : {prec:.4f}  ({prec*100:.2f}%)")
    print(f"   Recall    : {rec:.4f}  ({rec*100:.2f}%)")
    print(f"   F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")

    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"   ROC-AUC   : {auc:.4f}")

    print(f"\n📋 Classification Report:\n")
    report = classification_report(y_true, y_pred, target_names=["Genuine (0)", "Forged (1)"])
    print(report)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, save=True):
    """
    Plot and save the confusion matrix.
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        save (bool): Whether to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine", "Forged"],
        yticklabels=["Genuine", "Forged"],
        annot_kws={"size": 18, "fontweight": "bold"},
        linewidths=2,
        linecolor="white",
    )
    plt.title("Confusion Matrix", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=14, labelpad=10)
    plt.ylabel("True Label", fontsize=14, labelpad=10)
    plt.tight_layout()

    if save:
        path = os.path.join(VIZ_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   💾 Confusion matrix saved to: {path}")
    plt.close()

    # Print confusion matrix breakdown
    tn, fp, fn, tp = cm.ravel()
    print(f"\n   Confusion Matrix Breakdown:")
    print(f"   ┌─────────────────┬───────────┬──────────┐")
    print(f"   │                 │ Pred Gen. │ Pred For.│")
    print(f"   ├─────────────────┼───────────┼──────────┤")
    print(f"   │ Actual Genuine  │  TN={tn:<4}  │  FP={fp:<4} │")
    print(f"   │ Actual Forged   │  FN={fn:<4}  │  TP={tp:<4} │")
    print(f"   └─────────────────┴───────────┴──────────┘")


def plot_roc_curve(y_true, y_proba, save=True):
    """
    Plot and save the ROC curve.
    
    Parameters:
        y_true: True labels
        y_proba: Predicted probabilities for the positive class
        save (bool): Whether to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Logistic Regression (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier")
    plt.fill_between(fpr, tpr, alpha=0.15, color="#2563eb")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14, labelpad=10)
    plt.ylabel("True Positive Rate", fontsize=14, labelpad=10)
    plt.title("ROC Curve — Fake Currency Detection", fontsize=16, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(VIZ_DIR, "roc_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   💾 ROC curve saved to: {path}")
    plt.close()


def plot_feature_distributions(df, save=True):
    """
    Plot feature distributions for genuine vs forged banknotes.
    
    Parameters:
        df (pd.DataFrame): The full dataset with 'class' column
        save (bool): Whether to save the plot
    """
    features = ["variance", "skewness", "kurtosis", "entropy"]
    colors = {"Genuine": "#22c55e", "Forged": "#ef4444"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Feature Distributions: Genuine vs Forged", fontsize=18, fontweight="bold", y=1.02)

    for idx, (feature, ax) in enumerate(zip(features, axes.ravel())):
        genuine = df[df["class"] == 0][feature]
        forged = df[df["class"] == 1][feature]

        ax.hist(genuine, bins=30, alpha=0.6, color=colors["Genuine"], label="Genuine", edgecolor="white")
        ax.hist(forged, bins=30, alpha=0.6, color=colors["Forged"], label="Forged", edgecolor="white")

        ax.set_title(f"{feature.capitalize()}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Value", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save:
        path = os.path.join(VIZ_DIR, "feature_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   💾 Feature distributions saved to: {path}")
    plt.close()


def plot_correlation_heatmap(df, save=True):
    """
    Plot a correlation heatmap of the features.
    
    Parameters:
        df (pd.DataFrame): The full dataset
        save (bool): Whether to save the plot
    """
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        mask=mask,
        square=True,
        linewidths=2,
        linecolor="white",
        annot_kws={"size": 12},
    )
    plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()

    if save:
        path = os.path.join(VIZ_DIR, "correlation_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   💾 Correlation heatmap saved to: {path}")
    plt.close()


def generate_all_visualizations(df, y_true, y_pred, y_proba=None):
    """Generate all evaluation visualizations."""
    print("\n" + "=" * 60)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 60 + "\n")

    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_confusion_matrix(y_true, y_pred)

    if y_proba is not None:
        plot_roc_curve(y_true, y_proba)

    print(f"\n   ✅ All visualizations saved to: {VIZ_DIR}/")
