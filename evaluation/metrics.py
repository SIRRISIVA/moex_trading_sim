import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import spearmanr

def compute_drawdown(nav):
    peak = np.maximum.accumulate(nav)
    return (nav - peak) / peak

def total_return(nav):
    return nav[-1] / nav[0] - 1

def sharpe(nav):
    returns = np.diff(nav) / nav[:-1]
    return np.mean(returns) / (np.std(returns) + 1e-8)

def evaluate_predictions(y_true, y_probs, real_returns):
    # 1. Классические метрики
    y_pred = (y_probs > 0.5).astype(int)
    print("--- Classification Report ---")
    print(classification_report(y_true, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_true, y_probs):.4f}")

    # 2. Information Coefficient (IC)
    # Корреляция Спирмена между вероятностью роста и реальной доходностью
    ic, _ = spearmanr(y_probs, real_returns)
    print(f"Information Coefficient (IC): {ic:.4f}")

    return ic