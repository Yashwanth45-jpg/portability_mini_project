import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import numpy as np

print("="*70)
print("📊 FINAL PERFECTION METRICS")
print("="*70)

# Load optimized data
df = pd.read_csv("water_optimized.csv")

# ==========================================
# 1. CLASS BALANCE CHECK
# ==========================================

print("\n" + "="*70)
print("⚖️  CLASS BALANCE CHECK")
print("="*70)

print("\n🔢 Class Distribution:")
print(df["Potability"].value_counts())
print("\n📊 Percentage Distribution:")
print(df["Potability"].value_counts(normalize=True) * 100)

class_counts = df["Potability"].value_counts()
imbalance_ratio = class_counts.max() / class_counts.min()

print(f"\n📈 Imbalance Ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("\n⚠️  RECOMMENDATION: Use SMOTE for better balance")
    print("   (Currently using class_weight='balanced' instead)")
else:
    print("\n✅ Classes are reasonably balanced")

# ==========================================
# 2. PREPARE DATA
# ==========================================

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Dataset Split:")
print(f"   Total samples: {len(df)}")
print(f"   Training: {len(X_train)}")
print(f"   Test: {len(X_test)}")

# ==========================================
# 3. RANDOM FOREST METRICS
# ==========================================

print("\n" + "="*70)
print("🌲 RANDOM FOREST - DETAILED METRICS")
print("="*70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)
y_test_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Calculate metrics
train_acc_rf = accuracy_score(y_train, y_train_pred_rf)
test_acc_rf = accuracy_score(y_test, y_test_pred_rf)
train_f1_rf = f1_score(y_train, y_train_pred_rf)
test_f1_rf = f1_score(y_test, y_test_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_test_proba_rf)

print("\n📈 ACCURACY:")
print(f"   Training Accuracy: {train_acc_rf:.4f}")
print(f"   Test Accuracy:     {test_acc_rf:.4f}")
print(f"   Gap (Train - Test): {train_acc_rf - test_acc_rf:.4f}")

if train_acc_rf - test_acc_rf > 0.1:
    print("   ⚠️  OVERFITTING DETECTED!")
elif train_acc_rf - test_acc_rf < -0.05:
    print("   ⚠️  UNDERFITTING!")
else:
    print("   ✅ GOOD GENERALIZATION")

print(f"\n🎯 F1-SCORE:")
print(f"   Training F1: {train_f1_rf:.4f}")
print(f"   Test F1:     {test_f1_rf:.4f}")

print(f"\n📊 ROC-AUC Score: {roc_auc_rf:.4f}")

print("\n📋 Classification Report (Test Set):")
print("-" * 70)
print(classification_report(y_test, y_test_pred_rf, 
                          target_names=['Non-Potable', 'Potable']))

# ==========================================
# 4. XGBOOST METRICS
# ==========================================

print("\n" + "="*70)
print("🚀 XGBOOST - DETAILED METRICS")
print("="*70)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# Predictions
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)
y_test_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Calculate metrics
train_acc_xgb = accuracy_score(y_train, y_train_pred_xgb)
test_acc_xgb = accuracy_score(y_test, y_test_pred_xgb)
train_f1_xgb = f1_score(y_train, y_train_pred_xgb)
test_f1_xgb = f1_score(y_test, y_test_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_test_proba_xgb)

print("\n📈 ACCURACY:")
print(f"   Training Accuracy: {train_acc_xgb:.4f}")
print(f"   Test Accuracy:     {test_acc_xgb:.4f}")
print(f"   Gap (Train - Test): {train_acc_xgb - test_acc_xgb:.4f}")

if train_acc_xgb - test_acc_xgb > 0.1:
    print("   ⚠️  OVERFITTING DETECTED!")
elif train_acc_xgb - test_acc_xgb < -0.05:
    print("   ⚠️  UNDERFITTING!")
else:
    print("   ✅ GOOD GENERALIZATION")

print(f"\n🎯 F1-SCORE:")
print(f"   Training F1: {train_f1_xgb:.4f}")
print(f"   Test F1:     {test_f1_xgb:.4f}")

print(f"\n📊 ROC-AUC Score: {roc_auc_xgb:.4f}")

print("\n📋 Classification Report (Test Set):")
print("-" * 70)
print(classification_report(y_test, y_test_pred_xgb, 
                          target_names=['Non-Potable', 'Potable']))

# ==========================================
# 5. MODEL COMPARISON TABLE
# ==========================================

print("\n" + "="*70)
print("🏆 COMPREHENSIVE MODEL COMPARISON")
print("="*70)

print("\n" + "="*70)
print(" METRIC COMPARISON TABLE")
print("="*70)
print(f"{'Metric':<25} {'Random Forest':<20} {'XGBoost':<20}")
print("="*70)
print(f"{'Training Accuracy':<25} {train_acc_rf:.4f}{' '*14} {train_acc_xgb:.4f}")
print(f"{'Test Accuracy':<25} {test_acc_rf:.4f}{' '*14} {test_acc_xgb:.4f}")
print(f"{'Overfitting Gap':<25} {train_acc_rf - test_acc_rf:+.4f}{' '*14} {train_acc_xgb - test_acc_xgb:+.4f}")
print(f"{'Training F1-Score':<25} {train_f1_rf:.4f}{' '*14} {train_f1_xgb:.4f}")
print(f"{'Test F1-Score':<25} {test_f1_rf:.4f}{' '*14} {test_f1_xgb:.4f}")
print(f"{'ROC-AUC Score':<25} {roc_auc_rf:.4f}{' '*14} {roc_auc_xgb:.4f}")
print("="*70)

# Determine best model
if roc_auc_rf > roc_auc_xgb:
    winner = "Random Forest"
    winner_roc = roc_auc_rf
elif roc_auc_xgb > roc_auc_rf:
    winner = "XGBoost"
    winner_roc = roc_auc_xgb
else:
    winner = "Random Forest" if test_acc_rf >= test_acc_xgb else "XGBoost"
    winner_roc = roc_auc_rf if test_acc_rf >= test_acc_xgb else roc_auc_xgb

print(f"\n🥇 WINNER: {winner}")
print(f"   Best ROC-AUC: {winner_roc:.4f}")

# ==========================================
# 6. FINAL SUMMARY
# ==========================================

print("\n" + "="*70)
print("✨ FINAL SUMMARY - PERFECTION CHECK")
print("="*70)

print("\n✅ CHECKLIST:")
print(f"   [{'✓' if imbalance_ratio <= 3 else '⚠'}] Class Balance: {imbalance_ratio:.2f}:1")
print(f"   [{'✓' if abs(train_acc_rf - test_acc_rf) <= 0.1 else '✗'}] RF No Overfitting: Gap = {train_acc_rf - test_acc_rf:.4f}")
print(f"   [{'✓' if abs(train_acc_xgb - test_acc_xgb) <= 0.1 else '✗'}] XGB No Overfitting: Gap = {train_acc_xgb - test_acc_xgb:.4f}")
print(f"   [{'✓' if test_acc_rf >= 0.9 else '✗'}] RF Test Accuracy: {test_acc_rf:.4f}")
print(f"   [{'✓' if test_acc_xgb >= 0.9 else '✗'}] XGB Test Accuracy: {test_acc_xgb:.4f}")
print(f"   [{'✓' if roc_auc_rf >= 0.9 else '✗'}] RF ROC-AUC: {roc_auc_rf:.4f}")
print(f"   [{'✓' if roc_auc_xgb >= 0.9 else '✗'}] XGB ROC-AUC: {roc_auc_xgb:.4f}")

print("\n🎯 QUICK COPY-PASTE ANSWERS:")
print("="*70)
print("\n📊 RANDOM FOREST:")
print(f'   print("Accuracy:", {test_acc_rf:.4f})')
print(f'   print("F1:", {test_f1_rf:.4f})')
print(f'   print("ROC-AUC:", {roc_auc_rf:.4f})')
print(f'\n   Training accuracy = {train_acc_rf:.4f}')
print(f'   Test accuracy = {test_acc_rf:.4f}')
print(f'   Overfitting gap = {train_acc_rf - test_acc_rf:.4f}')

print("\n📊 XGBOOST:")
print(f'   print("Accuracy:", {test_acc_xgb:.4f})')
print(f'   print("F1:", {test_f1_xgb:.4f})')
print(f'   print("ROC-AUC:", {roc_auc_xgb:.4f})')
print(f'\n   Training accuracy = {train_acc_xgb:.4f}')
print(f'   Test accuracy = {test_acc_xgb:.4f}')
print(f'   Overfitting gap = {train_acc_xgb - test_acc_xgb:.4f}')

print("\n📊 CLASS BALANCE:")
print(f'   Potable (1):     {class_counts.get(1, 0)} samples ({class_counts.get(1, 0)/len(df)*100:.1f}%)')
print(f'   Non-Potable (0): {class_counts.get(0, 0)} samples ({class_counts.get(0, 0)/len(df)*100:.1f}%)')
print(f'   Imbalance Ratio: {imbalance_ratio:.2f}:1')

if imbalance_ratio > 3:
    print("\n💡 SMOTE RECOMMENDATION:")
    print("   from imblearn.over_sampling import SMOTE")
    print("   smote = SMOTE(random_state=42)")
    print("   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)")
else:
    print("\n✅ No SMOTE needed - using class_weight='balanced'")

print("\n" + "="*70)
print("🔥 ANALYSIS COMPLETE - ALL METRICS EXTRACTED!")
print("="*70)
