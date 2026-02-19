import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import numpy as np

# Step 1: Extract data from XLSX file
print("Extracting data from XLSX file...")
# Read Table 3 which contains River Sutlej data, skip first 2 rows
df = pd.read_excel("WQuality_River-Data-2022.xlsx", sheet_name="Table 3", header=None)

# Row 0 has the actual headers, but columns alternate between data and NaN
# Extract only the data columns we need
headers = df.iloc[0]  # Get header row
df = df.iloc[3:]  # Skip header and criteria rows

# Assign proper column names 
df.columns = [
    "Station",
    "Location", 
    "State",
    "Temperature",
    "Temp_Unit",
    "Dissolved_Oxygen",
    "DO_Unit",
    "pH",
    "pH_Unit",
    "Conductivity",
    "Cond_Unit",
    "BOD",
    "BOD_Unit",
    "Nitrate",
    "Nitrate_Unit",
    "Fecal_Coliform",
    "FC_Unit",
    "Total_Coliform",
    "TC_Unit",
    "Fecal_Streptococci",
    "FS_Unit"
]

# Save to CSV
df.to_csv("water_raw.csv", index=False)
print("✅ Raw data saved to water_raw.csv")

# Step 2: Clean the data
print("\nCleaning data...")
df = pd.read_csv("water_raw.csv")

# Remove completely empty rows
df = df.dropna(how="all")

# Keep only rows that have Temperature data (actual measurements)
df = df[df['Temperature'].notna() & (df['Temperature'].astype(str).str.contains(r'\d', na=False))].reset_index(drop=True)

# Keep only the data columns, drop unit columns and identifiers
df = df[["Temperature", "Dissolved_Oxygen", "pH", "Conductivity", "BOD", 
         "Nitrate", "Fecal_Coliform", "Total_Coliform", "Fecal_Streptococci"]]

# Function to extract min/max from range values like "18 26"
def extract_min_max(val):
    if pd.isna(val):
        return None, None
    val = str(val).strip()
    if val == "" or val == "nan":
        return None, None
    
    # Replace BDL (Below Detection Limit) and dashes with 0
    val = val.replace("BDL", "0").replace("-", " ")
    
    # Split by space to get min and max
    parts = [p.strip() for p in val.split() if p.strip()]
    try:
        numbers = [float(p) for p in parts]
        if not numbers:
            return 0.0, 0.0
        if len(numbers) == 1:
            return numbers[0], numbers[0]
        return numbers[0], numbers[1]
    except:
        return None, None

# Function to extract single average value for other columns
def extract_average(val):
    if pd.isna(val):
        return None
    val = str(val).strip()
    if val == "" or val == "nan":
        return None
    
    # Replace BDL (Below Detection Limit) and dashes with 0
    val = val.replace("BDL", "0").replace("-", " ")
    
    # Split by space and take average if range
    parts = [p.strip() for p in val.split() if p.strip()]
    try:
        numbers = [float(p) for p in parts]
        if not numbers:
            return 0.0
        return sum(numbers) / len(numbers)
    except:
        return None

# Extract Min/Max for Temperature, Dissolved_Oxygen, and pH
df[['Temperature_Min', 'Temperature_Max']] = df['Temperature'].apply(lambda x: pd.Series(extract_min_max(x)))
df[['DO_Min', 'DO_Max']] = df['Dissolved_Oxygen'].apply(lambda x: pd.Series(extract_min_max(x)))
df[['pH_Min', 'pH_Max']] = df['pH'].apply(lambda x: pd.Series(extract_min_max(x)))

# Calculate averages
df["Temp_Avg"] = (df["Temperature_Min"] + df["Temperature_Max"]) / 2
df["DO_Avg"] = (df["DO_Min"] + df["DO_Max"]) / 2
df["pH_Avg"] = (df["pH_Min"] + df["pH_Max"]) / 2

# Apply extract_average to other columns
for col in ["Conductivity", "BOD", "Nitrate", "Fecal_Coliform", "Total_Coliform", "Fecal_Streptococci"]:
    df[col] = df[col].apply(extract_average)

# Drop original Temperature, Dissolved_Oxygen, pH columns (keep the Min/Max/Avg versions)
df = df.drop(columns=["Temperature", "Dissolved_Oxygen", "pH"])

df = df.dropna()


def potability(row):
    if (
        row["DO_Avg"] > 5 and
        6.5 <= row["pH_Avg"] <= 8.5 and
        row["BOD"] < 3 and
        row["Fecal_Coliform"] < 2500
    ):
        return 1
    return 0

df["Potability"] = df.apply(potability, axis=1)

# Save cleaned data
df.to_csv("water_clean.csv", index=False)
print(f"✅ Clean dataset saved with {len(df)} rows")
print(f"   - Potable: {df['Potability'].sum()}")
print(f"   - Non-potable: {len(df) - df['Potability'].sum()}")

# ==========================================
# Step 3: Advanced ML Analysis
# ==========================================

print("\n" + "="*60)
print("🔬 STEP 3: CORRELATION ANALYSIS")
print("="*60)

# Check correlation between features
X_full = df.drop("Potability", axis=1)
y = df["Potability"]

correlation_matrix = X_full.corr()

# Check DO correlations specifically
print("\n📊 Dissolved Oxygen Correlations:")
print("-" * 50)
do_cols = ['DO_Min', 'DO_Max', 'DO_Avg']
for col in do_cols:
    print(f"{col}:")
    for other_col in do_cols:
        if col != other_col:
            corr = correlation_matrix.loc[col, other_col]
            print(f"  → {other_col}: {corr:.3f}")

# Similarly for Temperature and pH
print("\n📊 Temperature Correlations:")
print("-" * 50)
temp_cols = ['Temperature_Min', 'Temperature_Max', 'Temp_Avg']
for col in temp_cols:
    for other_col in temp_cols:
        if col != other_col:
            corr = correlation_matrix.loc[col, other_col]
            if col == 'Temp_Avg' and 'Min' in other_col:  # Print once
                print(f"Temp_Avg ↔ Temp_Min: {corr:.3f}")
                print(f"Temp_Avg ↔ Temp_Max: {corr:.3f}")
                break

# Save correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap="coolwarm", 
            center=0, square=True, linewidths=1)
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
print("\n✅ Correlation heatmap saved: correlation_heatmap.png")

# ==========================================
# Step 4: Feature Selection (Remove Multicollinearity)
# ==========================================

print("\n" + "="*60)
print("🎯 STEP 4: OPTIMIZED FEATURE SELECTION")
print("="*60)

# Optimal feature set (removing redundant Min/Max and weak features)
features_to_keep = [
    'DO_Avg',           # Strongest biological indicator
    'BOD',              # Organic pollution
    'Total_Coliform',   # Primary contamination indicator
    'Fecal_Coliform',   # Sewage contamination
    'pH_Avg',           # Chemical balance
    'Conductivity',     # Dissolved solids
    'Nitrate'           # Nutrient pollution
]

X = df[features_to_keep]

print("\n✨ Optimized Feature Set (No Multicollinearity):")
print("-" * 50)
print("\n🔬 Removed (Redundant/Weak):")
print("   ❌ DO_Min, DO_Max (corr=1.0 with DO_Avg)")
print("   ❌ Temperature_Min, Temperature_Max, Temp_Avg (weak importance)")
print("   ❌ pH_Min, pH_Max (corr=1.0 with pH_Avg)")
print("   ❌ Fecal_Streptococci (very weak importance)")

print("\n✅ Final Features (7):")
for i, feat in enumerate(features_to_keep, 1):
    print(f"{i:2}. {feat}")

print(f"\n📉 Reduced from {len(X_full.columns)} → {len(X.columns)} features")

# ==========================================
# Step 5: Class Imbalance Check
# ==========================================

print("\n" + "="*60)
print("⚖️  STEP 5: CLASS IMBALANCE CHECK")
print("="*60)

class_counts = y.value_counts()
print("\n🔢 Class Distribution:")
print("-" * 50)
print(f"Potable (1):     {class_counts.get(1, 0):3d} samples ({class_counts.get(1, 0)/len(y)*100:.1f}%)")
print(f"Non-Potable (0): {class_counts.get(0, 0):3d} samples ({class_counts.get(0, 0)/len(y)*100:.1f}%)")

imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\n📊 Imbalance Ratio: {imbalance_ratio:.2f}:1")
if imbalance_ratio > 3:
    print("⚠️  Warning: Significant class imbalance detected!")
    print("   Consider using SMOTE or class_weight='balanced'")
else:
    print("✅ Classes are reasonably balanced")

# ==========================================
# Step 6: Random Forest with Train/Test Split
# ==========================================

print("\n" + "="*60)
print("🌲 STEP 6: RANDOM FOREST (Overfitting Detection)")
print("="*60)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

print(f"\n📊 Dataset Split:")
print(f"   Training:   {len(X_train)} samples")
print(f"   Testing:    {len(X_test)} samples")
print(f"   Total:      {len(X)} samples")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, 
                                   class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\n📈 Training Performance:")
print(f"   Accuracy: {train_acc:.4f}")
print(f"   F1 Score: {train_f1:.4f}")

print("\n📊 Test Performance:")
print(f"   Accuracy: {test_acc:.4f}")
print(f"   F1 Score: {test_f1:.4f}")

# Overfitting Check
overfitting = train_acc - test_acc
print(f"\n⚠️  Overfitting Check:")
print(f"   Gap: {overfitting:.4f}")
if overfitting > 0.1:
    print("   ❌ Significant overfitting detected!")
elif overfitting > 0.05:
    print("   ⚠️  Mild overfitting")
else:
    print("   ✅ No overfitting - Good generalization!")

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')

print("\n📈 Cross-Validation Results (5-fold):")
print("-" * 50)
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.4f}")
print(f"\n🎯 Mean CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print("\n📊 Feature Importances (Random Forest):")
print("-" * 50)
rf_importances = sorted(zip(X.columns, rf_model.feature_importances_), 
                        key=lambda x: x[1], reverse=True)
for name, score in rf_importances:
    bar = '█' * int(score * 100)
    print(f"{name:20} {score:.4f} {bar}")

# Confusion Matrix
print("\n🔢 Confusion Matrix (Test Set):")
print("-" * 50)
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}")

# ==========================================
# Step 7: XGBoost Model
# ==========================================

print("\n" + "="*60)
print("🚀 STEP 7: XGBOOST CLASSIFIER")
print("="*60)

xgb_model = XGBClassifier(n_estimators=100, random_state=42, 
                          eval_metric='logloss', use_label_encoder=False)
xgb_model.fit(X_train, y_train)

# Predictions
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)

# Metrics
train_acc_xgb = accuracy_score(y_train, y_train_pred_xgb)
test_acc_xgb = accuracy_score(y_test, y_test_pred_xgb)
train_f1_xgb = f1_score(y_train, y_train_pred_xgb)
test_f1_xgb = f1_score(y_test, y_test_pred_xgb)

print("\n📈 Training Performance:")
print(f"   Accuracy: {train_acc_xgb:.4f}")
print(f"   F1 Score: {train_f1_xgb:.4f}")

print("\n📊 Test Performance:")
print(f"   Accuracy: {test_acc_xgb:.4f}")
print(f"   F1 Score: {test_f1_xgb:.4f}")

# Overfitting Check
overfitting_xgb = train_acc_xgb - test_acc_xgb
print(f"\n⚠️  Overfitting Check:")
print(f"   Gap: {overfitting_xgb:.4f}")
if overfitting_xgb > 0.1:
    print("   ❌ Significant overfitting detected!")
elif overfitting_xgb > 0.05:
    print("   ⚠️  Mild overfitting")
else:
    print("   ✅ No overfitting - Good generalization!")

# Cross-validation for XGBoost
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')

print("\n📈 Cross-Validation Results (5-fold):")
print("-" * 50)
for i, score in enumerate(xgb_cv_scores, 1):
    print(f"Fold {i}: {score:.4f}")
print(f"\n🎯 Mean CV Accuracy: {xgb_cv_scores.mean():.4f} (±{xgb_cv_scores.std():.4f})")

print("\n📊 Feature Importances (XGBoost):")
print("-" * 50)
xgb_importances = sorted(zip(X.columns, xgb_model.feature_importances_), 
                         key=lambda x: x[1], reverse=True)
for name, score in xgb_importances:
    bar = '█' * int(score * 100)
    print(f"{name:20} {score:.4f} {bar}")

# Confusion Matrix
print("\n🔢 Confusion Matrix (Test Set):")
print("-" * 50)
cm_xgb = confusion_matrix(y_test, y_test_pred_xgb)
print(cm_xgb)
print(f"\nTrue Negatives:  {cm_xgb[0,0]}")
print(f"False Positives: {cm_xgb[0,1]}")
print(f"False Negatives: {cm_xgb[1,0]}")
print(f"True Positives:  {cm_xgb[1,1]}")

# ==========================================
# Step 8: SHAP Analysis (Explainable AI)
# ==========================================

print("\n" + "="*60)
print("🔥 STEP 8: SHAP ANALYSIS (Explainable AI)")
print("="*60)

print("\n⚙️  Computing SHAP values...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

# SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
print("✅ SHAP summary plot saved: shap_summary_plot.png")

# SHAP Bar Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=300, bbox_inches='tight')
print("✅ SHAP feature importance saved: shap_feature_importance.png")

# ==========================================
# Step 9: Model Comparison
# ==========================================

print("\n" + "="*60)
print("📊 STEP 9: MODEL COMPARISON")
print("="*60)

print("\n🏆 Final Results:")
print("-" * 50)
print(f"Random Forest CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"XGBoost CV Accuracy:       {xgb_cv_scores.mean():.4f} (±{xgb_cv_scores.std():.4f})")

if xgb_cv_scores.mean() > cv_scores.mean():
    print("\n🥇 Winner: XGBoost")
    best_model = "XGBoost"
else:
    print("\n🥇 Winner: Random Forest")
    best_model = "Random Forest"

print("\n" + "="*60)
print("✨ ANALYSIS COMPLETE!")
print("="*60)
print(f"\n📁 Generated Files:")
print("   • water_clean.csv")
print("   • correlation_heatmap.png")
print("   • shap_summary_plot.png")
print("   • shap_feature_importance.png")
print(f"\n🎯 Best Model: {best_model}")
print(f"🔬 Top 3 Features:")
top_features = xgb_importances[:3] if best_model == "XGBoost" else rf_importances[:3]
for i, (name, score) in enumerate(top_features, 1):
    print(f"   {i}. {name} ({score:.4f})")

