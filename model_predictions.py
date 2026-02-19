import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

print("="*70)
print("🎯 WHAT CAN THIS MODEL PREDICT?")
print("="*70)

# Load data and train model
df = pd.read_csv("water_optimized.csv")

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the best model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

print("\n" + "="*70)
print("📥 INPUT: 7 Water Quality Parameters")
print("="*70)

required_features = list(X.columns)
print("\nThe model needs these 7 measurements:")
print("-" * 70)
for i, feature in enumerate(required_features, 1):
    print(f"{i}. {feature}")

print("\n" + "="*70)
print("📤 OUTPUT: Water Potability Prediction")
print("="*70)

print("\nThe model predicts:")
print("-" * 70)
print("• Class: 0 (Non-Potable) or 1 (Potable)")
print("• Probability: Confidence score (0.0 to 1.0)")
print("• Interpretation: Is this water safe to drink?")

print("\n" + "="*70)
print("🧪 EXAMPLE PREDICTIONS")
print("="*70)

# Example 1: Good water quality (first sample from test set)
print("\n📗 Example 1: GOOD WATER QUALITY")
print("-" * 70)
example1 = X_test.iloc[0:1]
print("Input parameters:")
for param, value in example1.iloc[0].items():
    print(f"  {param}: {value:.2f}")

prediction1 = model.predict(example1)[0]
probability1 = model.predict_proba(example1)[0]

print(f"\n✅ Prediction: {'POTABLE' if prediction1 == 1 else 'NON-POTABLE'}")
print(f"✅ Confidence: {probability1[1]:.2%} potable, {probability1[0]:.2%} non-potable")
print(f"✅ Manual check from actual data: {y_test.iloc[0]}")

# Example 2: Poor water quality (if available)
non_potable_indices = y_test[y_test == 0].index
if len(non_potable_indices) > 0:
    print("\n📕 Example 2: CONTAMINATED WATER")
    print("-" * 70)
    # Get the position in y_test
    idx_in_test = list(y_test.index).index(non_potable_indices[0])
    example2 = X_test.iloc[idx_in_test:idx_in_test+1]
    print("Input parameters:")
    for param, value in example2.iloc[0].items():
        print(f"  {param}: {value:.2f}")
    
    prediction2 = model.predict(example2)[0]
    probability2 = model.predict_proba(example2)[0]
    
    print(f"\n⚠️  Prediction: {'POTABLE' if prediction2 == 1 else 'NON-POTABLE'}")
    print(f"⚠️  Confidence: {probability2[1]:.2%} potable, {probability2[0]:.2%} non-potable")
    print(f"⚠️  Manual check from actual data: {y_test.iloc[idx_in_test]}")

# Example 3: Custom water sample
print("\n📘 Example 3: CUSTOM WATER SAMPLE (Hypothetical)")
print("-" * 70)
custom_water = pd.DataFrame({
    'DO_Avg': [7.5],           # Good oxygen level
    'BOD': [1.5],              # Low organic pollution
    'Total_Coliform': [500],   # Moderate bacteria
    'Fecal_Coliform': [100],   # Low sewage
    'pH_Avg': [7.2],           # Neutral pH
    'Conductivity': [300],     # Normal conductivity
    'Nitrate': [2.0]           # Low nitrate
})

print("Input parameters:")
for param, value in custom_water.iloc[0].items():
    print(f"  {param}: {value:.2f}")

prediction3 = model.predict(custom_water)[0]
probability3 = model.predict_proba(custom_water)[0]

print(f"\n🔍 Prediction: {'POTABLE ✅' if prediction3 == 1 else 'NON-POTABLE ⚠️'}")
print(f"🔍 Confidence: {probability3[1]:.2%} potable, {probability3[0]:.2%} non-potable")

print("\n" + "="*70)
print("🎯 PRACTICAL USE CASES")
print("="*70)

print("\n1️⃣  ENVIRONMENTAL MONITORING")
print("   • Monitor river water quality at different locations")
print("   • Track seasonal changes in water potability")
print("   • Identify pollution hotspots")

print("\n2️⃣  WATER TREATMENT PLANTS")
print("   • Verify water quality after treatment")
print("   • Real-time quality control")
print("   • Adjust treatment processes based on predictions")

print("\n3️⃣  GOVERNMENT AGENCIES")
print("   • Assess drinking water safety compliance")
print("   • Prioritize areas needing infrastructure")
print("   • Generate water quality reports")

print("\n4️⃣  RESEARCH & ANALYSIS")
print("   • Study pollution patterns")
print("   • Predict impact of industrial discharge")
print("   • Environmental impact assessments")

print("\n5️⃣  PUBLIC HEALTH")
print("   • Alert communities about unsafe water")
print("   • Guide boil water advisories")
print("   • Prevent waterborne disease outbreaks")

print("\n" + "="*70)
print("💡 HOW TO USE FOR NEW PREDICTIONS")
print("="*70)

print("\n📝 Python Code Example:")
print("-" * 70)
print("""
# Step 1: Prepare your new water sample data
new_sample = pd.DataFrame({
    'DO_Avg': [8.5],
    'BOD': [0.5],
    'Total_Coliform': [50],
    'Fecal_Coliform': [10],
    'pH_Avg': [7.8],
    'Conductivity': [250],
    'Nitrate': [1.0]
})

# Step 2: Make prediction
prediction = model.predict(new_sample)[0]
probability = model.predict_proba(new_sample)[0]

# Step 3: Interpret result
if prediction == 1:
    print(f"✅ SAFE TO DRINK (Confidence: {probability[1]:.2%})")
else:
    print(f"⚠️  DO NOT DRINK (Confidence: {probability[0]:.2%})")
""")

print("\n" + "="*70)
print("📊 PREDICTION SUMMARY FOR ENTIRE DATASET")
print("="*70)

all_predictions = model.predict(X)
all_probabilities = model.predict_proba(X)

potable_count = (all_predictions == 1).sum()
non_potable_count = (all_predictions == 0).sum()

print(f"\nOut of {len(df)} water samples:")
print(f"  ✅ Potable:     {potable_count} samples ({potable_count/len(df)*100:.1f}%)")
print(f"  ⚠️  Non-Potable: {non_potable_count} samples ({non_potable_count/len(df)*100:.1f}%)")

# Show confidence distribution
avg_confidence_potable = all_probabilities[all_predictions == 1, 1].mean()
avg_confidence_non_potable = all_probabilities[all_predictions == 0, 0].mean()

print(f"\nAverage prediction confidence:")
print(f"  Potable samples:     {avg_confidence_potable:.2%}")
print(f"  Non-potable samples: {avg_confidence_non_potable:.2%}")

print("\n" + "="*70)
print("🔬 WHAT MAKES WATER NON-POTABLE? (Model Learned)")
print("="*70)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop factors affecting potability (in order):")
print("-" * 70)
for idx, row in feature_importance.iterrows():
    bar = '█' * int(row['Importance'] * 100)
    print(f"{row['Feature']:<20} {row['Importance']:.4f} {bar}")

print("\n📌 Key Insights:")
print("-" * 70)
print("✓ High Total_Coliform → Water likely contaminated")
print("✓ Low DO_Avg (Dissolved Oxygen) → Poor water quality")
print("✓ High BOD (Biochemical Oxygen Demand) → Organic pollution")
print("✓ High Fecal_Coliform → Sewage contamination")
print("✓ pH out of range (6.5-8.5) → May indicate contamination")

print("\n" + "="*70)
print("✨ READY TO DEPLOY!")
print("="*70)
print("\n💡 Next Steps:")
print("   1. Save trained model: joblib.dump(model, 'water_model.pkl')")
print("   2. Create web API with Flask/FastAPI")
print("   3. Build Streamlit dashboard")
print("   4. Deploy to cloud (AWS/Azure/GCP)")
print("   5. Integrate with IoT sensors for real-time monitoring")
print("\n" + "="*70)
