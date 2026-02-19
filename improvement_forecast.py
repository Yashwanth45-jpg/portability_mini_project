import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔮 WATER QUALITY IMPROVEMENT & TREND PREDICTION SYSTEM")
print("="*80)

# Load data and train model
df = pd.read_csv("water_optimized.csv")

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# ==========================================
# 1. IMPROVEMENT RECOMMENDATIONS
# ==========================================

print("\n" + "="*80)
print("💡 PART 1: HOW TO IMPROVE CONTAMINATED WATER")
print("="*80)

# WHO/EPA Standards (reference values)
STANDARDS = {
    'DO_Avg': {'min': 5.0, 'optimal': 8.0, 'unit': 'mg/L', 'direction': 'higher'},
    'BOD': {'max': 3.0, 'optimal': 1.0, 'unit': 'mg/L', 'direction': 'lower'},
    'Total_Coliform': {'max': 2500, 'optimal': 50, 'unit': 'MPN/100mL', 'direction': 'lower'},
    'Fecal_Coliform': {'max': 2500, 'optimal': 10, 'unit': 'MPN/100mL', 'direction': 'lower'},
    'pH_Avg': {'min': 6.5, 'max': 8.5, 'optimal': 7.0, 'unit': 'pH', 'direction': 'neutral'},
    'Conductivity': {'max': 1500, 'optimal': 300, 'unit': 'μmhos/cm', 'direction': 'lower'},
    'Nitrate': {'max': 10.0, 'optimal': 1.0, 'unit': 'mg/L', 'direction': 'lower'}
}

def analyze_water_sample(sample_data, sample_name="Water Sample"):
    """Analyze a water sample and provide improvement recommendations"""
    
    print(f"\n{'='*80}")
    print(f"🧪 ANALYZING: {sample_name}")
    print(f"{'='*80}")
    
    # Make prediction
    sample_df = pd.DataFrame([sample_data])
    prediction = model.predict(sample_df)[0]
    probability = model.predict_proba(sample_df)[0]
    
    print(f"\n📊 Current Status:")
    print(f"   Prediction: {'✅ POTABLE' if prediction == 1 else '⚠️  NON-POTABLE'}")
    print(f"   Confidence: {probability[prediction]:.1%}")
    
    print(f"\n📋 Current Values vs Standards:")
    print("-" * 80)
    print(f"{'Parameter':<20} {'Current':<15} {'Standard':<20} {'Status':<15}")
    print("-" * 80)
    
    violations = []
    improvements_needed = {}
    
    for param, value in sample_data.items():
        std = STANDARDS[param]
        unit = std['unit']
        
        # Check violations
        status = "✅ OK"
        issue = None
        
        if 'min' in std and 'max' in std:  # pH range
            if value < std['min']:
                status = "⚠️  TOO LOW"
                issue = f"Increase to {std['min']}-{std['max']}"
                improvements_needed[param] = std['optimal'] - value
                violations.append(param)
            elif value > std['max']:
                status = "⚠️  TOO HIGH"
                issue = f"Decrease to {std['min']}-{std['max']}"
                improvements_needed[param] = value - std['optimal']
                violations.append(param)
        elif 'max' in std:  # Should be below max
            if value > std['max']:
                status = "🚨 EXCEEDED"
                issue = f"Reduce to <{std['max']}"
                improvements_needed[param] = value - std['optimal']
                violations.append(param)
        elif 'min' in std:  # Should be above min
            if value < std['min']:
                status = "🚨 TOO LOW"
                issue = f"Increase to >{std['min']}"
                improvements_needed[param] = std['optimal'] - value
                violations.append(param)
        
        standard_str = f"<{std.get('max', 'N/A')}" if 'max' in std else f">{std.get('min', 'N/A')}"
        if 'min' in std and 'max' in std:
            standard_str = f"{std['min']}-{std['max']}"
        
        print(f"{param:<20} {value:<15.2f} {standard_str:<20} {status:<15}")
    
    # Improvement recommendations
    if violations:
        print(f"\n🔧 IMPROVEMENT RECOMMENDATIONS:")
        print("-" * 80)
        print(f"Found {len(violations)} parameter(s) needing improvement:\n")
        
        # Sort by feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        violations_sorted = sorted(violations, 
                                  key=lambda x: feature_importance[x], 
                                  reverse=True)
        
        for i, param in enumerate(violations_sorted, 1):
            std = STANDARDS[param]
            current_val = sample_data[param]
            importance = feature_importance[param]
            
            print(f"{i}. **{param}** (Importance: {importance:.1%})")
            print(f"   Current: {current_val:.2f} {std['unit']}")
            print(f"   Target:  {std['optimal']:.2f} {std['unit']}")
            print(f"   Change needed: {abs(std['optimal'] - current_val):.2f} {std['unit']}")
            
            # Specific treatment recommendations
            print(f"   💊 Treatment Methods:")
            if param == 'DO_Avg':
                print(f"      • Aeration (add oxygen)")
                print(f"      • Reduce organic pollution")
                print(f"      • Install aerators/fountains")
            elif param == 'BOD':
                print(f"      • Wastewater treatment")
                print(f"      • Reduce organic discharge")
                print(f"      • Biological treatment plants")
            elif param in ['Total_Coliform', 'Fecal_Coliform']:
                print(f"      • Chlorination/UV treatment")
                print(f"      • Stop sewage discharge")
                print(f"      • Improve sanitation infrastructure")
            elif param == 'pH_Avg':
                if current_val < std['min']:
                    print(f"      • Add lime/sodium carbonate (increase pH)")
                else:
                    print(f"      • Add sulfuric acid/CO2 (decrease pH)")
            elif param == 'Nitrate':
                print(f"      • Biological denitrification")
                print(f"      • Control agricultural runoff")
                print(f"      • Ion exchange treatment")
            elif param == 'Conductivity':
                print(f"      • Reverse osmosis")
                print(f"      • Reduce industrial discharge")
            print()
        
        # Estimate timeline to fix
        print(f"⏱️  ESTIMATED IMPROVEMENT TIMELINE:")
        print("-" * 80)
        
        # Priority levels based on feature importance
        high_priority = [p for p in violations_sorted if feature_importance[p] > 0.15]
        medium_priority = [p for p in violations_sorted if 0.05 < feature_importance[p] <= 0.15]
        low_priority = [p for p in violations_sorted if feature_importance[p] <= 0.05]
        
        if high_priority:
            print(f"🔴 HIGH PRIORITY (Fix first): {', '.join(high_priority)}")
            print(f"   Timeline: 1-3 months with proper treatment")
            print(f"   Cost: $$$ High (requires infrastructure)")
        if medium_priority:
            print(f"🟡 MEDIUM PRIORITY: {', '.join(medium_priority)}")
            print(f"   Timeline: 3-6 months")
            print(f"   Cost: $$ Moderate")
        if low_priority:
            print(f"🟢 LOW PRIORITY: {', '.join(low_priority)}")
            print(f"   Timeline: 6-12 months")
            print(f"   Cost: $ Low")
        
        print(f"\n📅 TOTAL ESTIMATED TIME TO POTABILITY: {estimate_timeline(violations_sorted, feature_importance)}")
    else:
        print(f"\n✅ All parameters within acceptable limits!")
        print(f"   This water meets potability standards.")

def estimate_timeline(violations, importance):
    """Estimate time needed to fix water based on violations"""
    if not violations:
        return "Already potable"
    
    # Weight by importance
    severity_score = sum([importance[v] for v in violations])
    
    if severity_score > 0.5:
        return "6-12 months (Major treatment needed)"
    elif severity_score > 0.3:
        return "3-6 months (Moderate treatment)"
    else:
        return "1-3 months (Minor improvements)"

# ==========================================
# 2. ANALYZE CONTAMINATED SAMPLES
# ==========================================

# Find worst case in dataset
non_potable_samples = df[df['Potability'] == 0]

if len(non_potable_samples) > 0:
    worst_sample = non_potable_samples.iloc[0].drop('Potability').to_dict()
    analyze_water_sample(worst_sample, "Worst Contaminated Sample from Dataset")

# ==========================================
# 3. POLLUTION TREND PREDICTION
# ==========================================

print("\n" + "="*80)
print("📈 PART 2: POLLUTION TREND PREDICTION")
print("="*80)

print("\n⚠️  Note: This dataset is cross-sectional (single point in time).")
print("    For accurate trend prediction, we need time-series data.")
print("    Below is a simulation based on common pollution patterns.\n")

def simulate_pollution_trends(sample_data, sample_name="River Location"):
    """Simulate future pollution trends if no action is taken"""
    
    print(f"\n{'='*80}")
    print(f"🔮 POLLUTION FORECAST: {sample_name}")
    print(f"{'='*80}")
    
    # Typical pollution increase rates (% per year) if unchecked
    POLLUTION_RATES = {
        'DO_Avg': -0.05,          # Oxygen decreases 5% per year
        'BOD': 0.10,              # BOD increases 10% per year
        'Total_Coliform': 0.15,   # Bacteria increase 15% per year
        'Fecal_Coliform': 0.15,   # Fecal bacteria increase 15% per year
        'pH_Avg': -0.01,          # pH becomes more acidic
        'Conductivity': 0.08,     # Conductivity increases 8% per year
        'Nitrate': 0.12           # Nitrate increases 12% per year (agriculture)
    }
    
    current_prediction = model.predict(pd.DataFrame([sample_data]))[0]
    current_prob = model.predict_proba(pd.DataFrame([sample_data]))[0]
    
    print(f"\n📊 Current Status (Year 0):")
    print(f"   Status: {'✅ POTABLE' if current_prediction == 1 else '⚠️  NON-POTABLE'}")
    print(f"   Potability Probability: {current_prob[1]:.1%}")
    
    # Simulate future years
    years_to_simulate = 10
    future_data = []
    
    print(f"\n📅 Projected Pollution Trends (No Intervention):")
    print("-" * 80)
    print(f"{'Year':<8} {'DO_Avg':<12} {'BOD':<12} {'Coliform':<15} {'pH':<10} {'Potable%':<12}")
    print("-" * 80)
    
    current_state = sample_data.copy()
    year_contaminated = None
    
    for year in range(years_to_simulate + 1):
        if year > 0:
            # Apply degradation rates
            for param in current_state:
                rate = POLLUTION_RATES[param]
                current_state[param] = current_state[param] * (1 + rate)
                
                # Apply bounds
                if param == 'DO_Avg':
                    current_state[param] = max(0, current_state[param])  # Can't go below 0
                if param == 'pH_Avg':
                    current_state[param] = max(5.0, current_state[param])  # Min pH
        
        # Predict
        pred_df = pd.DataFrame([current_state])
        prediction = model.predict(pred_df)[0]
        prob = model.predict_proba(pred_df)[0]
        
        future_data.append({
            'Year': year,
            'DO_Avg': current_state['DO_Avg'],
            'BOD': current_state['BOD'],
            'Total_Coliform': current_state['Total_Coliform'],
            'pH_Avg': current_state['pH_Avg'],
            'Potable_Prob': prob[1],
            'Status': prediction
        })
        
        # Print every 2 years
        if year % 2 == 0:
            print(f"{year:<8} {current_state['DO_Avg']:<12.2f} {current_state['BOD']:<12.2f} "
                  f"{current_state['Total_Coliform']:<15.0f} {current_state['pH_Avg']:<10.2f} "
                  f"{prob[1]:<12.1%}")
        
        # Track when it becomes non-potable
        if prediction == 0 and year_contaminated is None and current_prediction == 1:
            year_contaminated = year
    
    # Analysis
    print("\n" + "="*80)
    if current_prediction == 1:
        if year_contaminated:
            print(f"🚨 WARNING: Water will become NON-POTABLE in ~{year_contaminated} years!")
            print(f"   Action required before: {2026 + year_contaminated}")
            print(f"\n💡 Immediate Actions Needed:")
            print(f"   1. Implement pollution control measures NOW")
            print(f"   2. Monitor water quality quarterly")
            print(f"   3. Strengthen wastewater treatment")
            print(f"   4. Control industrial/agricultural discharge")
        else:
            print(f"✅ Water remains POTABLE over {years_to_simulate}-year projection")
            print(f"   However, quality will decline without intervention")
    else:
        final_prob = future_data[-1]['Potable_Prob']
        print(f"⚠️  Water is ALREADY NON-POTABLE")
        print(f"   Projected to worsen: {current_prob[1]:.1%} → {final_prob:.1%} in {years_to_simulate} years")
        print(f"\n🚨 URGENT: Immediate treatment required!")
    
    # Create visualization
    df_trends = pd.DataFrame(future_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Potability Probability
    axes[0, 0].plot(df_trends['Year'], df_trends['Potable_Prob'] * 100, 
                    marker='o', linewidth=2, color='dodgerblue')
    axes[0, 0].axhline(y=50, color='red', linestyle='--', label='Threshold')
    axes[0, 0].fill_between(df_trends['Year'], 0, 50, alpha=0.2, color='red')
    axes[0, 0].fill_between(df_trends['Year'], 50, 100, alpha=0.2, color='green')
    axes[0, 0].set_xlabel('Years from Now', fontweight='bold')
    axes[0, 0].set_ylabel('Potability Probability (%)', fontweight='bold')
    axes[0, 0].set_title('Predicted Potability Over Time', fontweight='bold', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Dissolved Oxygen
    axes[0, 1].plot(df_trends['Year'], df_trends['DO_Avg'], 
                    marker='s', linewidth=2, color='teal')
    axes[0, 1].axhline(y=5.0, color='red', linestyle='--', label='Min Safe Level')
    axes[0, 1].set_xlabel('Years from Now', fontweight='bold')
    axes[0, 1].set_ylabel('Dissolved Oxygen (mg/L)', fontweight='bold')
    axes[0, 1].set_title('DO Degradation Trend', fontweight='bold', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Total Coliform
    axes[1, 0].plot(df_trends['Year'], df_trends['Total_Coliform'], 
                    marker='^', linewidth=2, color='darkred')
    axes[1, 0].axhline(y=2500, color='red', linestyle='--', label='Max Safe Level')
    axes[1, 0].set_xlabel('Years from Now', fontweight='bold')
    axes[1, 0].set_ylabel('Total Coliform (MPN/100mL)', fontweight='bold')
    axes[1, 0].set_title('Bacterial Contamination Growth', fontweight='bold', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    
    # Plot 4: BOD
    axes[1, 1].plot(df_trends['Year'], df_trends['BOD'], 
                    marker='d', linewidth=2, color='darkorange')
    axes[1, 1].axhline(y=3.0, color='red', linestyle='--', label='Max Safe Level')
    axes[1, 1].set_xlabel('Years from Now', fontweight='bold')
    axes[1, 1].set_ylabel('BOD (mg/L)', fontweight='bold')
    axes[1, 1].set_title('Organic Pollution Trend', fontweight='bold', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('pollution_forecast.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved: pollution_forecast.png")
    
    return df_trends

# Simulate for a contaminated sample
if len(non_potable_samples) > 0:
    contaminated_sample = non_potable_samples.iloc[0].drop('Potability').to_dict()
    trend_df = simulate_pollution_trends(contaminated_sample, "Severely Contaminated River Section")

# Simulate for a currently safe sample
potable_samples = df[df['Potability'] == 1]
if len(potable_samples) > 0:
    safe_sample = potable_samples.iloc[0].drop('Potability').to_dict()
    trend_df_safe = simulate_pollution_trends(safe_sample, "Currently Safe River Section")

# ==========================================
# 4. ACTIONABLE SUMMARY
# ==========================================

print("\n" + "="*80)
print("📋 ACTIONABLE SUMMARY & RECOMMENDATIONS")
print("="*80)

print("\n🎯 IMMEDIATE ACTIONS (Next 6 Months):")
print("1. Install continuous water quality monitoring sensors")
print("2. Implement bacterial treatment (chlorination/UV)")
print("3. Enforce industrial discharge regulations")
print("4. Upgrade sewage treatment plants")

print("\n🎯 MEDIUM-TERM (6-12 Months):")
print("1. Aeration systems to increase dissolved oxygen")
print("2. Biological treatment for organic pollution")
print("3. Riparian zone restoration")
print("4. Community awareness programs")

print("\n🎯 LONG-TERM (1-3 Years):")
print("1. Watershed management plan")
print("2. Agricultural runoff control")
print("3. Green infrastructure development")
print("4. Regular water quality audits")

print("\n💰 ESTIMATED COSTS:")
print("• Emergency treatment: $50,000 - $200,000")
print("• Infrastructure upgrade: $500,000 - $2,000,000")
print("• Ongoing monitoring: $10,000 - $30,000 per year")

print("\n⏰ TIMELINE TO FULL RECOVERY:")
print("• Minor contamination: 6-12 months")
print("• Moderate contamination: 1-2 years")
print("• Severe contamination: 3-5 years")
print("• Complete ecosystem recovery: 5-10 years")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
