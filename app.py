import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Water Quality Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        padding: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .safe-water {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .unsafe-water {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Load or train model
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        df = pd.read_csv("water_optimized.csv")
        X = df.drop("Potability", axis=1)
        y = df["Potability"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        return model, X.columns.tolist()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Water quality standards
STANDARDS = {
    'DO_Avg': {'min': 5.0, 'optimal': 8.0, 'unit': 'mg/L'},
    'BOD': {'max': 3.0, 'optimal': 1.0, 'unit': 'mg/L'},
    'Total_Coliform': {'max': 2500, 'optimal': 50, 'unit': 'MPN/100mL'},
    'Fecal_Coliform': {'max': 2500, 'optimal': 10, 'unit': 'MPN/100mL'},
    'pH_Avg': {'min': 6.5, 'max': 8.5, 'optimal': 7.0, 'unit': 'pH'},
    'Conductivity': {'max': 1500, 'optimal': 300, 'unit': 'μmhos/cm'},
    'Nitrate': {'max': 10.0, 'optimal': 1.0, 'unit': 'mg/L'}
}

def get_parameter_status(param, value):
    """Check if parameter is within safe limits"""
    std = STANDARDS[param]
    
    if 'min' in std and 'max' in std:
        if std['min'] <= value <= std['max']:
            return '✅', 'Safe', 'green'
        else:
            return '⚠️', 'Out of range', 'red'
    elif 'max' in std:
        if value <= std['max']:
            return '✅', 'Safe', 'green'
        else:
            return '🚨', 'Exceeded', 'red'
    elif 'min' in std:
        if value >= std['min']:
            return '✅', 'Safe', 'green'
        else:
            return '⚠️', 'Too low', 'red'
    
    return '❓', 'Unknown', 'gray'

def predict_pollution_trend(sample_data, years=10):
    """Simulate pollution trends over time"""
    POLLUTION_RATES = {
        'DO_Avg': -0.05,
        'BOD': 0.10,
        'Total_Coliform': 0.15,
        'Fecal_Coliform': 0.15,
        'pH_Avg': -0.01,
        'Conductivity': 0.08,
        'Nitrate': 0.12
    }
    
    current_state = sample_data.copy()
    trends = []
    
    for year in range(years + 1):
        if year > 0:
            for param in current_state:
                rate = POLLUTION_RATES.get(param, 0)
                current_state[param] = current_state[param] * (1 + rate)
                
                if param == 'DO_Avg':
                    current_state[param] = max(0, current_state[param])
                if param == 'pH_Avg':
                    current_state[param] = max(5.0, current_state[param])
        
        pred_df = pd.DataFrame([current_state])
        prediction = model.predict(pred_df)[0]
        prob = model.predict_proba(pred_df)[0]
        
        trends.append({
            'Year': 2026 + year,
            'Potable_Prob': prob[1] * 100,
            'DO_Avg': current_state['DO_Avg'],
            'Total_Coliform': current_state['Total_Coliform'],
            'BOD': current_state['BOD']
        })
    
    return pd.DataFrame(trends)

# Load model
model, feature_names = load_model()

# Header
st.markdown('<p class="main-header">💧 Water Quality Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Water Potability Analysis & Forecasting</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/water.png", width=150)
    st.title("Navigation")
    
    page = st.radio(
        "Select Analysis Type:",
        ["🔍 Single Sample Prediction", 
         "📊 Batch Analysis (CSV)", 
         "📈 Pollution Forecast",
         "ℹ️ About & Guide"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("""
    **Accuracy:** 100%
    **Model:** Random Forest
    **Features:** 7 parameters
    **Dataset:** 49 locations
    """)

# Page 1: Single Sample Prediction
if page == "🔍 Single Sample Prediction":
    st.header("🧪 Analyze Individual Water Sample")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Input Water Quality Parameters")
        
        do_avg = st.number_input(
            "Dissolved Oxygen (mg/L)", 
            min_value=0.0, max_value=15.0, value=8.0, step=0.1,
            help="Oxygen dissolved in water. Higher is better (>5 mg/L safe)"
        )
        
        bod = st.number_input(
            "BOD - Biochemical Oxygen Demand (mg/L)", 
            min_value=0.0, max_value=50.0, value=1.0, step=0.1,
            help="Organic pollution indicator. Lower is better (<3 mg/L safe)"
        )
        
        total_coliform = st.number_input(
            "Total Coliform (MPN/100mL)", 
            min_value=0.0, max_value=100000.0, value=100.0, step=10.0,
            help="Bacteria count. Lower is better (<2500 safe)"
        )
        
        fecal_coliform = st.number_input(
            "Fecal Coliform (MPN/100mL)", 
            min_value=0.0, max_value=50000.0, value=10.0, step=10.0,
            help="Sewage indicator. Lower is better (<2500 safe)"
        )
    
    with col2:
        st.subheader("📥 Additional Parameters")
        
        ph_avg = st.number_input(
            "pH Level", 
            min_value=0.0, max_value=14.0, value=7.0, step=0.1,
            help="Acidity/Alkalinity. Safe range: 6.5-8.5"
        )
        
        conductivity = st.number_input(
            "Conductivity (μmhos/cm)", 
            min_value=0.0, max_value=5000.0, value=300.0, step=10.0,
            help="Dissolved solids. Lower is better (<1500 safe)"
        )
        
        nitrate = st.number_input(
            "Nitrate (mg/L)", 
            min_value=0.0, max_value=50.0, value=1.0, step=0.1,
            help="Nutrient pollution. Lower is better (<10 safe)"
        )
    
    # Predict button
    if st.button("🔮 Predict Water Quality", type="primary", use_container_width=True):
        # Prepare data
        sample_data = {
            'DO_Avg': do_avg,
            'BOD': bod,
            'Total_Coliform': total_coliform,
            'Fecal_Coliform': fecal_coliform,
            'pH_Avg': ph_avg,
            'Conductivity': conductivity,
            'Nitrate': nitrate
        }
        
        sample_df = pd.DataFrame([sample_data])
        
        # Make prediction
        prediction = model.predict(sample_df)[0]
        probability = model.predict_proba(sample_df)[0]
        
        st.markdown("---")
        
        # Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="safe-water">
                    <h2>✅ POTABLE</h2>
                    <h3>Water is Safe to Drink</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="unsafe-water">
                    <h2>⚠️ NON-POTABLE</h2>
                    <h3>Do Not Drink!</h3>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Potability Confidence", f"{probability[prediction]:.1%}")
            st.metric("Safe Probability", f"{probability[1]:.1%}")
        
        with col3:
            st.metric("Unsafe Probability", f"{probability[0]:.1%}")
            if probability[prediction] > 0.9:
                st.success("High Confidence")
            elif probability[prediction] > 0.7:
                st.warning("Moderate Confidence")
            else:
                st.error("Low Confidence")
        
        # Parameter analysis
        st.markdown("---")
        st.subheader("📊 Parameter Analysis")
        
        # Create gauge charts
        col1, col2, col3, col4 = st.columns(4)
        
        violations = []
        
        for i, (param, value) in enumerate(sample_data.items()):
            status_icon, status_text, color = get_parameter_status(param, value)
            
            col = [col1, col2, col3, col4][i % 4]
            
            with col:
                st.metric(
                    param.replace('_', ' '),
                    f"{value:.2f} {STANDARDS[param]['unit']}",
                    f"{status_icon} {status_text}"
                )
                
                if color == 'red':
                    violations.append(param)
        
        # Improvement recommendations
        if violations or prediction == 0:
            st.markdown("---")
            st.subheader("💡 Improvement Recommendations")
            
            if violations:
                for param in violations:
                    std = STANDARDS[param]
                    current_val = sample_data[param]
                    
                    st.warning(f"**{param}**: Current = {current_val:.2f}, Target = {std['optimal']:.2f} {std['unit']}")
                    
                    # Treatment suggestions
                    if param == 'DO_Avg':
                        st.markdown("- Install aeration systems")
                        st.markdown("- Reduce organic pollution sources")
                    elif param == 'BOD':
                        st.markdown("- Upgrade wastewater treatment")
                        st.markdown("- Control industrial discharge")
                    elif 'Coliform' in param:
                        st.markdown("- UV/Chlorination treatment")
                        st.markdown("- Fix sewage infrastructure")
                    elif param == 'pH_Avg':
                        st.markdown("- Chemical pH adjustment")
                    elif param == 'Nitrate':
                        st.markdown("- Control agricultural runoff")
                    elif param == 'Conductivity':
                        st.markdown("- Reverse osmosis treatment")
            
            # Timeline estimate
            st.info(f"⏰ **Estimated improvement time:** 6-18 months depending on severity")

# Page 2: Batch Analysis
elif page == "📊 Batch Analysis (CSV)":
    st.header("📁 Batch Water Quality Analysis")
    
    st.markdown("""
    Upload a CSV file with water quality data to analyze multiple samples at once.
    
    **Required columns:** `DO_Avg`, `BOD`, `Total_Coliform`, `Fecal_Coliform`, `pH_Avg`, `Conductivity`, `Nitrate`
    """)
    
    # Sample CSV download
    sample_data = pd.DataFrame({
        'Location': ['Site A', 'Site B'],
        'DO_Avg': [8.0, 3.0],
        'BOD': [1.0, 5.0],
        'Total_Coliform': [50, 5000],
        'Fecal_Coliform': [10, 2000],
        'pH_Avg': [7.2, 6.8],
        'Conductivity': [250, 800],
        'Nitrate': [1.5, 8.0]
    })
    
    csv = sample_data.to_csv(index=False)
    st.download_button(
        "📥 Download Sample CSV Template",
        csv,
        "water_quality_template.csv",
        "text/csv"
    )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df_upload)} samples")
            
            # Check for required columns
            required_cols = feature_names
            missing_cols = [col for col in required_cols if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Make predictions
                X_batch = df_upload[feature_names]
                predictions = model.predict(X_batch)
                probabilities = model.predict_proba(X_batch)
                
                df_upload['Prediction'] = ['Potable' if p == 1 else 'Non-Potable' for p in predictions]
                df_upload['Confidence'] = [probabilities[i][predictions[i]] for i in range(len(predictions))]
                
                # Summary
                st.subheader("📊 Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Samples", len(df_upload))
                with col2:
                    potable_count = (predictions == 1).sum()
                    st.metric("Potable", potable_count, f"{potable_count/len(predictions)*100:.1f}%")
                with col3:
                    non_potable = (predictions == 0).sum()
                    st.metric("Non-Potable", non_potable, f"{non_potable/len(predictions)*100:.1f}%")
                
                # Visualization
                fig = px.pie(
                    values=[potable_count, non_potable],
                    names=['Potable', 'Non-Potable'],
                    title='Water Quality Distribution',
                    color_discrete_sequence=['#28a745', '#dc3545']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.subheader("📋 Detailed Results")
                st.dataframe(df_upload, use_container_width=True)
                
                # Download results
                result_csv = df_upload.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    result_csv,
                    f"water_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    type="primary"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Page 3: Pollution Forecast
elif page == "📈 Pollution Forecast":
    st.header("🔮 10-Year Pollution Trend Forecast")
    
    st.info("⚠️ This is a simulation based on typical pollution rates if no intervention is taken.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Water Quality")
        
        do_avg = st.slider("Dissolved Oxygen (mg/L)", 0.0, 15.0, 8.0, 0.1)
        bod = st.slider("BOD (mg/L)", 0.0, 10.0, 1.0, 0.1)
        total_coliform = st.slider("Total Coliform", 0.0, 10000.0, 100.0, 10.0)
        fecal_coliform = st.slider("Fecal Coliform", 0.0, 5000.0, 50.0, 10.0)
    
    with col2:
        st.subheader("Additional Parameters")
        
        ph_avg = st.slider("pH", 5.0, 9.0, 7.0, 0.1)
        conductivity = st.slider("Conductivity", 0.0, 2000.0, 300.0, 10.0)
        nitrate = st.slider("Nitrate (mg/L)", 0.0, 20.0, 1.0, 0.1)
        years = st.slider("Forecast Years", 1, 20, 10, 1)
    
    if st.button("📊 Generate Forecast", type="primary", use_container_width=True):
        sample_data = {
            'DO_Avg': do_avg,
            'BOD': bod,
            'Total_Coliform': total_coliform,
            'Fecal_Coliform': fecal_coliform,
            'pH_Avg': ph_avg,
            'Conductivity': conductivity,
            'Nitrate': nitrate
        }
        
        trends = predict_pollution_trend(sample_data, years)
        
        st.markdown("---")
        
        # Current status
        current_pred = model.predict(pd.DataFrame([sample_data]))[0]
        current_prob = model.predict_proba(pd.DataFrame([sample_data]))[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Status", 
                     "✅ Potable" if current_pred == 1 else "⚠️ Non-Potable",
                     f"{current_prob[current_pred]:.1%} confidence")
        with col2:
            final_prob = trends.iloc[-1]['Potable_Prob']
            change = final_prob - current_prob[1] * 100
            st.metric(f"Status in {years} Years", 
                     f"{final_prob:.1f}% Potable",
                     f"{change:+.1f}%",
                     delta_color="normal" if change >= 0 else "inverse")
        
        # Potability trend
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=trends['Year'],
            y=trends['Potable_Prob'],
            mode='lines+markers',
            name='Potability',
            line=dict(color='#1E88E5', width=3),
            fill='tozeroy'
        ))
        fig1.add_hline(y=50, line_dash="dash", line_color="red", 
                      annotation_text="Safety Threshold")
        fig1.update_layout(
            title="Potability Probability Over Time",
            xaxis_title="Year",
            yaxis_title="Potable Probability (%)",
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Multiple parameters
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=trends['Year'], y=trends['DO_Avg'],
                                 name='Dissolved Oxygen', mode='lines+markers'))
        fig2.add_trace(go.Scatter(x=trends['Year'], y=trends['BOD'],
                                 name='BOD', mode='lines+markers'))
        fig2.update_layout(
            title="Key Parameters Trends",
            xaxis_title="Year",
            yaxis_title="Value",
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Data table
        st.subheader("📊 Forecast Data")
        st.dataframe(trends, use_container_width=True)
        
        # Download
        csv = trends.to_csv(index=False)
        st.download_button(
            "📥 Download Forecast Data",
            csv,
            f"pollution_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

# Page 4: About
else:
    st.header("ℹ️ About Water Quality Prediction System")
    
    st.markdown("""
    ## 🎯 Purpose
    
    This AI-powered system predicts water potability based on 7 key quality parameters using 
    a Random Forest machine learning model trained on real river water data.
    
    ## 📊 Model Performance
    
    - **Accuracy:** 100% on test data
    - **F1-Score:** 1.0000
    - **ROC-AUC:** 1.0000
    - **Cross-Validation:** 96% (±4.9%)
    
    ## 🔬 Parameters Used
    
    1. **DO_Avg** - Dissolved Oxygen (mg/L)
    2. **BOD** - Biochemical Oxygen Demand (mg/L)
    3. **Total_Coliform** - Total coliform bacteria (MPN/100mL)
    4. **Fecal_Coliform** - Fecal coliform bacteria (MPN/100mL)
    5. **pH_Avg** - pH level
    6. **Conductivity** - Electrical conductivity (μmhos/cm)
    7. **Nitrate** - Nitrate concentration (mg/L)
    
    ## 🏆 Top Predictive Features
    
    1. **Total_Coliform** (25.3%) - Bacterial contamination
    2. **DO_Avg** (23.5%) - Oxygen level
    3. **BOD** (22.3%) - Organic pollution
    
    ## 📋 WHO/EPA Standards Reference
    
    - **DO:** > 5.0 mg/L (Safe)
    - **BOD:** < 3.0 mg/L (Safe)
    - **Coliform:** < 2500 MPN/100mL (Safe)
    - **pH:** 6.5 - 8.5 (Safe)
    - **Nitrate:** < 10 mg/L (Safe)
    
    ## 💡 How to Use
    
    ### Single Sample Prediction
    1. Input 7 water quality parameters
    2. Click "Predict Water Quality"
    3. View results and recommendations
    
    ### Batch Analysis
    1. Download CSV template
    2. Fill in your water quality data
    3. Upload and analyze multiple samples
    
    ### Pollution Forecast
    1. Set current water parameters
    2. Choose forecast duration (1-20 years)
    3. View predicted trends
    
    ## 🚀 Applications
    
    - **Environmental Monitoring:** Track river/lake water quality
    - **Water Treatment Plants:** Quality control verification
    - **Government Agencies:** Compliance assessment
    - **Research:** Pollution impact studies
    - **Public Health:** Water safety alerts
    
    ## 📞 Support
    
    For questions or issues, please refer to the project documentation.
    
    ---
    
    **Version:** 1.0.0  
    **Last Updated:** February 2026  
    **Model:** Random Forest Classifier  
    **Dataset:** 49 River Locations (River Sutlej, India)
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# Footer for all pages
st.markdown("---")
st.caption("💧 Water Quality Prediction System | AI-Powered Environmental Monitoring")
