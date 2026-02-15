import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.diabetes.org',
        'Report a bug': None,
        'About': "Diabetes Risk Prediction System - Powered by ML"
    }
)

# Custom CSS for better styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        padding: 20px;
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .prediction-box-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(81, 207, 102, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model('best_mlp (1).keras')
    
    # Create and fit scaler with typical ranges based on training data
    scaler = MinMaxScaler()
    typical_data = np.array([[0, 44, 24, 7, 14, 18.2, 0.078, 21],
                              [17, 199, 122, 99, 846, 67.1, 2.42, 81]])
    scaler.fit(typical_data)
    
    return model, scaler

model, scaler = load_model_and_scaler()

# Feature names
feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']

# Initialize session state for tabs
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/256/000000/medical-doctor.png", width=80)
    st.title("🏥 Diabetes Risk Predictor")
    st.divider()
    
    # Horizontal menu
    current_tab = option_menu(
        menu_title=None,
        options=["📊 Prediction", "📈 Analytics", "ℹ️ About"],
        icons=["graph-up", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#ff6b6b", "color": "white"}
        }
    )

# Main Content Area
if current_tab == "📊 Prediction":
    st.markdown("## 🔮 Patient Risk Assessment")
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📋 Patient Information")
        st.info("💡 Use the sliders below to input patient health metrics")
        
        pregnancies = st.slider('👶 Pregnancies', 0, 17, 3, help="Number of times pregnant")
        glucose = st.slider('🩸 Glucose Level (mg/dL)', 44, 199, 120, help="Fasting blood glucose")
        blood_pressure = st.slider('❤️ Blood Pressure (mmHg)', 24, 122, 72, help="Diastolic blood pressure")
        skin_thickness = st.slider('📏 Skin Thickness (mm)', 7, 99, 29, help="Triceps skinfold thickness")
        
    with col2:
        st.subheader("📊 Additional Metrics")
        
        insulin = st.slider('🩹 Insulin Level (mu U/ml)', 14, 846, 125, help="2-hour serum insulin")
        bmi = st.slider('⚖️ BMI (kg/m²)', 18.2, 67.1, 32.0, step=0.1, help="Body Mass Index")
        diabetes_pedigree = st.slider('🧬 Diabetes Pedigree Function', 0.078, 2.42, 0.372, step=0.01, help="Family history factor")
        age = st.slider('🎂 Age (years)', 21, 81, 29, help="Patient age")
    
    st.divider()
    
    # Create prediction data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    
    # Scale the data
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction_proba = model.predict(scaled_input, verbose=0)[0][0]
    prediction = 1 if prediction_proba > 0.5 else 0
    
    # Display Results
    st.markdown("## 🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-box-high">
                    ⚠️ HIGH RISK<br>
                    Diabetes Likely
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-box-low">
                    ✅ LOW RISK<br>
                    Diabetes Unlikely
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Risk Probability", f"{prediction_proba*100:.1f}%", 
                 delta="High Alert" if prediction_proba > 0.5 else "Normal")
    
    with col3:
        risk_level = "Critical" if prediction_proba > 0.8 else "High" if prediction_proba > 0.5 else "Moderate" if prediction_proba > 0.3 else "Low"
        st.metric("Risk Level", risk_level)
    
    st.divider()
    
    # Feature contribution visualization
    st.markdown("## 📊 Feature Analysis")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # Radar chart for input values
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[pregnancies/17*100, glucose/199*100, blood_pressure/122*100, 
               skin_thickness/99*100, insulin/846*100, bmi/67.1*100, 
               diabetes_pedigree/2.42*100, age/81*100],
            theta=feature_names,
            fill='toself',
            name='Patient Values',
            line=dict(color='#ff6b6b'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=400,
            font=dict(size=11),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Patient metrics summary
        metrics_df = pd.DataFrame({
            'Metric': feature_names,
            'Value': [pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, diabetes_pedigree, age],
            'Status': ['Normal' if v < 50 else 'Attention' for v in 
                      [pregnancies/17*100, glucose/199*100, blood_pressure/122*100, 
                       skin_thickness/99*100, insulin/846*100, bmi/67.1*100, 
                       diabetes_pedigree/2.42*100, age/81*100]]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Risk distribution gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction_proba*100,
        title={'text': "Diabetes Risk Score"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "#90EE90"},
                {'range': [33, 66], 'color': "#FFD700"},
                {'range': [66, 100], 'color': "#FF6347"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig_gauge.update_layout(height=400, font=dict(size=14), margin=dict(l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Recommendations
    st.markdown("## 💊 Health Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if glucose > 126:
            st.warning("⚠️ **High Glucose**: Consider reducing sugar intake and consulting with a doctor")
        if bmi > 30:
            st.warning("⚠️ **Elevated BMI**: Regular exercise and balanced diet recommended")
    
    with col2:
        if age > 60:
            st.info("ℹ️ **Age Factor**: Regular check-ups recommended for older adults")
        if diabetes_pedigree > 1:
            st.info("ℹ️ **Family History**: Increased risk due to family history - preventive care important")

elif current_tab == "📈 Analytics":
    st.markdown("## 📈 Model Analytics & Statistics")
    st.info("This section provides insights into the model's performance and training data")
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.metric("Model Type", "Neural Network (MLP)", help="Multi-Layer Perceptron")
        st.metric("Training Features", "8 Health Metrics")
    
    with col2:
        st.metric("Activation", "ReLU + Sigmoid", help="Hidden: ReLU, Output: Sigmoid")
        st.metric("Optimization", "Adam Optimizer")
    
    with col3:
        st.metric("Framework", "TensorFlow/Keras", help="Deep Learning Framework")
        st.metric("Data Scaling", "MinMax Scaler")
    
    st.divider()
    
    # Feature distribution
    st.subheader("📊 Feature Value Distributions")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#f0f2f6')
    
    feature_ranges = [
        (0, 17, "Pregnancies"),
        (44, 199, "Glucose"),
        (24, 122, "Blood Pressure"),
        (7, 99, "Skin Thickness"),
        (14, 846, "Insulin"),
        (18.2, 67.1, "BMI"),
        (0.078, 2.42, "Diabetes Pedigree"),
        (21, 81, "Age")
    ]
    
    axes = axes.flatten()
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9', '#fd79a8', '#a29bfe']
    
    for idx, (min_val, max_val, name) in enumerate(feature_ranges):
        range_val = np.linspace(min_val, max_val, 100)
        axes[idx].hist(range_val, bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
        axes[idx].set_title(name, fontweight='bold', fontsize=10)
        axes[idx].set_xlabel('Value', fontsize=9)
        axes[idx].set_ylabel('Frequency', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.divider()
    
    # Model information
    st.subheader("🔧 Model Architecture")
    
    with st.expander("View Model Summary"):
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.code('\n'.join(model_summary), language='text')

elif current_tab == "ℹ️ About":
    st.markdown("## ℹ️ About This Application")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This Diabetes Risk Prediction System uses machine learning to assess the likelihood 
        of a patient having diabetes based on their health metrics.
        
        ### 📚 How It Works
        1. **Data Input**: Enter patient health metrics using the interactive sliders
        2. **Model Prediction**: Our trained neural network analyzes the data
        3. **Risk Assessment**: Receive a probability score and risk level
        4. **Recommendations**: Get personalized health recommendations
        
        ### 🏥 Health Metrics Used
        - **Pregnancies**: Number of times pregnant (pregnancy history)
        - **Glucose**: Fasting blood glucose level (mg/dL)
        - **Blood Pressure**: Diastolic blood pressure (mmHg)
        - **Skin Thickness**: Triceps skinfold thickness (mm)
        - **Insulin**: 2-hour serum insulin level (mu U/ml)
        - **BMI**: Body Mass Index (kg/m²)
        - **Diabetes Pedigree Function**: Family history scoring
        - **Age**: Patient age in years
        
        ### ⚠️ Important Disclaimer
        This tool is for **informational purposes only** and should not replace professional 
        medical advice. Always consult with a healthcare provider for accurate diagnosis and treatment.
        
        ### 🔬 Model Details
        - **Algorithm**: Multi-Layer Perceptron (Neural Network)
        - **Framework**: TensorFlow/Keras
        - **Data Preprocessing**: MinMax Scaling
        - **Output**: Probability score (0-1)
        - **Threshold**: 0.5 (>0.5 = High Risk, ≤0.5 = Low Risk)
        
        ### 📞 Contact & Resources
        - [Diabetes Information](https://www.diabetes.org)
        - [WHO Diabetes Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/diabetes)
        - [CDC Diabetes Prevention](https://www.cdc.gov/diabetes)
        """)
    
    with col2:
        st.image("https://img.icons8.com/color/256/000000/dna.png", width=150)
        st.markdown("---")
        st.markdown("""
        ### 💻 Tech Stack
        - **Python** 3.x
        - **Streamlit** Web UI
        - **TensorFlow/Keras** ML
        - **Plotly** Visualizations
        - **Pandas** Data Processing
        - **Scikit-learn** Preprocessing
        
        ### 🎓 Dataset
        Based on Pima Indians Diabetes Dataset
        (PIDD)
        """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px; margin-top: 20px;'>
    <p>🏥 Diabetes Risk Prediction System | Built with ❤️ using Streamlit</p>
    <p><em>Disclaimer: This tool provides predictions only. Always consult healthcare professionals for medical decisions.</em></p>
    </div>
""", unsafe_allow_html=True)
