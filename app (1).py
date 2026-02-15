import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the trained model and scaler
best_lr_model = joblib.load('tuned_logistic_regression_model.pkl')
scaler = joblib.load('minmax_scaler.pkl')

# Define feature names (consistent with training data)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                 'BMI', 'DiabetesPedigreeFunction', 'Age']

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title('Diabetes Prediction App')
st.write('Enter the patient details to predict the likelihood of diabetes.')

# Create input fields for user
with st.sidebar:
    st.header('Patient Input Features')
    pregnancies = st.slider('Pregnancies', 0, 17, 3)
    glucose = st.slider('Glucose', 44, 199, 120)
    blood_pressure = st.slider('Blood Pressure', 24, 122, 72)
    skin_thickness = st.slider('Skin Thickness', 7, 99, 29)
    insulin = st.slider('Insulin', 14, 846, 125)
    bmi = st.slider('BMI', 18.2, 67.1, 32.0)
    diabetes_pedigree_function = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.372, step=0.001)
    age = st.slider('Age', 21, 81, 29)

# Create a DataFrame from user input
input_data = pd.DataFrame([{
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': diabetes_pedigree_function,
    'Age': age
}])

# Scale the input data using the loaded scaler
# Note: The scaler was fit on specific columns in a specific order.
# Ensure the order and columns used for scaling here match the training.
# The scaler was fitted on ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']
scaled_input_data = input_data.copy()
scaled_input_data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']] = scaler.transform(
    input_data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']]
)

# Make prediction
prediction = best_lr_model.predict(scaled_input_data)
prediction_proba = best_lr_model.predict_proba(scaled_input_data)[:, 1]

st.subheader('Prediction Result')
if prediction[0] == 1:
    st.error('The patient is predicted to have Diabetes.')
else:
    st.success('The patient is predicted NOT to have Diabetes.')

st.write(f'Probability of Diabetes: {prediction_proba[0]:.2f}')

st.subheader('Explanation of Prediction (SHAP Values)')

# Create a SHAP explainer
explainer = shap.LinearExplainer(best_lr_model, scaled_input_data)
shap_values = explainer.shap_values(scaled_input_data)

# Plot SHAP values
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, scaled_input_data, feature_names=feature_names, show=False, ax=ax)
st.pyplot(fig)

st.write("The SHAP summary plot shows how each feature contributes to the prediction. Red points indicate features that push the prediction higher (towards diabetes), while blue points indicate features that push the prediction lower (away from diabetes).")

st.subheader('How to run this app locally:')
st.markdown("""
1. Save the code above as `app.py` in the same directory as your trained models (`tuned_logistic_regression_model.pkl`, `minmax_scaler.pkl`).
2. Make sure you have all the required libraries installed (`pip install -r requirements.txt`).
3. Open your terminal or command prompt.
4. Navigate to the directory where you saved `app.py`.
5. Run the command: `streamlit run app.py`
6. Your web browser will open with the Streamlit application.
""")
