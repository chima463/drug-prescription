import os
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title='Drug Prediction App', page_icon='💊', layout='wide')

st.title('Drug Prediction App')
st.write('Enter the patient details to predict the prescribed drug.')
data_path = os.path.join(os.getcwd(), 'combined_dataset.csv')
best_model_path = os.path.join(os.getcwd(), 'dt_model.pkl')
label_encoder_path = os.path.join(os.getcwd(), 'label_encoder.pkl')

# load all resources
@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    best_model = joblib.load(best_model_path)
    label_encoder = joblib.load(label_encoder_path)
    return df, best_model, label_encoder

try:
    df_unique_values, best_model, label_encoder = load_data()
except FileNotFoundError:
    st.error("combined_dataset.csv not found. Please make sure it's in the same directory.")
    st.stop()

# Create two columns for the input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Info & History")
    age = st.number_input('Age', value=40)  # average adult age
    gender = st.selectbox('Gender', df_unique_values['Gender'].unique())
    height_cm = st.number_input('Height (cm)', value=170.0)  # avg adult height
    weight_kg = st.number_input('Weight (kg)', value=70.0)   # avg adult weight
    blood_type = st.selectbox('Blood Type', df_unique_values['Blood_Type'].unique())
    allergies = st.selectbox('Allergies', df_unique_values['Allergies'].unique())
    past_diseases = st.selectbox('Past Diseases', df_unique_values['Past_Diseases'].unique())
    past_drugs_used = st.selectbox('Past Drugs Used', df_unique_values['Past_Drugs_Used'].unique())

with col2:
    st.subheader("Lifestyle & Vitals")
    smoking_status = st.selectbox('Smoking Status', df_unique_values['Smoking Status'].unique())
    alcohol_consumption = st.selectbox('Alcohol Consumption', df_unique_values['Alcohol Consumption'].unique())
    exercise_frequency = st.selectbox('Exercise Frequency', df_unique_values['Exercise Frequency'].unique())
    co_morbidities = st.selectbox('Co-morbidities', df_unique_values['Co-morbidities'].unique())
    blood_pressure = st.text_input('Current Blood Pressure (e.g. 120/80)', value="120/80")
    avg_blood_sugar_8_months = st.number_input('Avg Blood Sugar (8 mo)', value=90.0)  # mg/dL
    avg_systolic_bp_8_months = st.number_input('Avg Systolic BP (8 mo)', value=120.0) # mmHg
    avg_diastolic_bp_8_months = st.number_input('Avg Diastolic BP (8 mo)', value=80.0) # mmHg

if st.button('Predict Drug'):
    # Validate blood pressure format
    if "/" not in blood_pressure:
        st.error("❌ Invalid blood pressure format. Please enter in the format systolic/diastolic (e.g., 120/80).")
        st.stop()

    try:
        systolic, diastolic = blood_pressure.split("/")
        systolic, diastolic = int(systolic.strip()), int(diastolic.strip())
    except ValueError:
        st.error("❌ Invalid blood pressure values. Please enter numbers in the format systolic/diastolic (e.g., 120/80).")
        st.stop()

    # collect input into a DataFrame
    input_data = pd.DataFrame([[age, gender, smoking_status, alcohol_consumption, co_morbidities,
                                exercise_frequency, blood_pressure, avg_blood_sugar_8_months,
                                avg_systolic_bp_8_months, avg_diastolic_bp_8_months, weight_kg,
                                allergies, past_diseases, past_drugs_used, height_cm, blood_type]],
                              columns=['Age', 'Gender', 'Smoking Status', 'Alcohol Consumption',
                                       'Co-morbidities', 'Exercise Frequency', 'Blood Pressure',
                                       'Avg Blood Sugar 8 Months', 'Avg Systolic BP 8 Months',
                                       'Avg Diastolic BP 8 Months', 'Weight_kg', 'Allergies',
                                       'Past_Diseases', 'Past_Drugs_Used', 'Height_cm', 'Blood_Type'])

    # add systolic/diastolic as numeric columns
    input_data['Systolic BP'] = systolic
    input_data['Diastolic BP'] = diastolic

    input_data.drop('Blood Pressure', axis=1, inplace=True)
    prediction_encoded = best_model.predict(input_data)
    predicted_drug = label_encoder.inverse_transform(prediction_encoded)

    st.success(f'✅ Predicted Drug: {predicted_drug[0]}')
