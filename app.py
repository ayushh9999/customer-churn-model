import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
try:
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
    st.success("Model and Scaler loaded successfully!")
except FileNotFoundError:
    st.error("Model or Scaler file not found. Make sure 'random_forest_model.joblib' and 'scaler.joblib' are in the same directory.")
    st.stop()

# Define the features in the exact order used during training
features_order = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary', 'Gender', 'Geography_Spain', 'Geography_Germany',
                'Balance_zero', 'balance_to_salary_ratio', 'Product_usage',
                'Male_Germany', 'Male_Spain',
                'Age_Group_26-35', 'Age_Group_36-45', 'Age_Group_46-55', 'Age_Group_56-65',
                'Age_Group_66-75', 'Age_Group_76-85', 'Age_Group_86-95',
                'Tenure_Group_3-5', 'Tenure_Group_6-7', 'Tenure_Group_8-10']

st.title('Customer Churn Prediction App')
st.write('Enter customer details to predict if they will churn.')

# Create input fields for user data
with st.sidebar:
    st.header('Customer Details')
    credit_score = st.number_input('Credit Score', min_value=350, max_value=850, value=650)
    age = st.number_input('Age', min_value=18, max_value=92, value=35)
    tenure = st.number_input('Tenure (years)', min_value=0, max_value=10, value=5)
    balance = st.number_input('Balance', min_value=0.0, value=60000.0)
    num_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox('Has Credit Card?', options=[0, 1], format_func=lambda x: 'Yes' if x==1 else 'No', index=1)
    is_active_member = st.selectbox('Is Active Member?', options=[0, 1], format_func=lambda x: 'Yes' if x==1 else 'No', index=1)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=100000.0)
    gender_input = st.selectbox('Gender', options=['Female', 'Male'], index=0)
    geography_input = st.selectbox('Geography', options=['France', 'Germany', 'Spain'], index=0)

# Preprocess user input to match model's expected format
def preprocess_input(credit_score, age, tenure, balance, num_products, has_cr_card, is_active_member, 
                    estimated_salary, gender_input, geography_input):

    # Initialize all features to 0 or their default numerical value
    input_data = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'Gender': 0, # Default to Female, will be updated
        'Geography_Spain': 0,
        'Geography_Germany': 0,
        'Balance_zero': 0,
        'balance_to_salary_ratio': 0.0,
        'Product_usage': 0,
        'Male_Germany': 0,
        'Male_Spain': 0,
        'Age_Group_26-35': 0,
        'Age_Group_36-45': 0,
        'Age_Group_46-55': 0,
        'Age_Group_56-65': 0,
        'Age_Group_66-75': 0,
        'Age_Group_76-85': 0,
        'Age_Group_86-95': 0,
        'Tenure_Group_3-5': 0,
        'Tenure_Group_6-7': 0,
        'Tenure_Group_8-10': 0
    }

    # Gender encoding (assuming Female=0, Male=1 from LabelEncoder)
    if gender_input == 'Male':
        input_data['Gender'] = 1

    # Geography one-hot encoding
    if geography_input == 'Spain':
        input_data['Geography_Spain'] = 1
    elif geography_input == 'Germany':
        input_data['Geography_Germany'] = 1
    # 'France' is the base, so both remain 0

    # Engineered Features
    input_data['Balance_zero'] = 1 if balance == 0 else 0
    input_data['balance_to_salary_ratio'] = balance / estimated_salary if estimated_salary != 0 else 0.0
    input_data['Product_usage'] = num_products * is_active_member
    input_data['Male_Germany'] = input_data['Gender'] * input_data['Geography_Germany']
    input_data['Male_Spain'] = input_data['Gender'] * input_data['Geography_Spain']

    # Age Group one-hot encoding
    if 26 <= age <= 35: input_data['Age_Group_26-35'] = 1
    elif 36 <= age <= 45: input_data['Age_Group_36-45'] = 1
    elif 46 <= age <= 55: input_data['Age_Group_46-55'] = 1
    elif 56 <= age <= 65: input_data['Age_Group_56-65'] = 1
    elif 66 <= age <= 75: input_data['Age_Group_66-75'] = 1
    elif 76 <= age <= 85: input_data['Age_Group_76-85'] = 1
    elif 86 <= age <= 95: input_data['Age_Group_86-95'] = 1

    # Tenure Group one-hot encoding
    if 3 <= tenure <= 5: input_data['Tenure_Group_3-5'] = 1
    elif 6 <= tenure <= 7: input_data['Tenure_Group_6-7'] = 1
    elif 8 <= tenure <= 10: input_data['Tenure_Group_8-10'] = 1

    # Create a DataFrame with a single row, ensuring column order matches features_order
    processed_df = pd.DataFrame([input_data], columns=features_order)
    return processed_df

# When the user clicks the predict button
if st.button('Predict Churn'):
    processed_data = preprocess_input(credit_score, age, tenure, balance, num_products, has_cr_card, 
                                    is_active_member, estimated_salary, gender_input, geography_input)
    
    # Scale the processed data
    scaled_data = scaler.transform(processed_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f"The customer is likely to churn. (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.success(f"The customer is not likely to churn. (Probability: {prediction_proba[0][0]:.2f})")

    st.write("Raw input data:")
    st.write(processed_data)
    st.write("Scaled data for prediction:")
    st.write(scaled_data)
