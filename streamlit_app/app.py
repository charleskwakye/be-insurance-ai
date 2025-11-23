import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

st.set_page_config(
    page_title="Belgian MTPL Claim Severity Predictor",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / "best_model_backup" / "best_model.pkl"
    preprocessor_path = Path(__file__).parent.parent / "best_model_backup" / "preprocessor.pkl"
    
    if not model_path.exists() or not preprocessor_path.exists():
        st.error("Model files not found. Please run the notebook first to train and save the model.")
        st.stop()
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_model()

st.title("🚗 Belgian Motor Third-Party Liability Insurance")
st.subheader("Claim Severity Prediction")

st.markdown("""
This app predicts the expected claim severity (amount in EUR) for Belgian motor third-party liability insurance claims.
Enter the policy and vehicle details below to get a prediction.
""")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Policy Information")
        policy_year = st.number_input("Policy Year *", min_value=2000, max_value=2030, value=2008, step=1, help="Year when the policy was issued")
        policy_holder_age = st.number_input("Policy Holder Age *", min_value=18, max_value=100, value=45, step=1, help="Age of the policy holder in years")
        driver_license_age = st.number_input("Driver License Age (years) *", min_value=0, max_value=70, value=20, step=1, help="Years since driver obtained license")
        claim_time = st.selectbox("Claim Time *", ["Day", "Night"], help="Time of day when the claim occurred")
    
    with col2:
        st.markdown("### Vehicle Information")
        vehicle_age = st.number_input("Vehicle Age (years) *", min_value=0, max_value=50, value=5, step=1, help="Age of the vehicle in years")
        vehicle_brand = st.text_input("Vehicle Brand *", value="BMW", help="Manufacturer brand (e.g., BMW, Audi, Toyota)")
        vehicle_model = st.text_input("Vehicle Model *", value="320i", help="Specific model name")
        mileage = st.number_input("Mileage (km) *", min_value=0, max_value=500000, value=50000, step=1000, help="Total kilometers driven")
        vehicle_power = st.number_input("Vehicle Power (HP) *", min_value=0, max_value=1000, value=150, step=10, help="Horsepower of the vehicle")
        catalog_value = st.number_input("Catalog Value (EUR) *", min_value=0, max_value=200000, value=25000, step=1000, help="Original catalog value of the vehicle")
    
    st.caption("* Required fields")
    submitted = st.form_submit_button("Predict Claim Severity", use_container_width=True)

if submitted:
    if not all([vehicle_brand.strip(), vehicle_model.strip()]):
        st.error("Please fill in all required fields, including Vehicle Brand and Vehicle Model.")
    else:
        try:
            feature_order = [
                'policy_year', 'vehicle_age', 'policy_holder_age', 'driver_license_age',
                'vehicle_brand', 'vehicle_model', 'mileage', 'vehicle_power',
                'catalog_value', 'claim_time'
            ]
            
            input_data = pd.DataFrame({
                'policy_year': [int(policy_year)],
                'vehicle_age': [int(vehicle_age)],
                'policy_holder_age': [int(policy_holder_age)],
                'driver_license_age': [int(driver_license_age)],
                'vehicle_brand': [vehicle_brand.strip()],
                'vehicle_model': [vehicle_model.strip()],
                'mileage': [int(mileage)],
                'vehicle_power': [int(vehicle_power)],
                'catalog_value': [int(catalog_value)],
                'claim_time': [claim_time]
            })[feature_order]
            
            X_processed = preprocessor.transform(input_data)
            prediction_log = model.predict(X_processed)[0]
            prediction_eur = np.expm1(prediction_log)
            
            st.success("✅ Prediction Generated Successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col2:
                st.metric(
                    label="Predicted Claim Severity",
                    value=f"€{prediction_eur:,.2f}",
                    delta=None
                )
            
            st.markdown("---")
            st.markdown("### Input Summary")
            st.dataframe(input_data, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that all fields are filled correctly.")

st.markdown("---")
with st.expander("ℹ️ About this Model"):
    st.markdown("""
    **Model Details:**
    - Trained on Belgian Motor Third-Party Liability (beMTPL16) dataset
    - Predicts claim severity in EUR
    - Uses log-transformed target for better accuracy
    
    **Features Used:**
    - Policy information (year, holder age, license age, claim time)
    - Vehicle information (age, brand, model, mileage, power, catalog value)
    
    **Note:** This is a prediction model. Actual claim amounts may vary based on many factors not included in this model.
    """)

