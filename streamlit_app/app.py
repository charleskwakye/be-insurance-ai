import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go

st.set_page_config(
    page_title="Belgian MTPL Claim Severity Predictor",
    page_icon="🚗",
    layout="wide"
)

# Constants
REPO_ID = os.getenv(
    "HF_REPO_ID", "charlesnanakwakye/belgian-mtpl-claim-severity")
FEATURE_ORDER = [
    'policy_year', 'vehicle_age', 'policy_holder_age', 'driver_license_age',
    'vehicle_brand', 'vehicle_model', 'mileage', 'vehicle_power',
    'catalog_value', 'claim_time'
]

# Model error margin based on Random Forest MAPE (~40% for insurance claims)
ERROR_MARGIN_PERCENT = 0.40


@st.cache_resource
def load_artifacts():
    """Load model, preprocessor, categories, and metrics from Hugging Face Hub."""
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID, filename="model.pkl", cache_dir="./hf_cache")
        preprocessor_path = hf_hub_download(
            repo_id=REPO_ID, filename="preprocessor.pkl", cache_dir="./hf_cache")
        categories_path = hf_hub_download(
            repo_id=REPO_ID, filename="categories.json", cache_dir="./hf_cache")

        # Try to load metrics if available
        try:
            metrics_path = hf_hub_download(
                repo_id=REPO_ID, filename="metric_info.json", cache_dir="./hf_cache")
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except:
            metrics = None

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        with open(categories_path, 'r') as f:
            categories = json.load(f)

        return model, preprocessor, categories, metrics
    except Exception as e:
        st.error(f"Failed to load artifacts: {str(e)}")
        st.stop()


def create_gauge_chart(value, min_val, max_val, title="Predicted Claim Severity"):
    """Create a gauge chart for the prediction."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'prefix': "€", 'valueformat': ",.0f"},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, max(max_val * 1.2, 10000)], 'tickprefix': '€'},
            'bar': {'color': "#2E86AB"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_val * 0.5], 'color': '#90EE90'},
                {'range': [max_val * 0.5, max_val * 0.8], 'color': '#FFD700'},
                {'range': [max_val * 0.8, max_val * 1.2], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#333", 'family': "Arial"}
    )
    return fig


def create_range_chart(low, mid, high):
    """Create a horizontal bar showing prediction range."""
    fig = go.Figure()

    # Add range bar
    fig.add_trace(go.Bar(
        y=['Prediction Range'],
        x=[high - low],
        base=[low],
        orientation='h',
        marker=dict(
            color='rgba(46, 134, 171, 0.3)',
            line=dict(color='rgba(46, 134, 171, 1)', width=2)
        ),
        name='Uncertainty Range',
        hovertemplate='Range: €%{base:,.0f} - €%{x:,.0f}<extra></extra>'
    ))

    # Add point estimate marker
    fig.add_trace(go.Scatter(
        x=[mid],
        y=['Prediction Range'],
        mode='markers',
        marker=dict(size=20, color='#2E86AB', symbol='diamond'),
        name='Best Estimate',
        hovertemplate='Best Estimate: €%{x:,.0f}<extra></extra>'
    ))

    # Add annotations for values
    fig.add_annotation(x=low, y='Prediction Range', text=f"€{low:,.0f}",
                       showarrow=False, yshift=-25, font=dict(size=12, color='#666'))
    fig.add_annotation(x=mid, y='Prediction Range', text=f"€{mid:,.0f}",
                       showarrow=False, yshift=25, font=dict(size=14, color='#2E86AB', weight='bold'))
    fig.add_annotation(x=high, y='Prediction Range', text=f"€{high:,.0f}",
                       showarrow=False, yshift=-25, font=dict(size=12, color='#666'))

    fig.update_layout(
        height=150,
        showlegend=False,
        xaxis=dict(
            title='Claim Severity (EUR)',
            tickprefix='€',
            tickformat=',',
            range=[max(0, low * 0.8), high * 1.2]
        ),
        yaxis=dict(showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_risk_factors_chart(input_data):
    """Create a simple risk factor visualization."""
    # Calculate relative risk scores (simplified heuristic)
    risk_scores = {}

    # Vehicle age risk (older = higher risk)
    vehicle_age = input_data['vehicle_age'].values[0]
    risk_scores['Vehicle Age'] = min(vehicle_age / 15 * 100, 100)

    # Driver experience (less experience = higher risk)
    license_age = input_data['driver_license_age'].values[0]
    risk_scores['Driver Experience'] = max(0, 100 - (license_age / 30 * 100))

    # Vehicle power (more power = higher risk)
    power = input_data['vehicle_power'].values[0]
    risk_scores['Vehicle Power'] = min((power - 50) / 300 * 100, 100)

    # Mileage (more mileage = higher exposure)
    mileage = input_data['mileage'].values[0]
    risk_scores['Mileage Exposure'] = min(mileage / 200000 * 100, 100)

    # Time of claim
    claim_time = input_data['claim_time'].values[0]
    risk_scores['Time Risk'] = 70 if claim_time == 'Night' else 30

    factors = list(risk_scores.keys())
    scores = list(risk_scores.values())
    colors = ['#FF6B6B' if s > 60 else '#FFD700' if s >
              40 else '#90EE90' for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=factors,
        orientation='h',
        marker=dict(color=colors, line=dict(color='#333', width=1)),
        text=[f'{s:.0f}%' for s in scores],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))

    fig.update_layout(
        title=dict(text='Risk Factor Analysis', font=dict(size=16)),
        height=250,
        xaxis=dict(title='Relative Risk Score', range=[0, 100]),
        yaxis=dict(title=''),
        margin=dict(l=20, r=20, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# Load artifacts
model, preprocessor, categories, metrics = load_artifacts()

vehicle_brands = categories.get('vehicle_brand', [])
vehicle_models = categories.get('vehicle_model', [])
claim_times = categories.get('claim_time', ['Day', 'Night'])

# Sidebar for inputs
with st.sidebar:
    st.header("🚗 Input Parameters")

    with st.form("prediction_form"):
        st.subheader("Policy Information")

        policy_year = st.slider(
            "Policy Year", min_value=2004, max_value=2025, value=2020)
        policy_holder_age = st.slider(
            "Policy Holder Age", min_value=18, max_value=85, value=40)
        driver_license_age = st.slider(
            "Driver License Age (years)", min_value=0, max_value=65, value=15)
        claim_time = st.selectbox("Claim Time", options=claim_times)

        st.divider()
        st.subheader("Vehicle Information")

        vehicle_age = st.slider("Vehicle Age (years)",
                                min_value=0, max_value=25, value=5)
        vehicle_brand = st.selectbox("Vehicle Brand", options=vehicle_brands)
        vehicle_model = st.selectbox("Vehicle Model", options=vehicle_models)
        mileage = st.slider("Mileage (km)", min_value=0,
                            max_value=300000, value=50000, step=5000)
        vehicle_power = st.slider(
            "Vehicle Power (HP)", min_value=50, max_value=400, value=120, step=10)
        catalog_value = st.slider(
            "Catalog Value (EUR)", min_value=5000, max_value=150000, value=20000, step=1000)

        st.divider()
        submitted = st.form_submit_button(
            "🔮 Predict Claim Severity", use_container_width=True, type="primary")

# Main page header
st.title("🚗 Belgian Motor Insurance")
st.subheader("Claim Severity Prediction")

st.markdown("""
Predict the expected claim severity for Belgian motor third-party liability insurance.
The prediction includes an **uncertainty range** based on model error margins.

👈 **Enter your policy and vehicle details in the sidebar to get started.**
""")

if submitted:
    try:
        input_data = pd.DataFrame({
            'policy_year': [int(policy_year)],
            'vehicle_age': [int(vehicle_age)],
            'policy_holder_age': [int(policy_holder_age)],
            'driver_license_age': [int(driver_license_age)],
            'vehicle_brand': [str(vehicle_brand).upper()],
            'vehicle_model': [str(vehicle_model).upper()],
            'mileage': [int(mileage)],
            'vehicle_power': [int(vehicle_power)],
            'catalog_value': [int(catalog_value)],
            'claim_time': [str(claim_time)]
        })[FEATURE_ORDER]

        X_processed = preprocessor.transform(input_data)

        if np.any(np.isinf(X_processed)) or np.any(np.isnan(X_processed)):
            st.error("Invalid values detected after preprocessing.")
            st.stop()

        prediction = model.predict(X_processed)[0]

        if np.isinf(prediction) or np.isnan(prediction):
            st.error("Model returned an invalid prediction.")
            st.stop()

        # Model predicts in log1p scale, convert back to EUR
        # (Values in log scale are typically < 20, EUR values are > 100)
        prediction_eur = prediction if prediction > 100 else np.expm1(
            prediction)
        prediction_eur = max(0, prediction_eur)

        # Calculate prediction range based on error margin
        low_estimate = max(0, prediction_eur * (1 - ERROR_MARGIN_PERCENT))
        high_estimate = prediction_eur * (1 + ERROR_MARGIN_PERCENT)

        st.divider()

        # Results section
        st.markdown("### Prediction Results")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.metric(
                label="Low Estimate",
                value=f"€{low_estimate:,.0f}",
                help="Conservative estimate (best case)"
            )

        with col2:
            st.metric(
                label="Best Estimate",
                value=f"€{prediction_eur:,.0f}",
                help="Most likely claim severity"
            )

        with col3:
            st.metric(
                label="High Estimate",
                value=f"€{high_estimate:,.0f}",
                help="Upper bound estimate (worst case)"
            )

        # Prediction range visualization
        st.plotly_chart(
            create_range_chart(low_estimate, prediction_eur, high_estimate),
            use_container_width=True
        )

        # Two column layout for additional charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.plotly_chart(
                create_gauge_chart(
                    prediction_eur, low_estimate, high_estimate),
                use_container_width=True
            )

        with chart_col2:
            st.plotly_chart(
                create_risk_factors_chart(input_data),
                use_container_width=True
            )

        # Confidence note
        st.info(
            f"**Prediction Confidence**: The range shown represents ±{ERROR_MARGIN_PERCENT*100:.0f}% "
            f"uncertainty based on typical model error. Actual claim amounts may vary based on "
            f"specific circumstances not captured in the input features."
        )

        with st.expander("View Input Summary"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        with st.expander("Debug Information"):
            import traceback
            st.code(traceback.format_exc())

st.divider()

with st.expander("About This Model"):
    st.markdown(f"""
    **Model Type**: Random Forest Regressor (optimized for individual claim accuracy)  
    **Training Data**: Belgian MTPL (beMTPL16) dataset - 70,000+ claims  
    **Source**: [Hugging Face Hub]({f"https://huggingface.co/{REPO_ID}"})  
    **Features**: {len(vehicle_brands)} vehicle brands, {len(vehicle_models)} models
    
    This model uses ensemble decision trees with log-transformed targets, selected for achieving 
    the lowest MAPE (Mean Absolute Percentage Error) - best for individual claim predictions.
    """)
