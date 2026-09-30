import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model.detector import BurnInAnomalyDetector

st.set_page_config(page_title="ISRO AI Burn-In Screener", layout="wide")

st.title("🚀 AI-Driven Anomaly Detection in Component Burn-In & Screening")
st.markdown("**SIH Problem Statement ID: 26170** | Organization: Indian Space Research Organisation (ISRO)")

# Sidebar for controls
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Component Telemetry CSV", type=["csv"])

@st.cache_data
def load_default_data():
    np.random.seed(42)
    n_samples = 150
    data = {
        'Component_ID': [f'ISRO-COMP-{1000+i}' for i in range(n_samples)],
        'Batch_ID': np.random.choice(['BATCH-A', 'BATCH-B', 'BATCH-C'], n_samples),
        'iddq_0h': np.random.normal(10.0, 0.5, n_samples),
        'iddq_24h': np.random.normal(10.2, 0.6, n_samples),
        'iddq_96h': np.random.normal(10.3, 0.8, n_samples),
        'iddq_168h': np.random.normal(10.4, 0.9, n_samples),
        'leakage_current_168h': np.random.normal(2.1, 0.2, n_samples),
        'propagation_delay_ns': np.random.normal(15.0, 1.2, n_samples)
    }
    # Inject synthetic drift anomalies
    anomaly_indices = [12, 45, 88, 102, 134]
    data['iddq_168h'][anomaly_indices] += np.random.uniform(5.0, 10.0, len(anomaly_indices))
    data['leakage_current_168h'][anomaly_indices] += np.random.uniform(3.0, 6.0, len(anomaly_indices))
    
    return pd.DataFrame(data)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()

st.subheader("📊 Raw Screening Telemetry Preview")
st.dataframe(df.head(), use_container_width=True)

# Run Detection Model
detector = BurnInAnomalyDetector()
processed_df, features = detector.fit_predict(df)

st.divider()
st.subheader("🔍 Screening Results & Latent Defect Identification")

col1, col2, col3 = st.columns(3)
total_comps = len(processed_df)
flagged_comps = len(processed_df[processed_df['Status'] == 'Review / Latent Risk'])
pass_comps = total_comps - flagged_comps

col1.metric("Total Components Screened", total_comps)
col2.metric("Passed Static & Drift Check", pass_comps)
col3.metric("Flagged for Latent Drift Risk", flagged_comps, delta_color="inverse")

# Display Filtered Results
status_filter = st.selectbox("Filter by Status", ["All", "Review / Latent Risk", "Pass"])
if status_filter != "All":
    filtered_view = processed_df[processed_df['Status'] == status_filter]
else:
    filtered_view = processed_df

st.dataframe(filtered_view[['Component_ID', 'Batch_ID', 'Status', 'Anomaly_Score'] + features], use_container_width=True)

# Visualization
st.subheader("📈 Time-Series Parametric Drift Analysis ($0\text{h} \rightarrow 168\text{h}$)")
if 'iddq_0h' in processed_df.columns and 'iddq_168h' in processed_df.columns:
    fig = px.scatter(
        processed_df, 
        x='iddq_0h', 
        y='iddq_168h', 
        color='Status',
        hover_data=['Component_ID', 'Batch_ID'],
        title="Standby Current ($I_{DDQ}$) Drift: Initial ($0\text{h}$) vs Final ($168\text{h}$)"
    )
    st.plotly_chart(fig, use_container_width=True)