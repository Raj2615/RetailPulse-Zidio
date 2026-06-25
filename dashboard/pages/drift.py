import streamlit as st
import os

st.set_page_config(page_title="Data Drift Detection", page_icon="🕵️", layout="wide")
st.title("🕵️ Data Drift Detection (Evidently AI)")

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
report_path = os.path.join(base_dir, 'dashboard', 'drift_report.html')

st.markdown("""
This page monitors the data drift in production to ensure model performance doesn't degrade. 
The report compares recent transactional data against historical training data.
""")

if os.path.exists(report_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    st.components.v1.html(html_data, height=1000, scrolling=True)
else:
    st.warning("Drift report not found. Please run `python src/drift_detection.py` to generate it.")
