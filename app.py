import random
import pandas as pd
import streamlit as st
import json
import time
import re
import numpy as np
import plotly.express as px
from datetime import datetime
from groq import Groq

st.set_page_config(page_title="Kano Model Feature Evaluation", page_icon="🤖", layout="wide")

# Sidebar
with st.sidebar:
    st.title("⚙️ A Dr. Pricing App")
    api_key = st.secrets["groq"]["api_key"]
    st.markdown("---")
    st.markdown("### How does it work?")
    st.markdown(
        """
        1. In the **Setup** tab, enter your product details and feature list.
        2. Synthetic respondents evaluate the features.
        3. In **Results**, view Kano statistics, classifications, and diagrams.
        """
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool uses the Kano Model to evaluate product features.")

st.title('🤖 Kano Model Feature Evaluation')
tab1, tab2 = st.tabs(["Setup", "Results"])

# -----------------------------
# TAB 1: Setup
# -----------------------------
with tab1:
    st.header("Setup")
    
    # Initialize session state if not present
    for key in ["start_experiment", "experiment_complete", "results"]:
        if key not in st.session_state:
            st.session_state[key] = False if key != "results" else None
    
    # Input fields
    st.subheader("Product Name")
    product_name = st.text_input('Enter product name', key="product_name")
    
    st.subheader("Target Customers")
    target_customers = st.text_area('Describe your target customers', height=150, key="target_customers")
    
    st.subheader("Features")
    features_input = st.text_area('List features (one per line)', height=150, key="features")
    
    st.subheader("Number of Synthetic Respondents")
    num_respondents = st.number_input('Number of respondents', min_value=1, max_value=100, value=7, key="num_respondents")
    
    if st.button('🚀 Start Survey', type="primary"):
        if not api_key:
            st.error("Please provide your Groq API key in the sidebar.")
        elif not product_name or not target_customers or not features_input:
            st.error("Please fill in all required fields.")
        else:
            st.session_state.start_experiment = True
            st.session_state.experiment_complete = False
            st.session_state.results = None

if st.session_state.start_experiment:
    with tab1:
        st.header("Survey Synthetic Respondents")
        progress_bar = st.progress(0)
        
        client = Groq(api_key=api_key)
        
        profiles = []
        personas = []
        MAX_RETRIES = 3
        RETRY_DELAY = 10
        
        # Generate synthetic respondent profiles
        for i in range(num_respondents):
            progress_bar.progress((i + 1) / (num_respondents * 2))
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    profile_resp = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "Create a synthetic respondent profile including age and gender based on the target customer description:"},
                            {"role": "user", "content": target_customers}
                        ],
                        temperature=0.1
                    )
                    profile_data = json.loads(profile_resp.choices[0].message.content)
                    profiles.append(profile_data)
                    time.sleep(2)
                    break
                except Exception:
                    retries += 1
                    time.sleep(RETRY_DELAY)
        profiles_df = pd.DataFrame(profiles)
        
        # Generate hidden persona descriptions
        for i, row in profiles_df.iterrows():
            progress_bar.progress((i + 1 + num_respondents) / (num_respondents * 2))
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    persona_resp = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "Create a customer persona based on:"},
                            {"role": "user", "content": f"Age: {row['Age']}; Gender: {row['Gender']}"}
                        ],
                        temperature=0.1
                    )
                    personas.append(persona_resp.choices[0].message.content)
                    time.sleep(2)
                    break
                except Exception:
                    retries += 1
                    time.sleep(RETRY_DELAY)
        if personas:
            profiles_df["Persona"] = personas 
        features = [f.strip() for f in features_input.splitlines() if f.strip()]
        kano_responses = []
        # Fetch Kano ratings
        for i, row in profiles_df.iterrows():
            progress_bar.progress((i + 1 + num_respondents) / (num_respondents * 2))
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    rating_resp = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": """
                            Evaluate product features using the Kano model. Return JSON format:
                            {"feature_name": {"functional": {"rating": X}, "dysfunctional": {"rating": X}}}
                            """},
                            {"role": "user", "content": f"Features: {features}"}
                        ],
                        temperature=1
                    )
                    kano_responses.append(rating_resp.choices[0].message.content)
                    time.sleep(2)
                    break
                except Exception:
                    retries += 1
                    time.sleep(RETRY_DELAY)
        progress_bar.progress(1.0)
        st.session_state.results = {"profiles": profiles_df, "responses": kano_responses, "features": features}
        st.session_state.experiment_complete = True
        st.success("✅ Survey completed! View results in 'Results'.")

# -----------------------------
# TAB 2: Results
# -----------------------------
with tab2:
    if not st.session_state.experiment_complete:
        st.info("Run the survey first.")
    else:
        st.header("Results")
        show_persona = st.checkbox("Show Persona Details", value=False)
        
        st.write("### Respondent Profiles")
        profiles_df = st.session_state.results["profiles"].copy()
        profiles_df.index += 1 
        
        # Debugging: Print columns and DataFrame head
        print(profiles_df.columns)
        print(profiles_df.head())
        
        if not show_persona and "Persona" in profiles_df.columns:
            profiles_df = profiles_df.drop(columns=["Persona"], errors='ignore')
        st.dataframe(profiles_df)
        
        kano_responses = st.session_state.results["responses"]
        classifications = []
        for i, resp in enumerate(kano_responses):
            try:
                parsed_json = json.loads(resp)
                for feature, data in parsed_json.items():
                    f_score = int(data["functional"]["rating"])
                    d_score = int(data["dysfunctional"]["rating"])
                    category = "Excitement" if f_score == 1 and d_score >= 4 else "Expected"
                    classifications.append({"Feature": feature, "Functional": f_score, "Dysfunctional": d_score, "Kano": category})
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
        if classifications:
            kano_df = pd.DataFrame(classifications)
            st.write("### Kano Classification Table")
            st.dataframe(kano_df)
            freq_df = kano_df.groupby(["Feature", "Kano"]).size().reset_index(name="Count")
            fig = px.bar(
                freq_df,
                x="Feature",
                y="Count",
                color="Kano",
                text="Count",
                title="Kano Classification Counts per Feature",
                barmode="stack" # Ensures a stacked column chart
            )
            
            fig.update_layout(
                xaxis_title="Feature",
                yaxis_title="Number of Responses",
                legend_title="Kano Classification",
                uniformtext_minsize=12,
                uniformtext_mode='hide'
            )
            
            st.plotly_chart(fig)
            st.download_button("Download Kano Data", kano_df.to_csv().encode(), "kano_results.csv")
        else:
            st.warning("🚨 No valid Kano classifications found.")
