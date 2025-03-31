import random
import pandas as pd
import streamlit as st
import json
import time
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
        1. In the **Setup** tab, you enter your product details and feature list.
        2. Synthetic respondents (aligned with target customer description) evaluate the features.
        3. In **Results**, you see Kano evaluation statistics, classifications in tables and diagrams.
        """
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool uses a Kano Model approach to evaluate product features.")

st.title('🤖 Kano Model Feature Evaluation')
tab1, tab2 = st.tabs(["Setup", "Results"])

with tab1:
    st.header("Setup")
    
    if 'start_experiment' not in st.session_state:
        st.session_state.start_experiment = False
    if 'experiment_complete' not in st.session_state:
        st.session_state.experiment_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None

    st.subheader("Product Name")
    product_name = st.text_input('Enter product name', key="product_name")
    
    st.subheader("Target Customers")
    target_customers = st.text_area('Describe your target customers', height=150, key="target_customers")
    
    st.subheader("Features")
    features_input = st.text_area('List features (one per line)', height=150, key="features")
    
    st.subheader("Number of Synthetic Respondents")
    num_respondents = st.number_input('Number of respondents', min_value=1, max_value=100, value=8, key="num_respondents")
    
    if st.button('🚀 Start Survey', type="primary"):
        if not api_key:
            st.error("Please provide your Groq API key in the sidebar.")
        elif not product_name or not target_customers or not features_input:
            st.error("Please fill in all required fields.")
        else:
            st.session_state.start_experiment = True
            st.session_state.experiment_complete = False
            st.session_state.results = None

            st.header("Survey Synthetic Respondents")
            progress_bar = st.progress(0)
            client = Groq(api_key=api_key)

            # Generate synthetic respondent profiles based on the target customer description
            profiles = []
            for i in range(num_respondents):
                progress_bar.progress(i / (num_respondents * 2))
                retries = 0
                while retries < 3:
                    try:
                        persona_resp = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {"role": "system", "content": "Create a customer persona that fits this target audience:"},
                                {"role": "user", "content": f"Target customers: {target_customers}"}
                            ],
                            temperature=0.1
                        )
                        profiles.append(persona_resp.choices[0].message.content)
                        time.sleep(2)
                        break
                    except Exception:
                        retries += 1
                        time.sleep(10)
            
            profiles_df = pd.DataFrame({"Persona": profiles})
            
            # Prepare feature list
            features = [f.strip() for f in features_input.splitlines() if f.strip()]
            kano_responses = []

            # Fetch Kano ratings for each synthetic respondent (now considering target customers)
            for i, persona in enumerate(profiles_df["Persona"]):
                progress_bar.progress((i + num_respondents) / (num_respondents * 2))
                retries = 0
                while retries < 3:
                    try:
                        rating_resp = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {"role": "system", "content": """
                                    You are a synthetic respondent evaluating product features using the Kano model.
                                    Your preferences are influenced by the target customer description and your persona.
                                    For each feature provided, rate it under two conditions:
                                    - Functional condition (feature present)
                                    - Dysfunctional condition (feature absent)
                                    Use a scale of 1 to 5 where:
                                      1: I like it,
                                      2: I expect it,
                                      3: I am indifferent,
                                      4: I can live with it,
                                      5: I dislike it.
                                    Return ONLY the ratings in the following JSON format:
                                    {"feature_name": {"functional": {"rating": X}, "dysfunctional": {"rating": X}}}
                                """},
                                {"role": "user", "content": f"Persona: {persona}\nTarget Customers: {target_customers}\nFeatures: {features}"}
                            ],
                            temperature=1
                        )
                        kano_responses.append(rating_resp.choices[0].message.content)
                        time.sleep(2)
                        break
                    except Exception:
                        retries += 1
                        time.sleep(10)

            progress_bar.progress(1.0)
            st.session_state.results = {"profiles": profiles_df, "responses": kano_responses, "features": features}
            st.session_state.experiment_complete = True
            st.success("✅ Survey completed! View results in 'Results'.")
