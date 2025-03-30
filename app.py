import random
import pandas as pd
import streamlit as st
import os
import json
import time
import numpy as np
import plotly.express as px
from datetime import datetime
from groq import Groq
from scipy import stats

st.set_page_config(page_title="Kano Model Feature Evaluation", page_icon="🤖", layout="wide")

with st.sidebar:
    st.title("⚙️ Instructions")
    api_key = st.secrets["groq"]["api_key"]
    st.markdown("---")
    st.markdown("### How does it work?")
    st.markdown("""
    1. Setup the survey in the **Setup** tab. 
    2. Analyze results in **Results**.
    """)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool evaluates features using a Kano Model approach.")

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
            ages = range(18, 78)
            genders = ["Male", "Female", "Unknown"]
            profiles = [{"Age": random.choice(ages), "Gender": random.choice(genders)} for _ in range(num_respondents)]
            profiles_df = pd.DataFrame(profiles)
            MAX_RETRIES = 3
            RETRY_DELAY = 10
            
            personas = []
            for i, row in profiles_df.iterrows():
                progress_bar.progress((i + 1) / (num_respondents * 2))
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        response = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {"role": "system", "content": "Create a customer persona based on:"},
                                {"role": "user", "content": f"Age: {row['Age']}; Gender: {row['Gender']}"}
                            ],
                            temperature=0
                        )
                        personas.append(response.choices[0].message.content)
                        time.sleep(5)
                        break
                    except Exception:
                        retries += 1
                        time.sleep(RETRY_DELAY)
            profiles_df["Persona"] = personas
            
            features = [f.strip() for f in features_input.splitlines() if f.strip()]
            kano_responses = []
            for i, row in profiles_df.iterrows():
                progress_bar.progress((i + 1 + num_respondents) / (num_respondents * 2))
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        response = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {"role": "system", "content": "Only return a JSON object as described."},
                                {"role": "user", "content": f"Evaluate features based on: {row['Persona']}"}
                            ],
                            temperature=0
                        )
                        kano_responses.append(response.choices[0].message.content)
                        time.sleep(5)
                        break
                    except Exception:
                        retries += 1
                        time.sleep(RETRY_DELAY)
            
            progress_bar.progress(1.0)  # Ensure it reaches 100%
            st.session_state.results = {"profiles": profiles_df, "responses": kano_responses, "features": features}
            st.session_state.experiment_complete = True
            st.success("✅ Survey completed! View results in 'Results'.")

with tab2:
    if not st.session_state.experiment_complete:
        st.info("Run the survey first.")
    else:
        st.header("Results")
        
        st.write("### Respondent Profiles")
        profiles_df = st.session_state.results["profiles"].copy()
        profiles_df.index = profiles_df.index + 1  # Fix indexing to start from 1
        st.dataframe(profiles_df)
        
        st.write("### Kano Evaluations")
        kano_responses = st.session_state.results["responses"]
        
        # Ensure responses exist
        if not kano_responses:
            st.warning("No Kano responses available. Ensure survey responses are properly formatted.")
            st.stop()

        # Debugging: Print responses to check if they exist
        st.write("📌 Debug: Kano Responses Length:", len(kano_responses))

        # Kano classification rules
        rating_map = {
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, 
            "I like it": 1, "I expect it": 2, "I am indifferent": 3, 
            "I can live with it": 4, "I dislike it": 5
        }

        def classify_kano(f, d):
            if f == 1 and d >= 4:
                return "Excitement"
            elif f == 2 and d == 5:
                return "Must-Have"
            elif f == 3 and d == 3:
                return "Indifferent"
            else:
                return "Expected"

        classifications = []
        debug_response_shown = False  # Only show debug output once

        for resp in kano_responses:
            try:
                parsed_json = json.loads(resp)

                # Show an example response for debugging
                if not debug_response_shown:
                    st.write("📌 Debug: Example API Response", parsed_json)
                    debug_response_shown = True

                if "features" not in parsed_json:
                    st.warning(f"Missing 'features' in response: {parsed_json}")
                    continue  # Skip if 'features' is missing

                for feat_obj in parsed_json.get("features", []):
                    if "feature" not in feat_obj or "when_present" not in feat_obj or "when_absent" not in feat_obj:
                        st.warning(f"Skipping invalid entry: {feat_obj}")
                        continue  # Skip this entry if essential keys are missing

                    f = rating_map.get(str(feat_obj["when_present"]), feat_obj["when_present"])
                    d = rating_map.get(str(feat_obj["when_absent"]), feat_obj["when_absent"])
                    classification = classify_kano(f, d)

                    classifications.append({
                        "Feature": feat_obj["feature"],
                        "Present": f,
                        "Absent": d,
                        "Net Score": f - d,
                        "Classification": classification
                    })
            except json.JSONDecodeError as e:
                st.warning(f"JSON parsing error: {e}")

        # Display results if valid classifications exist
        if classifications:
            kano_df = pd.DataFrame(classifications)
            kano_df.index = kano_df.index + 1  # Fix indexing
            st.dataframe(kano_df)

            csv = kano_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Kano Results", data=csv, file_name="kano_results.csv", mime="text/csv")
        else:
            st.warning("No valid Kano classifications found. Ensure survey responses are properly formatted.")
