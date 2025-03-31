import random
import pandas as pd
import streamlit as st
import os
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

# --- TAB 1: Setup ---
with tab1:
    st.header("Setup")
    
    if 'start_experiment' not in st.session_state:
        st.session_state.start_experiment = False
    if 'experiment_complete' not in st.session_state:
        st.session_state.experiment_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Inputs
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

            # Generate random respondent profiles
            ages = range(18, 78)
            genders = ["Male", "Female", "Unknown"]
            profiles = [{"Age": random.choice(ages), "Gender": random.choice(genders)} for _ in range(num_respondents)]
            profiles_df = pd.DataFrame(profiles)

            MAX_RETRIES = 3
            RETRY_DELAY = 10
            personas = []

            # Fetch persona descriptions (Optional, not needed for new approach)
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

            # Fetch Kano responses (Only ratings for features, not personas)
            for i, row in profiles_df.iterrows():
                progress_bar.progress((i + 1 + num_respondents) / (num_respondents * 2))
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        response = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {"role": "system", "content": """
                                    You are tasked with evaluating product features using a Kano model. For each feature, 
                                    please rate it on a scale from 1 to 5, based on the following meanings:
                                    - 1: "I like it"
                                    - 2: "I expect it"
                                    - 3: "I am indifferent"
                                    - 4: "I can live with it"
                                    - 5: "I dislike it"
                                    For each feature, you need to provide two ratings: one for the functional condition (feature present)
                                    and one for the dysfunctional condition (feature absent).
                                    Please return the ratings in the following format:
                                    {"feature_name": {"functional": {"rating": X}, "dysfunctional": {"rating": X}}}
                                """},
                                {"role": "user", "content": f"Please evaluate these features: {features}"}
                            ],
                            temperature=0
                        )
                        kano_responses.append(response.choices[0].message.content)
                        time.sleep(5)
                        break
                    except Exception:
                        retries += 1
                        time.sleep(RETRY_DELAY)

            progress_bar.progress(1.0)  
            st.session_state.results = {"profiles": profiles_df, "responses": kano_responses, "features": features}
            st.session_state.experiment_complete = True
            st.success("✅ Survey completed! View results in 'Results'.")

# --- Helper function to clean and parse JSON ---
def clean_and_parse_json(raw_response):
    """Cleans up raw response and attempts to parse it as valid JSON."""
    # Log the raw response for debugging
    st.write(f"Raw response: {raw_response}")  # Log the raw response for debugging
    
    # Check if the response is empty or None
    if not raw_response.strip():
        st.warning("⚠️ Empty response detected. Skipping this entry.")
        return None
    
    # Extract JSON using a regular expression that looks for curly braces ({}).
    # This will find the first JSON-like block in the response.
    json_pattern = r'(\{.*\})'
    matches = re.findall(json_pattern, raw_response, re.DOTALL)

    # If no matches, return warning and None
    if not matches:
        st.warning("❌ No valid JSON found in the response.")
        return None
    
    # Attempt to parse each JSON block and return a list of parsed JSON objects
    parsed_jsons = []
    for json_str in matches:
        try:
            parsed_json = json.loads(json_str)
            parsed_jsons.append(parsed_json)
        except json.JSONDecodeError as e:
            st.warning(f"❌ JSON parsing error: {e}")
            continue
    
    # If we have parsed JSON objects, return them, otherwise return None
    if parsed_jsons:
        return parsed_jsons
    else:
        st.warning("❌ No valid JSON could be parsed from the response.")
        return None

# --- TAB 2: Results ---
with tab2:
    if not st.session_state.experiment_complete:
        st.info("Run the survey first.")
    else:
        st.header("Results")
        
        st.write("### Respondent Profiles")
        profiles_df = st.session_state.results["profiles"].copy()
        profiles_df.index += 1  
        st.dataframe(profiles_df)

        st.write("### Kano Evaluations")
        kano_responses = st.session_state.results["responses"]
        
        # Check if kano_responses is None or empty
        if not kano_responses:
            st.warning("❌ No Kano responses found. Please ensure the survey ran successfully.")
        else:
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
            
            for i, resp in enumerate(kano_responses):
                try:
                    if not resp.strip():
                        st.warning(f"⚠️ Skipping empty response at index {i+1}.")
                        continue  

                    # Debugging: Output raw response to identify the problem
                    st.write(f"Raw response at index {i+1}: {resp}")

                    # Clean and parse JSON
                    parsed_json = clean_and_parse_json(resp)

                    if parsed_json:
                        # Process the parsed JSON
                        for feature, data in parsed_json.items():
                            f_score = rating_map.get(str(data['functional']['rating']), None)
                            d_score = rating_map.get(str(data['dysfunctional']['rating']), None)

                            if f_score is None or d_score is None:
                                st.warning(f"⚠️ Invalid scores at index {i+1} for feature {feature}. Skipping.")
                                continue

                            category = classify_kano(f_score, d_score)
                            classifications.append({
                                "Feature": feature,
                                "Kano Classification": category
                            })
                except json.JSONDecodeError as e:
                    st.warning(f"❌ JSON parsing error at index {i+1}: {e}")
                    continue

            if classifications:
                kano_df = pd.DataFrame(classifications)
                st.dataframe(kano_df)
            else:
                st.warning("🚨 No valid Kano classifications found.")
