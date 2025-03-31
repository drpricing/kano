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
        1. In the **Setup** tab, you enter your product details and feature list.
        2. Synthetic respondents (with hidden persona details) evaluate the features.
        3. In **Results**, you see Kano evaluation statistics, classifications in tables and diagrams.
        """
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool uses a Kano Model approach to evaluate product features.")

st.title('🤖 Kano Model Feature Evaluation')
tab1, tab2 = st.tabs(["Setup", "Results"])

# -----------------------------
# TAB 1: Setup
# -----------------------------
with tab1:
    st.header("Setup")
    
    # Initialize session state if not present
    if 'start_experiment' not in st.session_state:
        st.session_state.start_experiment = False
    if 'experiment_complete' not in st.session_state:
        st.session_state.experiment_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Input fields
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

            MAX_RETRIES = 3
            RETRY_DELAY = 10    
            
            def generate_synthetic_profiles(target_customers, num_respondents, progress_bar):
                """Generates synthetic customer profiles with error handling for JSON parsing."""
                profiles = []

                for i in range(num_respondents):
                    retries = 0
                    while retries < MAX_RETRIES:
                        try:
                            response = client.chat.completions.create(
                                model="llama3-70b-8192",
                                messages=[
                                    {"role": "system", "content": "Generate a synthetic user profile (age, gender, persona) based on this target customer description."},
                                    {"role": "user", "content": f"Target customers: {target_customers}"}
                                ],
                                temperature=0.1
                            )

                            profile = response.choices[0].message.content
                            profiles.append(profile)
                            time.sleep(2)  # Rate limiting
                            break
                        except Exception:
                            retries += 1
                            time.sleep(RETRY_DELAY)
                
                    progress_bar.progress((i + 1) / num_respondents)

                # Debugging: Print responses to check their structure
                for i, profile in enumerate(profiles):
                    try:
                        json.loads(profile)  # Test if JSON is valid
                    except json.JSONDecodeError:
                        st.error(f"🚨 JSONDecodeError in profile {i+1}: {profile}")
                        return None  # Stop processing if any profile is invalid

                # Convert to DataFrame only if all profiles are valid
                profiles_df = pd.DataFrame([json.loads(p) for p in profiles])
                return profiles_df

            profiles_df = generate_synthetic_profiles(target_customers, num_respondents, progress_bar)
            if profiles_df is None:
                st.warning("No valid profiles were generated. Please check the input and try again.")
            else:
                features = [f.strip() for f in features_input.splitlines() if f.strip()]
                kano_responses = []

                # Fetch Kano ratings for each synthetic respondent (ratings only)
                for i, row in profiles_df.iterrows():
                    progress_bar.progress((i + 1 + num_respondents) / (num_respondents * 2))
                    retries = 0
                    while retries < MAX_RETRIES:
                        try:
                            rating_resp = client.chat.completions.create(
                                model="llama3-70b-8192",
                                messages=[
                                    {"role": "system", "content": """
                                        You are tasked with evaluating product features using the Kano model. Your preferences are influenced by your persona.
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
# Helper function to parse JSON from ratings-only output
# -----------------------------
def clean_and_parse_json(raw_response):
    """Extracts and parses the first JSON object in the response (ratings only)."""
    if not raw_response.strip():
        st.warning("⚠️ Empty response detected. Skipping this entry.")
        return None
    json_start = raw_response.find("{")
    json_end = raw_response.rfind("}") + 1
    if json_start == -1 or json_end == -1:
        st.warning("❌ No valid JSON found in the response.")
        return None
    json_part = raw_response[json_start:json_end].strip()
    try:
        parsed = json.loads(json_part)
        return parsed
    except json.JSONDecodeError as e:
        st.warning(f"❌ JSON parsing error: {e}")
        return None

# -----------------------------
# TAB 2: Results
# -----------------------------
with tab2:
    if not st.session_state.experiment_complete:
        st.info("Run the survey first.")
    else:
        st.header("Results")
        
        # Toggle to show persona details
        show_persona = st.checkbox("Show Persona Details", value=False)
        
        st.write("### Respondent Profiles")
        
        # Ensure profiles_df exists and has the correct data
        if "profiles" in st.session_state.results and not st.session_state.results["profiles"].empty:
            profiles_df = st.session_state.results["profiles"].copy()
            profiles_df.index = profiles_df.index + 1  # Start index at 1 for better readability
        
            if not show_persona and "Persona" in profiles_df.columns:
                profiles_df = profiles_df.drop(columns=["Persona"])
        
            st.dataframe(profiles_df)
        else:
            st.warning("No respondent profiles available. Please generate profiles first.")
        
        # Process Kano ratings and classification
        kano_responses = st.session_state.results["responses"]
        features = st.session_state.results["features"]

        if not kano_responses:
            st.warning("❌ No Kano responses found. Please ensure the survey ran successfully.")
        else:
            rating_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

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
                parsed_json = clean_and_parse_json(resp)
                if parsed_json is None:
                    st.warning(f"⚠️ Invalid JSON response at index {i+1}. Skipping.")
                    continue
                for feature, data in parsed_json.items():
                    if "functional" in data and "dysfunctional" in data:
                        try:
                            f_score = int(data["functional"]["rating"])
                            d_score = int(data["dysfunctional"]["rating"])
                            feature_class = classify_kano(f_score, d_score)
                            classifications.append({
                                "Respondent": i+1,
                                "Feature": feature,
                                "Functional Rating": f_score,
                                "Dysfunctional Rating": d_score,
                                "Kano Classification": feature_class
                            })
                        except ValueError as e:
                            st.warning(f"⚠️ Invalid rating value in response: {e}. Skipping.")
        
            classification_df = pd.DataFrame(classifications)
            st.write("### Kano Classification Results")
            st.dataframe(classification_df)
            
            # Visualization: Kano classification pie chart
            if not classification_df.empty:
                classification_counts = classification_df['Kano Classification'].value_counts()
                fig = px.pie(classification_counts, values='value', names=classification_counts.index, title="Feature Classification Distribution")
                st.plotly_chart(fig)

                # Visualization: Average ratings per feature
                avg_ratings = classification_df.groupby('Feature').agg(
                    {'Functional Rating': 'mean', 'Dysfunctional Rating': 'mean'}).reset_index()

                st.write("### Average Ratings per Feature")
                st.dataframe(avg_ratings)

                # Visualization: Feature average rating comparison
                fig_avg_ratings = px.bar(
                    avg_ratings, x='Feature', y=['Functional Rating', 'Dysfunctional Rating'], barmode='group', title="Average Ratings per Feature"
                )
                st.plotly_chart(fig_avg_ratings)

        else:
            st.warning("❌ No Kano responses found. Please ensure the survey ran successfully.")
