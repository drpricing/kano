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
    st.title("⚙️ A Dr. Pricing")
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

            # Generate synthetic respondent profiles (with hidden personas)
            ages = range(18, 78)
            genders = ["Male", "Female", "Unknown"]
            profiles = [{"Age": random.choice(ages), "Gender": random.choice(genders)} for _ in range(num_respondents)]
            profiles_df = pd.DataFrame(profiles)

            MAX_RETRIES = 3
            RETRY_DELAY = 10
            personas = []

            # Generate hidden persona descriptions (backend only)
            for i, row in profiles_df.iterrows():
                progress_bar.progress((i + 1) / (num_respondents * 2))
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        persona_resp = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {"role": "system", "content": "Create a customer persona based on:"},
                                {"role": "user", "content": f"Age: {row['Age']}; Gender: {row['Gender']}"}
                            ],
                            temperature=0
                        )
                        personas.append(persona_resp.choices[0].message.content)
                        time.sleep(2)
                        break
                    except Exception:
                        retries += 1
                        time.sleep(RETRY_DELAY)
            profiles_df["Persona"] = personas  # Stored in backend; not shown to user

            # Prepare feature list
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
                                    You are tasked with evaluating product features using the Kano model.
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
                            temperature=0
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
        
        # Display respondent profiles (hide persona details)
        st.write("### Respondent Profiles")
        profiles_df = st.session_state.results["profiles"].copy()
        profiles_df.index += 1  
        st.dataframe(profiles_df.drop(columns=["Persona"]))
        
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
                        except (ValueError, KeyError):
                            st.warning(f"⚠️ Unable to parse ratings for feature {feature} at index {i+1}. Skipping.")
                            continue
                        net_score = f_score - d_score
                        category = classify_kano(f_score, d_score)
                        classifications.append({
                            "Feature": feature,
                            "Rating (Present)": f_score,
                            "Rating (Missing)": d_score,
                            "Net Kano Score": net_score,
                            "Kano Classification": category
                        })
                    else:
                        st.warning(f"⚠️ Missing rating details for feature {feature} at index {i+1}. Skipping.")

            if classifications:
                kano_df = pd.DataFrame(classifications)
                kano_df.index = range(1, len(kano_df)+1)  # Numbering starts at 1
                st.write("#### Kano Classification Table")
                st.dataframe(kano_df)
                
                st.markdown(
                    "**Scale Explanation:** Ratings are on a scale from 1 to 5 where 1 means 'I like it', 2 means 'I expect it', "
                    "3 means 'I am indifferent', 4 means 'I can live with it', and 5 means 'I dislike it'. "
                    
                    "**Classification Rules:** 'Excitement' is assigned when the functional rating is 1 and the dysfunctional rating is 4 or higher; "
                    "'Must-Have' is when the functional rating is 2 and the dysfunctional rating is 5; 'Indifferent' when both ratings are 3; "
                    "and 'Expected' for any other combination. The Net Kano Score is computed as (Rating (Present) - Rating (Missing))."
                )
                
                # Generate stacked column chart: x-axis = features, y-axis = count, legend = Kano Classification
                freq_df = kano_df.groupby(["Feature", "Kano Classification"]).size().reset_index(name="Count")
                fig = px.bar(freq_df, x="Feature", y="Count", color="Kano Classification",
                             barmode="stack", title="Stacked Kano Classification Counts per Feature")
                st.plotly_chart(fig)
                
                # Download button for CSV export
                csv_data = kano_df.to_csv(index=True).encode('utf-8')
                st.download_button(label="Download Kano Evaluation Data",
                                   data=csv_data,
                                   file_name=f"kano_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")
            else:
                st.warning("🚨 No valid Kano classifications found.")
