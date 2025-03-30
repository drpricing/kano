import random
import pandas as pd
import streamlit as st
import os
import json
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from groq import Groq
from scipy import stats

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="Kano Model Feature Evaluation", page_icon="🤖", layout="wide")

# -------------------- Sidebar Configuration --------------------
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

# -------------------- Streamlit Tabs --------------------
st.title('🤖 Kano Model Feature Evaluation')
tab1, tab2 = st.tabs(["Setup", "Results"])

# -------------------- Setup Tab --------------------
with tab1:
    st.header("Setup")
    if 'start_experiment' not in st.session_state:
        st.session_state.start_experiment = False
    if 'experiment_complete' not in st.session_state:
        st.session_state.experiment_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Product Information
    st.subheader("Product Name")
    product_name = st.text_input('Enter product name', key="product_name")
    st.subheader("Target Customers")
    target_customers = st.text_area('Describe your target customers', height=150, key="target_customers")

    # Features
    st.subheader("Features")
    features_input = st.text_area('List features (one per line)', height=150, key="features")

    # Number of Respondents
    st.subheader("Number of Synthetic Respondents")
    num_respondents = st.number_input('Number of respondents', min_value=1, max_value=100, value=8, key="num_respondents")

    # Start Experiment
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

            # Initialize Groq Client
            client = Groq(api_key=api_key)

            # Define respondent attributes
            ages = range(18, 78)
            genders = ["Male", "Female", "Non-binary"]

            # Generate respondent profiles
            profiles = []
            for _ in range(st.session_state.num_respondents):
                profiles.append({"Age": random.choice(ages), "Gender": random.choice(genders)})
            profiles_df = pd.DataFrame(profiles)

            # API Rate Limit Handling
            MAX_RETRIES = 3
            RETRY_DELAY = 10

            # Generate Personas
            personas = []
            for i, row in profiles_df.iterrows():
                progress_bar.progress((i + 1) / (len(profiles_df) * 3))
                input_text = f"Age: {row['Age']}; Gender: {row['Gender']}"
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        response = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[{"role": "system", "content": "Create a customer persona based on:"},
                                      {"role": "user", "content": input_text}],
                            temperature=0
                        )
                        personas.append(response.choices[0].message.content)
                        time.sleep(5)
                        break
                    except Exception:
                        retries += 1
                        st.warning(f"Rate limit hit. Retrying ({retries}/{MAX_RETRIES}) in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
            profiles_df["Persona"] = personas

            # Kano Model Evaluation
            features = [f.strip() for f in features_input.splitlines() if f.strip()]
            kano_responses = []
            for i, row in profiles_df.iterrows():
                progress_bar.progress((i + 1 + len(profiles_df)) / (len(profiles_df) * 3))
                prompt = f"""
                You are a synthetic respondent. Based on the persona below, evaluate each feature:
                - Rate the feature **when present** and **when absent** (1-5 scale). 
                - Respond **only in JSON format**.
                Persona: {row['Persona']}
                Features: {features}
                """
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        response = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[{"role": "system", "content": "Only return a JSON object."},
                                      {"role": "user", "content": prompt}],
                            temperature=0
                        )
                        kano_responses.append(response.choices[0].message.content)
                        time.sleep(5)
                        break
                    except Exception:
                        retries += 1
                        st.warning(f"Rate limit hit. Retrying ({retries}/{MAX_RETRIES}) in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
            st.session_state.results = {"profiles": profiles_df, "responses": kano_responses, "features": features}
            st.session_state.experiment_complete = True
            st.success("✅ Survey completed! View results in 'Results'.")

# -------------------- Results Tab --------------------
with tab2:
    if not st.session_state.experiment_complete:
        st.info("Run the survey first.")
    else:
        st.header("Results")

        # Show Respondent Profiles
        st.write("### Respondent Profiles")
        st.dataframe(st.session_state.results["profiles"])

        # Parse Kano Responses
        all_classifications = []
        for resp in st.session_state.results["responses"]:
            try:
                parsed_json = json.loads(resp)
                if "features" in parsed_json:
                    for feature in parsed_json["features"]:
                        all_classifications.append({
                            "Feature": feature["feature"],
                            "Present": feature["when_present"],
                            "Absent": feature["when_absent"]
                        })
                else:
                    st.warning(f"Invalid response format: {resp}")
            except Exception as e:
                st.warning(f"Error parsing response: {e}")

        if not all_classifications:
            st.error("No valid Kano classifications found.")
        else:
            kano_df = pd.DataFrame(all_classifications)
            kano_df.index += 1
            st.write("### Kano Evaluations")
            st.dataframe(kano_df)

            # Summary Statistics
            summary = kano_df.groupby("Feature").count()
            st.write("### Summary Statistics")
            st.dataframe(summary)

            # Visualize Results
            st.write("### Visualizations")
            fig = px.box(kano_df, x="Feature", y="Present", title="Feature Ratings When Present")
            st.plotly_chart(fig, use_container_width=True)
            fig = px.box(kano_df, x="Feature", y="Absent", title="Feature Ratings When Absent")
            st.plotly_chart(fig, use_container_width=True)

            # Histogram of Classification by Feature
            st.write("### Histogram of Classification by Feature")
            fig_hist = px.histogram(kano_df, x="Feature", title="Histogram of Classification by Feature")
            st.plotly_chart(fig_hist, use_container_width=True)

            # Download Button
            st.download_button("📥 Download Kano Results", data=kano_df.to_csv(index=False), file_name="kano_results.csv", mime="text/csv")
