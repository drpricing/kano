import random
import pandas as pd
import streamlit as st
import os
import json
import time
import re
from groq import Groq
from scipy import stats
import numpy as np
import plotly.express as px
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Product Feature Optimization",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .stTextArea>div>div>textarea {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for API key and settings
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.secrets["groq"]["api_key"]
    
    st.markdown("---")
    st.markdown("### How does it work?")
    st.markdown("""
    1️⃣ Setup the experiment in the **Setup** tab  
    2️⃣ Generate synthetic responses in **Survey Synthetic Respondents** tab  
    3️⃣ View and analyze results in **Results** tab  
    """)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool helps you evaluate features using synthetic respondents and the Kano Model.")

# Main content
st.title("🤖 Product Feature Optimization")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Setup", "Survey Synthetic Respondents", "Results"])

# -------------------- Setup Tab --------------------
with tab1:
    st.header("Setup")
    
    # Initialize session state
    for key in ['product_name', 'target_customers', 'features', 'num_respondents', 'experiment_complete', 'results']:
        if key not in st.session_state:
            st.session_state[key] = "" if key in ['product_name', 'target_customers', 'features'] else False if key == 'experiment_complete' else 8
    
    # Input fields
    st.subheader("Product Name")
    st.session_state.product_name = st.text_input("Enter the product name", value=st.session_state.product_name)

    st.subheader("Target Customers")
    st.session_state.target_customers = st.text_area("Describe your target customers", value=st.session_state.target_customers, height=150)

    st.subheader("Features")
    st.markdown("Enter one feature per line.")
    st.session_state.features = st.text_area("List the features to be tested", value=st.session_state.features, height=150)

    st.subheader("Sample Size")
    st.session_state.num_respondents = st.number_input("Number of respondents", min_value=1, max_value=100, value=st.session_state.num_respondents)

    # Start button
    if st.button("🚀 Start Experiment", type="primary"):
        if not api_key:
            st.error("Please provide your Groq API key in the sidebar.")
        elif not st.session_state.product_name or not st.session_state.target_customers or not st.session_state.features:
            st.error("Please fill in all required fields.")
        else:
            st.session_state.experiment_complete = False
            st.session_state.results = None
            st.info("Setup complete! Now, please navigate to 'Survey Synthetic Respondents'.")

# -------------------- Survey Synthetic Respondents Tab --------------------
with tab2:
    if not st.session_state.experiment_complete:
        st.header("Survey Synthetic Respondents")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Generate respondent profiles
        ages = range(18, 78)
        genders = ["Male", "Female", "Non-Binary"]
        
        profiles = [{"Age": random.choice(ages), "Gender": random.choice(genders)} for _ in range(st.session_state.num_respondents)]
        profiles_df = pd.DataFrame(profiles)

        # Generate detailed personas using Groq
        system_instructions = "Create a detailed persona for a target customer based on the given age and gender."
        personas = []
        for i, row in profiles_df.iterrows():
            progress_bar.progress((i + 1) / (st.session_state.num_respondents * 2))
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "system", "content": system_instructions},
                          {"role": "user", "content": f"Age: {row['Age']}, Gender: {row['Gender']}"}],
                temperature=0
            )
            personas.append(response.choices[0].message.content)
            time.sleep(2)

        profiles_df["Persona"] = personas

        # Define Kano survey
        feature_list = st.session_state.features.splitlines()
        responses = []
        for i, persona in enumerate(personas):
            progress_bar.progress((i + st.session_state.num_respondents) / (st.session_state.num_respondents * 2))
            prompt = f"Based on this persona: {persona}, evaluate each feature using Kano's model."
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "system", "content": "Provide only a valid JSON response."},
                          {"role": "user", "content": prompt}],
                temperature=0
            )
            responses.append(response.choices[0].message.content)
            time.sleep(2)

        # Store results
        st.session_state.results = {"personas": profiles_df, "responses": responses, "features": feature_list}
        st.session_state.experiment_complete = True
        st.success("✅ Survey completed! Go to 'Results'.")

# -------------------- Results Tab --------------------
with tab3:
    if not st.session_state.experiment_complete:
        st.info("Please complete the survey first.")
    else:
        st.header("Results Analysis")
        
        st.subheader("📊 Respondent Profiles")
        st.dataframe(st.session_state.results["personas"])
        
        st.subheader("🔍 Raw Kano Responses")
        st.write(st.session_state.results["responses"])

        rating_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
        classifications = []

        for response in st.session_state.results["responses"]:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                st.warning("Response does not contain valid JSON.")
                continue

            try:
                kano_data = json.loads(json_match.group(0))
                for feature, ratings in kano_data.items():
                    f_num = rating_map.get(str(ratings["present"]), None)
                    d_num = rating_map.get(str(ratings["absent"]), None)
                    if None in [f_num, d_num]:
                        continue
                    classifications.append({"Feature": feature, "Classification": "Excitement" if f_num == 1 and d_num >= 4 else "Expected"})
            except Exception as e:
                st.warning(f"Error parsing response: {e}")
                continue

        if classifications:
            kano_df = pd.DataFrame(classifications)
            st.subheader("✅ Kano Classification Results")
            st.dataframe(kano_df)
        else:
            st.error("No valid Kano classifications found. Check the raw responses above.")

        st.download_button("📥 Download Kano Results", data=kano_df.to_csv(index=False), file_name="kano_results.csv", mime="text/csv")