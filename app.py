import random
import pandas as pd
import streamlit as st
import os
from groq import Groq
import json
import time
from scipy import stats
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Kano Model Feature Optimization",
    page_icon="🔍",
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
    Step 1: Add the necessary information for the experiment in the Setup tab  
    Step 2: Run the experiment in the Run Experiment tab  
    Step 3: View the Kano evaluation results in the Results tab  
    """)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool helps you conduct synthetic experiments using a Kano Model approach to evaluate features.
    """)

# Main content
st.title('🔍 Kano Model Feature Optimization')

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Setup", "Run Experiment", "Results"])

# -------------------- Setup Tab --------------------
with tab1:
    st.header("Experiment Setup")
    
    # Initialize session state for input values if they don't exist
    if 'product_name' not in st.session_state:
        st.session_state.product_name = ""
    if 'target_customers' not in st.session_state:
        st.session_state.target_customers = ""
    if 'features' not in st.session_state:
        st.session_state.features = ""
    if 'num_respondents' not in st.session_state:
        st.session_state.num_respondents = 10
    if 'start_experiment' not in st.session_state:
        st.session_state.start_experiment = False
    if 'experiment_complete' not in st.session_state:
        st.session_state.experiment_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Product Name
    st.subheader("Product Name")
    product_name = st.text_input(
        'Enter the product name',
        value=st.session_state.product_name,
        key="product_name_input"
    )
    st.session_state.product_name = product_name
    
    # Target Customers
    st.subheader("Target Customers")
    target_customers = st.text_area(
        'Describe your target customers',
        value=st.session_state.target_customers,
        height=150,
        key="target_customers_input"
    )
    st.session_state.target_customers = target_customers
    
    # Features
    st.subheader("Features")
    st.markdown("Enter one feature per line.")
    features_input = st.text_area(
        'List the features to be tested (one per line)',
        value=st.session_state.features,
        height=150,
        key="features_input"
    )
    st.session_state.features = features_input
    
    # Number of respondents
    st.subheader("Sample Size")
    num_respondents = st.number_input(
        'Number of respondents',
        min_value=1,
        max_value=100,
        value=st.session_state.num_respondents,
        key="num_respondents_input"
    )
    st.session_state.num_respondents = num_respondents
    
    # Start button
    if st.button('🚀 Start Experiment', type="primary"):
        if not api_key:
            st.error("Please provide your Groq API key in the sidebar.")
        elif not product_name or not target_customers or not features_input:
            st.error("Please fill in all required fields.")
        else:
            st.session_state.start_experiment = True
            st.session_state.experiment_complete = False
            st.session_state.results = None
            st.experimental_rerun()

# -------------------- Run Experiment Tab --------------------
with tab2:
    if not st.session_state.start_experiment:
        st.info("Please complete the setup in the 'Setup' tab first.")
    elif st.session_state.experiment_complete:
        st.success("✅ Experiment completed successfully!")
        st.info("Please navigate to the 'Results' tab to view and analyze your data.")
    else:
        st.header("Running Experiment")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Define ranges for synthetic persona attributes
        ages = range(18, 66)
        experiences = range(1, 31)
        genders = ["male", "female", "not specified"]
        customer_types = ["early adopter", "mainstream", "laggard"]
        
        # Generate personas
        profiles = []
        for _ in range(st.session_state.num_respondents):
            profile = {
                "Age": random.choice(ages),
                "Experience": random.choice(experiences),
                "Gender": random.choice(genders),
                "Customer Type": random.choice(customer_types),
            }
            profiles.append(profile)
        profiles_df = pd.DataFrame(profiles)
        
        # Generate detailed personas using Groq
        system_instructions = """
        Create a detailed persona for a target customer based on the provided attributes.
        Include background, preferences, usage patterns, and motivations.
        """
        personas = []
        for i in range(profiles_df.shape[0]):
            progress_bar.progress((i + 1) / (profiles_df.shape[0] * 3))
            age = profiles_df['Age'].iloc[i]
            experience = profiles_df['Experience'].iloc[i]
            gender = profiles_df['Gender'].iloc[i]
            customer_type = profiles_df['Customer Type'].iloc[i]
            input_text = f"Age: {age}; Experience: {experience}; Gender: {gender}; Customer Type: {customer_type}"
            
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": input_text}
                ],
                temperature=0
            )
            personas.append(response.choices[0].message.content)
            time.sleep(random.uniform(5, 10))
        profiles_df['Persona'] = personas
        
        # Prepare the Kano evaluation prompt for each synthetic respondent
        # The prompt asks for ratings for each feature in two scenarios:
        # when the feature is present and when it is missing.
        # Use the scale: 1 = I like it, 2 = I expect it, 3 = I am indifferent, 4 = I can live with it, 5 = I dislike it.
        feature_list = [f.strip() for f in st.session_state.features.splitlines() if f.strip() != \"\"]\n        \n        kano_instructions = f\"\"\"\nYou are a synthetic respondent. Based on the following persona, please evaluate each of the following features in two scenarios:\n1. When the feature is present\n2. When the feature is missing\n\nUse the following scale for each rating:\n1 = I like it\n2 = I expect it\n3 = I am indifferent\n4 = I can live with it\n5 = I dislike it\n\nReturn your answers as a valid JSON object with the following structure:\n{{\n  \"Feature1\": {{\"present\": <rating>, \"absent\": <rating>}},\n  \"Feature2\": {{\"present\": <rating>, \"absent\": <rating>}},\n  ...\n}}\n\nPersona Description:\n{{persona}}\n\nFeatures to Evaluate:\n{feature_list}\n\"\"\"\n\n        responses_list = []\n        for i in range(profiles_df.shape[0]):\n            progress_bar.progress((i + 1 + profiles_df.shape[0]) / (profiles_df.shape[0] * 3))\n            persona_text = profiles_df['Persona'].iloc[i]\n            prompt = kano_instructions.replace(\"{persona}\", persona_text)\n            \n            response = client.chat.completions.create(\n                model=\"llama3-70b-8192\",\n                messages=[\n                    {\"role\": \"system\", \"content\": \"You are to respond only with a JSON object as described.\"},\n                    {\"role\": \"user\", \"content\": prompt}\n                ],\n                temperature=0\n            )\n            responses_list.append(response.choices[0].message.content)\n            time.sleep(random.uniform(5, 10))\n        \n        # Store all responses in session state for later processing\n        # We expect responses_list to be a list of JSON strings\n        st.session_state.results = {\n            \"personas\": profiles_df,\n            \"responses\": responses_list,\n            \"features\": feature_list\n        }\n        st.session_state.experiment_complete = True\n        st.success(\"✅ Experiment completed successfully!\")\n        st.info(\"Please navigate to the 'Results' tab to view and analyze your data.\")\n\n# -------------------- Results Tab --------------------\nwith tab3:\n    if not st.session_state.experiment_complete:\n        st.info(\"Please run the experiment in the 'Run Experiment' tab first.\")\n    else:\n        st.header(\"Results Analysis\")\n        \n        # Define mapping for responses\n        rating_mapping = {\n            \"1\": 1,\n            \"2\": 2,\n            \"3\": 3,\n            \"4\": 4,\n            \"5\": 5,\n            \"I like it\": 1,\n            \"I expect it\": 2,\n            \"I am indifferent\": 3,\n            \"I can live with it\": 4,\n            \"I dislike it\": 5\n        }\n        \n        # Function to classify Kano based on numeric ratings\n        def classify_kano_numeric(f, d):\n            # f: rating when present; d: rating when absent\n            # Lower rating indicates a more positive perception.\n            if f == 1 and d >= 4:\n                return \"Excitement\"\n            elif f == 2 and d == 5:\n                return \"Must-Have\"\n            elif f == 3 and d == 3:\n                return \"Indifferent\"\n            elif (d - f) >= 2:\n                return \"Must-Have\"\n            elif f < d:\n                return \"Excitement\"\n            else:\n                return \"Expected\"\n        \n        # Parse the JSON responses from all respondents\n        all_classifications = []\n        for resp_str in st.session_state.results[\"responses\"]:\n            try:\n                # Clean and parse the JSON\n                cleaned = resp_str.replace('```', '').strip()\n                resp_json = json.loads(cleaned)\n            except Exception as e:\n                st.warning(f\"Error parsing a response: {e}\")\n                continue\n            \n            # For each feature in the response, extract ratings and classify\n            for feat in st.session_state.results[\"features\"]:\n                if feat in resp_json:\n                    rating_obj = resp_json[feat]\n                    f_rating = rating_obj.get(\"present\")\n                    d_rating = rating_obj.get(\"absent\")\n                    \n                    # Convert ratings to numeric values\n                    try:\n                        f_num = rating_mapping.get(str(f_rating).strip(), None)\n                        d_num = rating_mapping.get(str(d_rating).strip(), None)\n                    except Exception as e:\n                        st.warning(f\"Error converting ratings for feature {feat}: {e}\")\n                        continue\n                    if f_num is None or d_num is None:\n                        continue\n                    classification = classify_kano_numeric(f_num, d_num)\n                    all_classifications.append({\n                        \"Feature\": feat,\n                        \"Response Present\": f_rating,\n                        \"Response Absent\": d_rating,\n                        \"Classification\": classification\n                    })\n        \n        if not all_classifications:\n            st.error(\"No valid responses found. Please check the experiment responses.\")\n            st.stop()\n        \n        # Create DataFrame from classifications\n        kano_df = pd.DataFrame(all_classifications)\n        st.write(\"## Detailed Kano Evaluations\")\n        st.dataframe(kano_df)\n        \n        # Aggregate classifications per feature\n        summary = kano_df.groupby(\"Feature\")['Classification'].value_counts(normalize=True).mul(100).rename(\"Percentage\").reset_index()\n        st.write(\"## Kano Classification Summary (%)\")\n        st.dataframe(summary)\n        \n        # Create bar charts for each feature\n        for feat in st.session_state.results[\"features\"]:\n            feat_data = summary[summary['Feature'] == feat]\n            if feat_data.empty:\n                continue\n            fig = px.bar(feat_data, x='Classification', y='Percentage', title=f'Feature: {feat}', text='Percentage')\n            st.plotly_chart(fig, use_container_width=True)\n        \n        # Additionally, show download option for detailed results\n        st.download_button(\n            label=\"📥 Download Detailed Kano Results\",\n            data=kano_df.to_csv(index=False),\n            file_name=f\"kano_model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv\",\n            mime=\"text/csv\"\n        )\n\n        # You can also perform further statistical analysis if needed\n        st.markdown(\"### Overall Classification Counts\")\n        overall_counts = kano_df['Classification'].value_counts()\n        st.bar_chart(overall_counts)"
