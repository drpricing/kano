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
st.title('🤖 Product Feature Optimization')

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
        st.session_state.num_respondents = 8
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
            # Removed st.experimental_rerun() to avoid potential errors
            st.info("Setup complete! Now, please navigate to the 'Run Experiment' tab.")

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
        #experiences = range(1, 31)
        genders = ["male", "female", "not specified"]
        #customer_types = ["price sensitive", "health concious", "convenience seeker","smart shopper"]
        
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
            #experience = profiles_df['Experience'].iloc[i]
            gender = profiles_df['Gender'].iloc[i]
            customer_type = profiles_df['Customer Type'].iloc[i]
            input_text = f"Age: {age}; Gender: {gender}; Customer Type: {customer_type}"
            
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
        feature_list = [f.strip() for f in st.session_state.features.splitlines() if f.strip() != ""]
        
        kano_instructions = f"""
You are a synthetic respondent. Based on the following persona, please evaluate each of the following features in two scenarios:
1. When the feature is present
2. When the feature is missing

Use the following scale for each rating:
1 = I like it
2 = I expect it
3 = I am indifferent
4 = I can live with it
5 = I dislike it

Return your answers as a valid JSON object with the following structure:
{{
  "Feature1": {{"present": <rating>, "absent": <rating>}},
  "Feature2": {{"present": <rating>, "absent": <rating>}},
  ...
}}

Persona Description:
{{persona}}

Features to Evaluate:
{feature_list}
"""
        
        responses_list = []
        for i in range(profiles_df.shape[0]):
            progress_bar.progress((i + 1 + profiles_df.shape[0]) / (profiles_df.shape[0] * 3))
            persona_text = profiles_df['Persona'].iloc[i]
            prompt = kano_instructions.replace("{persona}", persona_text)
            
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are to respond only with a JSON object as described."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            responses_list.append(response.choices[0].message.content)
            time.sleep(random.uniform(5, 10))
        
        # Store all responses in session state for later processing
        st.session_state.results = {
            "personas": profiles_df,
            "responses": responses_list,
            "features": feature_list
        }
        st.session_state.experiment_complete = True
        st.success("✅ Experiment completed successfully!")
        st.info("Please navigate to the 'Results' tab to view and analyze your data.")

# -------------------- Results Tab --------------------
with tab3:
    if not st.session_state.experiment_complete:
        st.info("Please run the experiment in the 'Run Experiment' tab first.")
    else:
        st.header("Results Analysis")
        
        # Define mapping for responses
        rating_mapping = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "I like it": 1,
            "I expect it": 2,
            "I am indifferent": 3,
            "I can live with it": 4,
            "I dislike it": 5
        }
        
        # Function to classify Kano based on numeric ratings
        def classify_kano_numeric(f, d):
            # f: rating when present; d: rating when absent
            # Lower rating indicates a more positive perception.
            if f == 1 and d >= 4:
                return "Excitement"
            elif f == 2 and d == 5:
                return "Must-Have"
            elif f == 3 and d == 3:
                return "Indifferent"
            elif (d - f) >= 2:
                return "Must-Have"
            elif f < d:
                return "Excitement"
            else:
                return "Expected"
        
        # Parse the JSON responses from all respondents
        all_classifications = []
        for resp_str in st.session_state.results["responses"]:
            try:
                # Clean and parse the JSON
                cleaned = resp_str.replace('```', '').strip()
                resp_json = json.loads(cleaned)
            except Exception as e:
                st.warning(f"Error parsing a response: {e}")
                continue
            
            # For each feature in the response, extract ratings and classify
            for feat in st.session_state.results["features"]:
                if feat in resp_json:
                    rating_obj = resp_json[feat]
                    f_rating = rating_obj.get("present")
                    d_rating = rating_obj.get("absent")
                    
                    # Convert ratings to numeric values
                    try:
                        f_num = rating_mapping.get(str(f_rating).strip(), None)
                        d_num = rating_mapping.get(str(d_rating).strip(), None)
                    except Exception as e:
                        st.warning(f"Error converting ratings for feature {feat}: {e}")
                        continue
                    if f_num is None or d_num is None:
                        continue
                    classification = classify_kano_numeric(f_num, d_num)
                    all_classifications.append({
                        "Feature": feat,
                        "Response Present": f_rating,
                        "Response Absent": d_rating,
                        "Classification": classification
                    })
        
        if not all_classifications:
            st.error("No valid responses found. Please check the experiment responses.")
            st.stop()
        
        # Create DataFrame from classifications
        kano_df = pd.DataFrame(all_classifications)
        st.write("## Detailed Kano Evaluations")
        st.dataframe(kano_df)
        
        # Aggregate classifications per feature
        summary = kano_df.groupby("Feature")['Classification'].value_counts(normalize=True).mul(100).rename("Percentage").reset_index()
        st.write("## Kano Classification Summary (%)")
        st.dataframe(summary)
        
        # Create bar charts for each feature
        for feat in st.session_state.results["features"]:
            feat_data = summary[summary['Feature'] == feat]
            if feat_data.empty:
                continue
            fig = px.bar(feat_data, x='Classification', y='Percentage', title=f'Feature: {feat}', text='Percentage')
            st.plotly_chart(fig, use_container_width=True)
        
        # Download option for detailed results
        st.download_button(
            label="📥 Download Detailed Kano Results",
            data=kano_df.to_csv(index=False),
            file_name=f"kano_model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.markdown("### Overall Classification Counts")
        overall_counts = kano_df['Classification'].value_counts()
        st.bar_chart(overall_counts)
