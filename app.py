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
    st.image("investor.jpeg", width=150)
    st.title("⚙️ Settings")
    api_key = st.text_input(
        'Groq API Key',
        type="password",
        help="Get your API key from: https://console.groq.com/keys"
    )
    
    st.markdown("---")
    st.markdown("### How does it work?")
    st.markdown("""
    Step 1: Add your GROQ API Key (go to https://www.youtube.com/watch?v=_Deu9x5efvQ for instruction video)
    
    Step 2: Add the necessary information for the experiment in the Setup tab
    
    Step 3: Check progress of the experiment in the Run Experiment tab
    
    Step 4: View the results in the Results tab
    """)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool helps you conduct synthetic experiments with target customers as respondents.
    It uses AI to simulate realistic customer responses to different scenarios.
    """)

# Main content
st.title('🔍 Kano Model Feature Optimization')

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Setup", "Run Experiment", "Results"])

with tab1:
    st.header("Experiment Setup")
    
    # Initialize session state for input values if they don't exist
    if 'product_name' not in st.session_state:
        st.session_state.product_name = ""
    if 'development_goals' not in st.session_state:
        st.session_state.development_goals = ""
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
    
    # Development Goals
    st.subheader("Development Goals")
    development_goals = st.text_area(
        'Specify your product development goals',
        value=st.session_state.development_goals,
        height=150,
        key="development_goals_input"
    )
    st.session_state.development_goals = development_goals
    
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
    features = st.text_area(
        'List the features to be tested (comma-separated)',
        value=st.session_state.features,
        height=150,
        key="features_input"
    )
    st.session_state.features = features
    
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
        elif not product_name or not development_goals or not target_customers or not features:
            st.error("Please fill in all required fields.")
        else:
            st.session_state.start_experiment = True
            st.session_state.experiment_complete = False
            st.session_state.results = None
            st.rerun()

with tab2:
    if not st.session_state.start_experiment:
        st.info("Please complete the setup in the 'Setup' tab first.")
    elif st.session_state.experiment_complete:
        st.success("✅ Experiment completed successfully!")
        st.info("Please navigate to the 'Results' tab to view and analyze your data.")
    else:
        st.header("Running Experiment")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize Groq client
        os.environ["GROQ_API_KEY"] = api_key
        client = Groq(api_key=api_key)
        
        # Define the ranges and options
        ages = range(18, 66)
        experiences = range(1, 31)
        genders = ["male", "female", "not specified"]
        customer_types = ["early adopter", "mainstream", "laggard"]
        
        # Generate profiles
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
        
        # System instructions (keep existing instructions)
        system_instructions = """
        Create a detailed persona for a target customer based on the information provided by the user.
        Consider factors such as the customer's background, preferences, usage patterns, and key motivations. Use the provided data to craft a comprehensive and realistic profile that will aid in understanding the customer's perspective and strategic approach.
        # Steps
        1. Analyze the provided information about the target customer.
        2. Identify key attributes and characteristics relevant to the customer's persona, such as:
        - Background and demographics
        - Professional experience
        - Usage patterns and preferences
        - Risk tolerance and goals
        - Personal motivations and values
        3. Integrate the attributes into a coherent narrative that highlights the customer's priorities and potential decision-making processes.
        # Output Format
        The output should be a comprehensive paragraph or set of paragraphs detailing the customer persona. Each attribute should be clearly integrated into the narrative to create a vivid, coherent portrait of the customer.
        # Notes
        - Ensure that all persona narratives are coherent and relevant to the provided background information and context.
        - Incorporate any specific goals or additional attributes mentioned by the user to tailor the persona accurately.
        - Maintain a balanced approach, intertwining professional aspects with personal motivations whenever applicable.
        """
        
        # Generate personas
        status_text.text("Generating customer personas...")
        personalist = []
        for i in range(profiles_df.shape[0]):
            progress_bar.progress((i + 1) / (profiles_df.shape[0] * 3))
            age = profiles_df['Age'].iloc[i]
            experience = profiles_df['Experience'].iloc[i]
            gender = profiles_df['Gender'].iloc[i]
            customer_type = profiles_df['Customer Type'].iloc[i]
            input_text = f"Age: {age}; Experience: {experience}; Gender: {gender}; Customer Type: {customer_type}"
            
            response = client.chat.com
