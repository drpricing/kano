import random
import pandas as pd
import streamlit as st
import os
from groq import Groq
import json
import time
import plotly.express as px
from datetime import datetime

# Set page config
st.set_page_config(page_title="Product Feature Optimization", page_icon="🤖", layout="wide")

# Sidebar for API key and settings
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.secrets["groq"]["api_key"]

# Main content
st.title('🤖 Product Feature Optimization')

# Create tabs for different sections
tab1, tab2 = st.tabs(["Setup", "Survey Synthetic Respondents"])

# -------------------- Setup Tab --------------------
with tab1:
    st.header("Setup")
    
    # Initialize session state
    session_defaults = {
        'product_name': "", 'target_customers': "", 'features': "", 
        'num_respondents': 8, 'experiment_complete': False, 'results': None
    }
    for key, default in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Inputs
    st.session_state.product_name = st.text_input("Enter the product name", value=st.session_state.product_name)
    st.session_state.target_customers = st.text_area("Describe your target customers", value=st.session_state.target_customers, height=150)
    st.session_state.features = st.text_area("List features to be tested (one per line)", value=st.session_state.features, height=150)
    st.session_state.num_respondents = st.number_input("Number of respondents", min_value=1, max_value=100, value=st.session_state.num_respondents)

    # Start button
    if st.button('🚀 Start Survey', type="primary"):
        if not api_key:
            st.error("Please provide your Groq API key in the sidebar.")
        elif not st.session_state.product_name or not st.session_state.target_customers or not st.session_state.features.strip():
            st.error("Please fill in all required fields.")
        else:
            st.session_state.experiment_complete = False
            st.session_state.results = None

            st.info("Running survey... Please wait.")
            progress_bar = st.progress(0)

            client = Groq(api_key=api_key)

            ages = range(18, 78)
            genders = ["male", "female", "not specified"]

            # Generate synthetic personas (Age & Gender only)
            profiles = [{"Age": random.choice(ages), "Gender": random.choice(genders)}
                        for _ in range(st.session_state.num_respondents)]
            profiles_df = pd.DataFrame(profiles)

            # Generate personas with Groq
            personas = []
            system_instructions = f"""
            Generate a detailed persona based on age, gender, and the following target customer description:
            {st.session_state.target_customers}

            Include assumed background, interests, and purchase behaviors.
            """

            for i, row in profiles_df.iterrows():
                progress_bar.progress((i + 1) / (profiles_df.shape[0] * 2))
                input_text = f"Age: {row['Age']}; Gender: {row['Gender']}"

                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "system", "content": system_instructions}, {"role": "user", "content": input_text}],
                    temperature=0
                )
                personas.append(response.choices[0].message.content)
                time.sleep(random.uniform(5, 10))

            profiles_df['Persona'] = personas

            # Extract features correctly
            feature_list = [f.strip() for f in st.session_state.features.splitlines() if f.strip()]
            if not feature_list:
                st.error("No valid features found. Please check your input.")
                st.stop()

            # Kano evaluation prompt
            kano_instructions = """
            Evaluate each feature in two scenarios:
            1. When the feature is present
            2. When the feature is missing

            Use this scale:
            1 = I like it  
            2 = I expect it  
            3 = I am indifferent  
            4 = I can live with it  
            5 = I dislike it  

            Return answers as JSON:
            {
              "Feature1": {"present": <rating>, "absent": <rating>},
              "Feature2": {"present": <rating>, "absent": <rating>}
            }

            Persona:
            {persona}

            Features:
            {features}
            """

            responses_list = []
            for i, persona_text in enumerate(profiles_df['Persona']):
                progress_bar.progress((i + 1 + profiles_df.shape[0]) / (profiles_df.shape[0] * 2))
                prompt = kano_instructions.replace("{persona}", persona_text).replace("{features}", str(feature_list))

                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                responses_list.append(response.choices[0].message.content)
                time.sleep(random.uniform(5, 10))

            st.session_state.results = {"personas": profiles_df, "responses": responses_list, "features": feature_list}
            st.session_state.experiment_complete = True
            st.success("✅ Survey completed! Go to the 'Survey Synthetic Respondents' tab.")

# -------------------- Results Tab --------------------
with tab2:
    if not st.session_state.experiment_complete:
        st.info("Run the survey first in the 'Setup' tab.")
    else:
        st.header("Results Analysis")

        # Display respondent profiles
        st.subheader("🧑‍🤝‍🧑 Respondent Profiles")
        st.dataframe(st.session_state.results["personas"][["Age", "Gender", "Persona"]])

        rating_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, 
                      "I like it": 1, "I expect it": 2, "I am indifferent": 3, 
                      "I can live with it": 4, "I dislike it": 5}
        
        def classify_kano(f, d):
            return "Excitement" if f == 1 and d >= 4 else "Must-Have" if f == 2 and d == 5 else "Indifferent" if f == 3 and d == 3 else "Expected"

        kano_data = []
        for response in st.session_state.results["responses"]:
            try:
                data = json.loads(response.replace("```", "").strip())
                for feature in st.session_state.results["features"]:
                    if feature in data:
                        f, d = rating_map.get(str(data[feature]["present"]).strip()), rating_map.get(str(data[feature]["absent"]).strip())
                        if f and d:
                            kano_data.append({"Feature": feature, "Classification": classify_kano(f, d)})
            except Exception as e:
                st.warning(f"Error parsing response: {e}")

        if not kano_data:
            st.error("No valid classifications found.")
            st.stop()

        kano_df = pd.DataFrame(kano_data)
        st.write("## Kano Evaluation Results")
        st.dataframe(kano_df)

        st.download_button("📥 Download Results", kano_df.to_csv(index=False), "kano_results.csv", "text/csv")