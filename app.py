import random
import pandas as pd
import streamlit as st
import json
import time
import plotly.express as px
from groq import Groq

st.set_page_config(page_title="Kano Model Feature Evaluation", page_icon="🤖", layout="wide")

# Sidebar: API Key Setup
with st.sidebar:
    st.title("⚙️ Instructions")
    api_key = st.secrets["groq"]["api_key"]
    st.markdown("This tool evaluates features using a Kano Model approach.")

st.title('🤖 Kano Model Feature Evaluation')
tab1, tab2 = st.tabs(["Setup", "Results"])

### TAB 1: SURVEY SETUP ###
with tab1:
    st.header("Setup")
    st.session_state.setdefault("start_experiment", False)
    st.session_state.setdefault("experiment_complete", False)
    st.session_state.setdefault("results", None)

    product_name = st.text_input('Product Name', key="product_name")
    target_customers = st.text_area('Target Customers', height=150, key="target_customers")
    features_input = st.text_area('List features (one per line)', height=150, key="features")
    num_respondents = st.number_input('Number of respondents', min_value=1, max_value=100, value=8, key="num_respondents")

    if st.button('🚀 Start Survey', type="primary"):
        if not api_key:
            st.error("❌ API key missing. Please provide it in the sidebar.")
        elif not product_name or not target_customers or not features_input:
            st.error("❌ All fields are required.")
        else:
            st.session_state.start_experiment = True
            st.session_state.experiment_complete = False
            st.session_state.results = None

            st.header("Survey Synthetic Respondents")
            progress_bar = st.progress(0)
            client = Groq(api_key=api_key)

            # Generate personas
            profiles = [{"Age": random.randint(18, 78), "Gender": random.choice(["Male", "Female", "Unknown"])} for _ in range(num_respondents)]
            profiles_df = pd.DataFrame(profiles)

            MAX_RETRIES = 3
            RETRY_DELAY = 10
            personas = []

            # Generate Personas via API
            for i, row in profiles_df.iterrows():
                progress_bar.progress(i / (2 * num_respondents))
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        response = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {"role": "system", "content": "Create a customer persona with structured JSON."},
                                {"role": "user", "content": f"Age: {row['Age']}; Gender: {row['Gender']} Return JSON format: {{'persona': 'Persona description here'}}"}
                            ],
                            temperature=0
                        )
                        response_text = response.choices[0].message.content if response.choices else ""
                        
                        if response_text.strip():
                            personas.append(response_text)
                        else:
                            st.warning(f"⚠️ Empty persona response for respondent {i+1}, skipping.")

                        time.sleep(5)
                        break
                    except Exception:
                        retries += 1
                        time.sleep(RETRY_DELAY)
            
            profiles_df["Persona"] = personas
            features = [f.strip() for f in features_input.splitlines() if f.strip()]
            kano_responses = []

            # Generate Kano Responses via API
            for i, row in profiles_df.iterrows():
                progress_bar.progress((i + num_respondents) / (2 * num_respondents))
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        response = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {"role": "system", "content": "You are a Kano Model survey assistant. Return structured JSON."},
                                {"role": "user", "content": f"""
Given this customer profile: {row['Persona']}
Evaluate these features using the Kano model:

{features}

Return valid JSON format:
{{
    "features": [
        {{
            "name": "Feature Name",
            "when_present": "Delighter | Must-Have | Performance | Indifferent | Reverse",
            "importance": 1-5
        }}
    ]
}}
                                """}
                            ],
                            temperature=0
                        )
                        response_text = response.choices[0].message.content if response.choices else ""

                        # Ensure non-empty response
                        if response_text.strip():
                            kano_responses.append(response_text)
                        else:
                            st.warning(f"⚠️ Empty Kano response for respondent {i+1}, skipping.")

                        time.sleep(5)
                        break
                    except Exception:
                        retries += 1
                        time.sleep(RETRY_DELAY)
            
            progress_bar.progress(1.0)
            st.session_state.results = {"profiles": profiles_df, "responses": kano_responses, "features": features}
            st.session_state.experiment_complete = True
            st.success("✅ Survey completed! View results in 'Results'.")

### TAB 2: RESULTS ANALYSIS ###
with tab2:
    if not st.session_state.experiment_complete:
        st.info("🚨 Run the survey first.")
    else:
        st.header("Results")

        st.write("### Respondent Profiles")
        profiles_df = st.session_state.results["profiles"].copy()
        profiles_df.index += 1  
        st.dataframe(profiles_df)

        st.write("### Kano Evaluations")
        kano_responses = st.session_state.results["responses"]

        classifications = []
        for i, resp in enumerate(kano_responses):
            try:
                if not resp.strip():
                    st.warning(f"⚠️ Skipping empty response at index {i+1}.")
                    continue  

                parsed_json = json.loads(resp)  

                if "features" not in parsed_json or not isinstance(parsed_json["features"], list):
                    st.warning(f"⚠️ Unexpected response format at index {i+1}: {parsed_json}")
                    continue  

                for feat_obj in parsed_json["features"]:
                    if "name" in feat_obj and "when_present" in feat_obj and "importance" in feat_obj:
                        classifications.append({
                            "Feature": feat_obj["name"],
                            "Kano Classification": feat_obj["when_present"],
                            "Importance": feat_obj["importance"]
                        })
                    else:
                        st.warning(f"⚠️ Skipping malformed entry at index {i+1}: {feat_obj}")

            except json.JSONDecodeError as e:
                st.warning(f"❌ JSON parsing error at index {i+1}: {e}")

        if classifications:
            kano_df = pd.DataFrame(classifications)
            kano_df.index += 1
            st.dataframe(kano_df)
            
            fig = px.bar(kano_df, x="Feature", y="Importance", color="Kano Classification", title="Kano Model Feature Importance")
            st.plotly_chart(fig)

            csv = kano_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Kano Results", data=csv, file_name="kano_results.csv", mime="text/csv")
        else:
            st.warning("🚨 No valid Kano classifications found. Ensure survey responses are properly formatted.")
