import random
import pandas as pd
import streamlit as st
import json
import time
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
        1. In the **Setup** tab, enter product details and features.
        2. Synthetic respondents evaluate the features.
        3. In **Results**, view Kano classifications, statistics, and a stacked bar diagram.
        """
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool evaluates product features using the Kano Model.")

st.title("🤖 Kano Model Feature Evaluation")
tab1, tab2 = st.tabs(["Setup", "Results"])

# Helper function to extract and parse JSON from a text response
def clean_and_parse_json(raw_response):
    """Extracts and parses the first JSON object in the response."""
    if not raw_response.strip():
        st.warning("⚠️ Empty response detected. Skipping this entry.")
        return None
    json_start = raw_response.find("{")
    json_end = raw_response.rfind("}") + 1
    if json_start == -1 or json_end == -1:
        st.warning("❌ No valid JSON found in the response.")
        return None
    json_str = raw_response[json_start:json_end].strip()
    try:
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError as e:
        st.warning(f"❌ JSON parsing error: {e}")
        return None

# -----------------------------
# TAB 1: Setup
# -----------------------------
with tab1:
    st.header("Setup")
    
    if 'start_experiment' not in st.session_state:
        st.session_state.start_experiment = False
    if 'experiment_complete' not in st.session_state:
        st.session_state.experiment_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None

    product_name = st.text_input('Enter product name', key="product_name")
    target_customers = st.text_area('Describe your target customers', height=150, key="target_customers")
    features_input = st.text_area('List features (one per line)', height=150, key="features")
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
            profiles = []
            
            # Generate synthetic respondent personas with a refined prompt for pure JSON
            for i in range(num_respondents):
                progress_bar.progress((i + 1) / (num_respondents * 2))
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        persona_resp = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[
                                {
                                    "role": "system",
                                    "content": """
Generate a synthetic customer persona based on the provided target customer description.
The persona must be returned as a pure JSON object (with no additional commentary) with exactly the following keys:
"Age" (an integer), "Gender" (a string), "Description" (a short string).
"""
                                },
                                {"role": "user", "content": f"Target Customer Description: {target_customers}"}
                            ],
                            temperature=0.7
                        )
                        raw_response = persona_resp.choices[0].message.content
                        #st.write(f"Raw profile response {i+1}:", raw_response)  # Debug output
                        persona_data = clean_and_parse_json(raw_response)
                        if not persona_data:
                            raise ValueError("Could not extract JSON from the response.")
                        
                        profiles.append({
                            "Age": persona_data.get("Age", random.randint(18, 78)),
                            "Gender": persona_data.get("Gender", random.choice(["Male", "Female", "Unknown"])),
                            "Persona": persona_data.get("Description", "No additional details provided.")
                        })
                        time.sleep(2)
                        break  # Exit retry loop if successful
                    except Exception as e:
                        retries += 1
                        st.warning(f"Error generating profile {i+1}: {e}")
                        time.sleep(RETRY_DELAY)
            
            if not profiles:
                st.error("No respondent profiles were generated. Please try again.")
            else:
                profiles_df = pd.DataFrame(profiles)
                features = [f.strip() for f in features_input.splitlines() if f.strip()]
                kano_responses = []
                
                # Fetch Kano ratings for each synthetic respondent with a refined prompt for pure JSON output
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
Your preferences are influenced by your persona.
When there is a conflict between age/gender and the target customer description, the latter prevails.
For each feature provided, rate it under two conditions:
- Functional condition (feature present)
- Dysfunctional condition (feature absent)
Use a scale of 1 to 5 where:
  1: I like it,
  2: I expect it,
  3: I am indifferent,
  4: I can live with it,
  5: I dislike it.
Return ONLY a pure JSON object in the following format:
{"feature_name": {"functional": {"rating": X}, "dysfunctional": {"rating": X}}}
"""}, 
                                    {"role": "user", "content": f"Features: {features}"}
                                ],
                                temperature=1
                            )
                            kano_responses.append(rating_resp.choices[0].message.content)
                            time.sleep(2)
                            break
                        except Exception as e:
                            retries += 1
                            st.warning(f"Error generating Kano response for respondent {i+1}: {e}")
                            time.sleep(RETRY_DELAY)
                
                progress_bar.progress(1.0)
                st.session_state.results = {"profiles": profiles_df, "responses": kano_responses, "features": features}
                st.session_state.experiment_complete = True
                st.success("✅ Survey completed! View results in 'Results'.")

# -----------------------------
# TAB 2: Results
# -----------------------------
with tab2:
    if not st.session_state.experiment_complete:
        st.info("Run the survey first.")
    else:
        st.header("Results")
        show_persona = st.checkbox("Show Persona Details", value=False)
        
        if "profiles" not in st.session_state.results or st.session_state.results["profiles"].empty:
            st.warning("No respondent profiles to display.")
        else:
            profiles_df = st.session_state.results["profiles"].copy()
            profiles_df.index += 1  
            st.write("### Respondent Profiles")
            if show_persona:
                st.dataframe(profiles_df)
            else:
                if "Persona" in profiles_df.columns:
                    st.dataframe(profiles_df.drop(columns=["Persona"]))
                else:
                    st.dataframe(profiles_df)
        
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
                    """
                    **Scale Explanation:** Ratings are on a scale from 1 to 5, where 1 means 'I like it', 2 means 'I expect it', 3 means 'I am indifferent', 4 means 'I can live with it', and 5 means 'I dislike it'. 
                    
                    **Classification Rules:** 'Excitement' is assigned when the functional rating is 1 and the dysfunctional rating is 4 or higher; 
                    'Must-Have' is when the functional rating is 2 and the dysfunctional rating is 5; 'Indifferent' when both ratings are 3; 
                    and 'Expected' for any other combination. The Net Kano Score is computed as (Rating (Present) - Rating (Missing)).
                    """
                )
                
                # Create frequency dataframe for the diagram
                freq_df = kano_df.groupby(["Feature", "Kano Classification"]).size().reset_index(name="Count")
                # Generate a stacked column chart
                fig = px.bar(
                    freq_df,
                    x="Feature",
                    y="Count",
                    color="Kano Classification",
                    text="Count",
                    title="Kano Classification Counts per Feature",
                    barmode="stack"
                )
                fig.update_layout(
                    xaxis_title="Feature",
                    yaxis_title="Number of Responses",
                    legend_title="Kano Classification",
                    uniformtext_minsize=12,
                    uniformtext_mode='hide'
                )
                st.plotly_chart(fig)
                
                # Download button for CSV export
                csv_data = kano_df.to_csv(index=True).encode('utf-8')
                st.download_button(label="Download Kano Evaluation Data",
                                   data=csv_data,
                                   file_name=f"kano_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")
            else:
                st.warning("🚨 No valid Kano classifications found.")
