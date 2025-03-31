import streamlit as st
import pandas as pd
import openai
import json

# Load API key from Streamlit secrets
openai.api_key = st.secrets["GROQ_API_KEY"]

def generate_profiles(n, target_customers):
    """Generate synthetic respondent profiles using Llama3 via Groq API."""
    prompt = f"""
    Generate {n} synthetic customer profiles based on the following target customer description:
    "{target_customers}"
    Each profile should include:
    - Age (as an integer value)
    - Gender (Male, Female, or Non-Binary)
    - Persona (short description, e.g., 'Tech-savvy early adopter', 'Price-sensitive family buyer')
    Return the profiles in JSON format as a list of dictionaries.
    """
    
    response = openai.ChatCompletion.create(
        model="llama3-70b-8192",
        messages=[{"role": "system", "content": "You are an expert market analyst generating synthetic respondent profiles."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    try:
        profiles_json = response["choices"][0]["message"]["content"]
        profiles_list = json.loads(profiles_json)
        return pd.DataFrame(profiles_list)
    except (KeyError, json.JSONDecodeError):
        st.error("Failed to generate valid profiles. Please refine input or try again.")
        return pd.DataFrame(columns=["Age", "Gender", "Persona"])

# Streamlit UI
st.title("Synthetic Survey Web App")
n = st.number_input("Number of respondents", min_value=1, max_value=500, value=10)
target_customers = st.text_area("Describe your target customers")

if st.button("Generate Profiles"):
    profiles_df = generate_profiles(n, target_customers)
    st.dataframe(profiles_df)
