import streamlit as st
import pandas as pd
import numpy as np

def generate_synthetic_responses(features, num_respondents=100):
    kano_scale = ["I dislike it", "I tolerate it", "I am indifferent", "I expect it", "I like it"]
    responses = []
    
    for feature in features:
        for _ in range(num_respondents):
            response_present = np.random.choice(kano_scale)
            response_absent = np.random.choice(kano_scale)
            responses.append([feature, response_present, response_absent])
    
    return pd.DataFrame(responses, columns=["Feature", "Response (Present)", "Response (Absent)"])

def classify_kano(present, absent):
    kano_matrix = {
        ("I like it", "I dislike it"): "Excitement",
        ("I like it", "I tolerate it"): "Excitement",
        ("I like it", "I am indifferent"): "Excitement",
        ("I like it", "I expect it"): "Excitement",
        ("I like it", "I like it"): "Excitement",
        
        ("I expect it", "I dislike it"): "Must-Have",
        ("I expect it", "I tolerate it"): "Must-Have",
        ("I expect it", "I am indifferent"): "Expected",
        ("I expect it", "I expect it"): "Expected",
        ("I expect it", "I like it"): "Expected",
        
        ("I am indifferent", "I dislike it"): "Indifferent",
        ("I am indifferent", "I tolerate it"): "Indifferent",
        ("I am indifferent", "I am indifferent"): "Indifferent",
        ("I am indifferent", "I expect it"): "Expected",
        ("I am indifferent", "I like it"): "Excitement",
    }
    
    return kano_matrix.get((present, absent), "Indifferent")

st.title("Kano Model Evaluation")

feature_input = st.text_area("Enter feature names (one per line):")
features = feature_input.split("\n") if feature_input else []

if st.button("Generate Responses"):
    df = generate_synthetic_responses(features)
    df["Kano Classification"] = df.apply(lambda row: classify_kano(row["Response (Present)"], row["Response (Absent)"]), axis=1)
    
    st.write("## Synthetic Responses")
    st.dataframe(df)
    
    st.write("## Kano Model Classification Results")
    summary = df.groupby("Feature")["Kano Classification"].value_counts(normalize=True).unstack().fillna(0) * 100
    st.bar_chart(summary)
