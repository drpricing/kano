import streamlit as st
import groq

def categorize_features_with_llama(features):
    api_key = st.secrets["groq"]["api_key"]
    client = groq.Client(api_key=api_key)
    response = client.llama.categorize(features)
    return response

def main():
    st.title("Kano Model Feature Optimization")
    st.write("Welcome to the Kano Model Feature Optimization App!")

    product_name = st.text_input("Enter the product name:")
    development_goals = st.text_area("Specify your product development goals:")
    features = st.text_area("List the features to be tested:")

    if st.button("Submit"):
        st.write(f"Product: {product_name}")
        st.write(f"Development Goals: {development_goals}")
        st.write(f"Features: {features}")

        categorized_features = categorize_features_with_llama(features)
        st.write("Categorized Features:")
        for feature, category in categorized_features.items():
            st.write(f"{feature}: {category}")

if __name__ == "__main__":
    main()
