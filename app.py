import streamlit as st
import groq

def categorize_features_with_llama(features, api_key):
    client = groq.Client(api_key)
    response = client.llama.categorize(features)
    return response

def main():
    st.title("Kano Model Feature Optimization")
    st.write("Welcome to the Kano Model Feature Optimization App!")

    api_key = st.text_input("Enter your Groq API key:", type="password")
    product_name = st.text_input("Enter the product name:")
    development_goals = st.text_area("Specify your product development goals:")
    features = st.text_area("List the features to be tested:")

    if st.button("Submit"):
        st.write(f"Product: {product_name}")
        st.write(f"Development Goals: {development_goals}")
        st.write(f"Features: {features}")

        if api_key:
            categorized_features = categorize_features_with_llama(features, api_key)
            st.write("Categorized Features:")
            for feature, category in categorized_features.items():
                st.write(f"{feature}: {category}")
        else:
            st.write("Please enter a valid API key.")

if __name__ == "__main__":
    main()
