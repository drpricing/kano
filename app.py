import streamlit as st
from groq import Groq

def categorize_features_with_llama(features):
    api_key = st.secrets["groq"]["api_key"]
    client = Groq(api_key=api_key)
    
    messages_payload = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": features}
    ]
    
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages_payload,
        temperature=0.7
    )
    
    categorized_features = response.choices[0].message.content
    return categorized_features

def kano_model_classification(features):
    # This function simulates the Kano Model classification
    # For simplicity, we use a mock classification here
    # In a real scenario, you would implement the Kano Model logic
    categorized_features = {
        "Must-have": [],
        "Expected": [],
        "Excitement": [],
        "Indifferent": [],
        "Killer": []
    }
    for feature in features.split(','):
        feature = feature.strip()
        if "must" in feature.lower():
            categorized_features["Must-have"].append(feature)
        elif "expect" in feature.lower():
            categorized_features["Expected"].append(feature)
        elif "excite" in feature.lower():
            categorized_features["Excitement"].append(feature)
        elif "indifferent" in feature.lower():
            categorized_features["Indifferent"].append(feature)
        else:
            categorized_features["Killer"].append(feature)
    return categorized_features

def main():
    st.title("Kano Model Feature Optimization")
    st.write("Welcome to the Kano Model Feature Optimization App!")

    product_name = st.text_input("Enter the product name:")
    development_goals = st.text_area("Specify your product development goals:")
    target_customers = st.text_area("Describe your target customers:")
    features = st.text_area("List the features to be tested (comma-separated):")

    if st.button("Submit"):
        st.write(f"Product: {product_name}")
        st.write(f"Development Goals: {development_goals}")
        st.write(f"Target Customers: {target_customers}")
        st.write(f"Features: {features}")

        try:
            categorized_features = kano_model_classification(features)
            st.write("Categorized Features:")
            for category, feats in categorized_features.items():
                st.write(f"{category}: {', '.join(feats)}")
        except Exception as e:
            st.error(f"Error during feature categorization: {e}")

if __name__ == "__main__":
    main()
