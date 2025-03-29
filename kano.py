import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import tempfile

# Function to extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        st.error(f"Error fetching content from {url}: {e}")
        return None

# Function to save uploaded file temporarily
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

# Streamlit UI
st.title("Chatbot with File and URL Processing")

# Text input for URL
url_input = st.text_input("Enter URL to process")

# File uploader
uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf', 'docx'])

# Process URL input
if url_input:
    st.write(f"Processing content from: {url_input}")
    extracted_text = extract_text_from_url(url_input)
    if extracted_text:
        st.text_area("Extracted Text", extracted_text, height=300)

# Process uploaded file
if uploaded_file:
    st.write(f"Processing uploaded file: {uploaded_file.name}")
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Implement file processing logic here
        st.success(f"File saved to {file_path}")
        os.remove(file_path)  # Clean up the temporary file
