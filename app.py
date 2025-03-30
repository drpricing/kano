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
    st.title("⚙️ Settings")
    api_key = st.secrets["groq"]["api_key"]
    
    st.markdown("---")
    st.markdown("### How does it work?")
    st.markdown("""
    Step 1: Add the necessary information for the experiment in the Setup tab
    
    Step 2: Check progress of the experiment in the Run Experiment tab
    
    Step 3: View the results in the Results tab
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
        elif not product_name or not target_customers or not features:
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
            
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": input_text}
                ],
                temperature=0
            )
            personalist.append(response.choices[0].message.content)
            # Add longer sleep to avoid rate limits
            time.sleep(random.uniform(5, 10))
        
        profiles_df['Persona'] = personalist
        
        # Experiment instructions
        Instructions_experiment = """
        The user will provide you (1) a description of the product, (2) a specific question that needs to be answered (3) a description of the person who needs to answer this question for the product.
        ## Steps to Complete the Task:
        1. Read the description of the product.
        2. Read the specific question that needs to be answered.
        3. Read the description of the person who needs to answer this question.
        4. Make sure that you fully embrace the perspective of the person described and provide an answer to the question that aligns with their characteristics and motivations.
        5. Provide a response to the question based on the information provided.
        ## Output Format:
        You MUST respond with a valid JSON object in the following exact format:
        {
            "Answer": "your answer here",
            "Explanation": "your explanation here"
        }
        Important:
        - The response must be a single, valid JSON object
        - Do not include any text before or after the JSON object
        - Do not use markdown code blocks
        - Ensure all quotes are straight quotes (") not curly quotes
        - Do not include any line breaks within the values
        """
        
        # Collect responses
        status_text.text("Collecting responses...")
        answerlist = []
        for i in range(profiles_df.shape[0]):
            progress_bar.progress((i + 1 + profiles_df.shape[0]) / (profiles_df.shape[0] * 3))
            input_text = f"Product description: {product_name}; Features: {features}; Person description: {profiles_df['Persona'].iloc[i]}"
            
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": Instructions_experiment},
                    {"role": "user", "content": input_text}
                ],
                temperature=0
            )
            answerlist.append(response.choices[0].message.content)
            # Add longer sleep to avoid rate limits
            time.sleep(random.uniform(5, 10))
        
        # Process responses
        gradelist = []
        explanationlist = []
        for i in range(len(answerlist)):
            overview = answerlist[i]
            try:
                # Clean the response string
                if isinstance(overview, str):
                    # Remove any markdown code block markers
                    overview = overview.replace('```json', '').replace('```', '').strip()
                    # Remove any control characters
                    overview = ''.join(char for char in overview if ord(char) >= 32 or char in '\\n\\r\\t')
                    # Remove any curly quotes
                    overview = overview.replace('“', '"').replace('”', '"')
                    # Try to parse as JSON
                    try:
                        overviewjson = json.loads(overview)
                    except json.JSONDecodeError as e:
                        # If JSON parsing fails, try to extract answer and explanation using regex
                        import re
                        answer_match = re.search(r'"Answer"\s*:\s*"\'["\']', overview)
                        explanation_match = re.search(r'"Explanation"\s*:\s*"\'["\']', overview)
                        
                        if answer_match and explanation_match:
                            overviewjson = {
                                'Answer': answer_match.group(1),
                                'Explanation': explanation_match.group(1)
                            }
                        else:
                            # Try one more time with a simpler pattern
                            answer_match = re.search(r'Answer\s*:\s*"\'["\']', overview)
                            explanation_match = re.search(r'Explanation\s*:\s*"\'["\']', overview)
                            
                            if answer_match and explanation_match:
                                overviewjson = {
                                    'Answer': answer_match.group(1),
                                    'Explanation': explanation_match.group(1)
                                }
                            else:
                                raise e
                    else:
                        overviewjson = overview
                    
                    # Extract answer and explanation
                    answer = overviewjson.get('Answer', None)
                    explanation = overviewjson.get('Explanation', None)
                    
                    # Store the values
                    gradelist.append(answer)
                    explanationlist.append(explanation)
                
            except Exception as e:
                st.warning(f"Error processing response {i+1}: {str(e)}")
                st.write("Raw response:", overview)
                gradelist.append(None)
                explanationlist.append(None)
        
        profiles_df['Answer'] = gradelist
        profiles_df['Explanation'] = explanationlist
        
        # Store results in session state
        st.session_state.results = profiles_df
        st.session_state.experiment_complete = True
        
        # Show success message and instructions
        st.success("✅ Experiment completed successfully!")
        st.info("Please navigate to the 'Results' tab to view and analyze your data.")

with tab3:
    if not st.session_state.experiment_complete:
        st.info("Please run the experiment in the 'Run Experiment' tab first.")
    else:
        st.header("Results Analysis")
        
        # Convert answers to numerical scores
        def convert_to_score(answer):
            try:
                # If answer is already a number, return it
                if isinstance(answer, (int, float)):
                    return answer
                
                # If answer is a string, try to extract the first number
                if isinstance(answer, str):
                    # Try to parse as JSON first
                    try:
                        answer_dict = json.loads(answer)
                        if isinstance(answer_dict, dict) and 'Answer' in answer_dict:
                            answer = answer_dict['Answer']
                    except:
                        pass
                    
                    # Extract the first number from the string
                    import re
                    numbers = re.findall(r'[-+]?\\d*\\.\\d+|\\d+', str(answer))
                    if numbers:
                        return float(numbers[0])
                    return None
                
                return None
            except Exception as e:
                st.warning(f"Error converting answer to score: {str(e)}")
                return None
        
        # Convert answers to scores
        profiles_df = st.session_state.results.copy() # Create a copy to prevent modifications to original data
        
        profiles_df['Score'] = profiles_df['Answer'].apply(convert_to_score)
        
        # Debug information
        st.write("Sample of Scores:", profiles_df['Score'].head())
        
        # Remove rows with missing scores
        profiles_df = profiles_df.dropna(subset=['Score'])
        
        if len(profiles_df) == 0:
            st.error("No valid scores found in the responses. Please check the answer format in your experiment setup.")
            st.stop() # Stop execution instead of using return
        
        # Display summary statistics
        st.metric(
            "Mean Score",
            f"{profiles_df['Score'].mean():.2f}",
            f"SD: {profiles_df['Score'].std():.2f}"
        )
        
        # Create box plot
        fig = go.Figure()
        fig.add_trace(go.Box(y=profiles_df['Score'], name='Scores'))
        fig.update_layout(
            title='Score Distribution',
            yaxis_title='Score',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical analysis
        st.subheader("Statistical Analysis")
        
        # Perform one-sample t-test
        t_stat, p_value = stats.ttest_1samp(profiles_df['Score'], 0)
        
        # Display results in a nice format
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("T-statistic", f"{t_stat:.4f}")
        with col2:
            st.metric("P-value", f"{p_value:.4f}")
        with col3:
            alpha = 0.05
            if p_value < alpha:
                st.metric("Result", "Significant", delta="Yes")
            else:
                st.metric("Result", "Not Significant", delta="No")
        
        # Detailed interpretation
        st.markdown("### Interpretation")
        if p_value < alpha:
            st.success("There is a statistically significant difference. The scores are significantly different from zero.")
        else:
            st.info("There is no statistically significant difference. The scores are not significantly different from zero.")
        
        # Print the results
        st.write(profiles_df)
        # Download results
        
        st.download_button(
            label="📥 Download Results",
            data=profiles_df.to_csv(index=False),
            file_name=f"kano_model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
