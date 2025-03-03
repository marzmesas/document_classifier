import streamlit as st
import requests
import json
import os

# Get API URL from environment variable or use default
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Configure the page
st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="centered"
)

def main():
    """
    Main function to run the Streamlit UI for document classification.
    
    This function:
    1. Sets up the Streamlit user interface with title and description
    2. Creates input fields for document text
    3. Processes user requests for document classification
    4. Displays classification results in a user-friendly format
    5. Handles errors and connection issues gracefully
    6. Provides debug information and documentation about document classes
    """
    # Title and description
    st.title("Document Classifier")
    st.markdown("""
    This application classifies documents into predefined categories.
    Enter your text below and click 'Predict' to get the classification.
    """)

    # Text input
    text_input = st.text_area(
        "Enter your text here:",
        height=200,
        placeholder="Type or paste your document text here..."
    )

    # Predict button
    if st.button("Predict", type="primary"):
        if not text_input:
            st.warning("Please enter some text before predicting.")
        else:
            try:
                # Make prediction request to FastAPI backend
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text_input}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results in a nice format
                    st.success("Prediction successful!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="Predicted Class",
                            value=f"Class {result['predicted_class']}"
                        )
                    
                    with col2:
                        st.metric(
                            label="Confidence",
                            value=f"{result['confidence']:.2%}"
                        )
                    
                    # Show probabilities as a bar chart
                    st.subheader("Class Probabilities")
                    probabilities = result['probabilities']
                    st.bar_chart(
                        {f"Class {i+1}": prob 
                         for i, prob in enumerate(probabilities)}
                    )
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            
            except requests.exceptions.ConnectionError:
                st.error(f"""
                    Could not connect to the backend service at {API_URL}. 
                    Please make sure the FastAPI server is running.
                """)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display connection information in development mode
    if st.checkbox("Show debug info"):
        st.info(f"Using API at: {API_URL}")

    # Add some information about the classes
    with st.expander("About the Document Classes"):
        st.markdown("""
        The classifier categorizes documents into 8 different classes:
        - Class 1: Category 1
        - Class 2: Category 2
        - Class 3: Category 3
        - Class 4: Category 4
        - Class 5: Category 5
        - Class 6: Category 6
        - Class 7: Category 7
        - Class 8: Category 8
        """)

if __name__ == "__main__":
    main() 