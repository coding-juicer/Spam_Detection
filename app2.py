import pandas as pd
import streamlit as st
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import os
from io import BytesIO

# Try to import PyPDF2, handle if not installed
try:
    import PyPDF2

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("⚠️ PyPDF2 not installed. PDF support disabled. Install with: pip install PyPDF2")

# Download NLTK data only if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


@st.cache_resource
def load_models():
    """Load the trained model and TF-IDF vectorizer with error handling."""
    try:
        if os.path.exists('model.pkl') and os.path.exists('tfidf.pkl'):
            model = pickle.load(open('model.pkl', 'rb'))
            tfidf = pickle.load(open('tfidf.pkl', 'rb'))
            return model, tfidf
        else:
            st.error("❌ Model files not found. Please ensure 'model.pkl' and 'tfidf.pkl' are in the same directory.")
            return None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    if not PDF_SUPPORT:
        st.error("❌ PDF support not available. Please install PyPDF2: pip install PyPDF2")
        return None

    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}")
        return None


def clean_text(text):
    """Clean and preprocess text for spam detection."""
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'http\S+', '', text)

    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return ""

    try:
        # Tokenize
        words = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word and word not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in filtered_words if word]

        # Join back to string
        cleaned_text = ' '.join(stemmed_words)
        return cleaned_text

    except Exception as e:
        st.error(f"❌ Error in text preprocessing: {str(e)}")
        return ""


def predict_spam(text, model, tfidf):
    """Predict if text is spam or not."""
    if not text.strip():
        return None, "Empty text provided"

    try:
        # Clean the text
        cleaned_text = clean_text(text)

        if not cleaned_text.strip():
            return None, "No meaningful text found after preprocessing"

        # Transform using TF-IDF
        vectorized_text = tfidf.transform([cleaned_text])

        # Make prediction
        prediction = model.predict(vectorized_text)[0]

        # Get prediction probability (if supported)
        try:
            prediction_proba = model.predict_proba(vectorized_text)[0]
            confidence = max(prediction_proba) * 100
        except:
            confidence = None

        return prediction, confidence

    except Exception as e:
        return None, f"Error in prediction: {str(e)}"


def main():
    st.set_page_config(
        page_title="Spam Detection App",
        page_icon="🛡️",
        layout="wide"
    )

    st.title('🛡️ Spam Detection App')
    st.markdown("---")
    st.write("**Detect spam messages using machine learning!**")
    st.write("Enter a message below or upload a file (PDF/TXT) to check if it's spam.")

    # Load models
    model, tfidf = load_models()

    if model is None or tfidf is None:
        st.stop()

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Text Input", "📁 File Upload"])

    with tab1:
        st.subheader("Enter Text Message")
        user_input = st.text_area(
            "✍️ Enter your message here:",
            height=200,
            placeholder="Type or paste your message here...",
            key="input_box"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            predict_button = st.button("🔍 Analyze Message", type="primary")

        if predict_button:
            if not user_input.strip():
                st.warning("⚠️ Please enter a message to analyze.")
            else:
                with st.spinner("Analyzing message..."):
                    prediction, confidence = predict_spam(user_input, model, tfidf)

                if prediction is None:
                    st.error(f"❌ {confidence}")
                else:
                    # Display results
                    if prediction == 1:
                        st.error("🚫 **SPAM DETECTED!**")
                        st.markdown("This message appears to be spam.")
                    else:
                        st.success("✅ **NOT SPAM**")
                        st.markdown("This message appears to be legitimate.")

                    if confidence:
                        st.info(f"🎯 Confidence: {confidence:.1f}%")

    with tab2:
        st.subheader("Upload File")

        # Show available file types based on installed packages
        if PDF_SUPPORT:
            file_types = ['pdf', 'txt']
            help_text = "Upload a PDF or TXT file to analyze"
        else:
            file_types = ['txt']
            help_text = "Upload a TXT file to analyze (PDF support requires PyPDF2)"

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=file_types,
            help=help_text
        )

        if uploaded_file is not None:
            try:
                with st.spinner("Processing file..."):
                    # Read file content
                    if uploaded_file.type == "application/pdf":
                        file_content = extract_text_from_pdf(uploaded_file.read())
                    else:  # txt file
                        try:
                            file_content = uploaded_file.read().decode('utf-8')
                        except UnicodeDecodeError:
                            file_content = uploaded_file.read().decode('latin-1')

                    if file_content is None:
                        st.error("❌ Could not read the file content.")
                    elif not file_content.strip():
                        st.warning("⚠️ The uploaded file appears to be empty.")
                    else:
                        # Show file preview
                        with st.expander("📄 File Content Preview"):
                            st.text(file_content[:500] + "..." if len(file_content) > 500 else file_content)

                        # Analyze file content
                        prediction, confidence = predict_spam(file_content, model, tfidf)

                        if prediction is None:
                            st.error(f"❌ {confidence}")
                        else:
                            # Display results
                            if prediction == 1:
                                st.error("🚫 **SPAM DETECTED IN FILE!**")
                                st.markdown("The file content appears to be spam.")
                            else:
                                st.success("✅ **FILE CONTENT IS NOT SPAM**")
                                st.markdown("The file content appears to be legitimate.")

                            if confidence:
                                st.info(f"🎯 Confidence: {confidence:.1f}%")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    # Add footer with information
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>This spam detection model uses machine learning to classify messages.</p>
        <p>Results are predictions and should be used as guidance only.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()