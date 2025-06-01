# Spam_Detection   :https://spamdetectionweb.streamlit.app/
# 🛡️ Spam Detection App

A machine learning-powered web application that automatically detects spam messages using Natural Language Processing and Streamlit.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.0+-red.svg)


## 🚀 Features

- **Real-time Spam Detection**: Analyze text messages instantly
- **File Upload Support**: Process TXT and PDF files
- **Advanced NLP Pipeline**: Text cleaning, stemming, and TF-IDF vectorization
- **Confidence Scoring**: Get prediction confidence percentages
- **User-Friendly Interface**: Modern, responsive web interface
- **Batch Processing**: Upload files for bulk analysis

## 🎯 Demo

![App Screenshot](./image/Screenshot%202025-06-01%20at%205.23.40%E2%80%AFPM.png)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/spam-detection-app.git
   cd spam-detection-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (first run only)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📋 Requirements

Create a `requirements.txt` file with:

```
streamlit>=1.25.0
pandas>=1.5.0
numpy>=1.21.0
nltk>=3.8
scikit-learn>=1.3.0
PyPDF2>=3.0.0
```

## 🏗️ Project Structure

```
spam-detection-app/
│
├── app.py                 # Main Streamlit application
├── model.pkl             # Pre-trained ML model
├── tfidf.pkl            # TF-IDF vectorizer
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── assets/              # Screenshots and demo files
    ├── demo.gif
    └── screenshot.png
```

## 🧠 How It Works

### Machine Learning Pipeline

1. **Text Preprocessing**
   - Convert to lowercase
   - Remove URLs and special characters
   - Tokenization using NLTK
   - Stop words removal
   - Porter Stemming

2. **Feature Extraction**
   - TF-IDF Vectorization
   - Convert text to numerical features

3. **Classification**
   - Pre-trained scikit-learn model
   - Binary classification (Spam/Not Spam)
   - Confidence score calculation

### Algorithm Flow
```
Input Text → Clean Text → Tokenize → Remove Stop Words → Stem → TF-IDF → Model → Prediction
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 95.2% |
| Precision | 94.8% |
| Recall | 95.6% |
| F1-Score | 95.2% |

*Note: Update these metrics based on your actual model performance*

## 🎮 Usage

### Text Input
1. Navigate to the "Text Input" tab
2. Enter your message in the text area
3. Click "Analyze Message"
4. View the prediction and confidence score

### File Upload
1. Switch to the "File Upload" tab
2. Choose a TXT or PDF file
3. View the file content preview
4. Get automatic spam detection results

## 🔧 Configuration

### Model Training
To train your own model, prepare your dataset with columns:
- `text`: Message content
- `label`: 0 for ham, 1 for spam

```python
# Example training code structure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Train your model
model = MultinomialNB()
tfidf = TfidfVectorizer(max_features=3000)

# Save models
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
```

## 🚨 Error Handling

The app includes comprehensive error handling for:
- Missing model files
- Corrupt or unreadable files
- Network issues
- Invalid input formats
- NLTK data download failures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [NLTK](https://www.nltk.org/) for natural language processing tools
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms





---

⭐ **If you found this project helpful, please give it a star!** ⭐
