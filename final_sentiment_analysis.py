import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define Deep Learning Models (BiLSTM, CNN)
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output


class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv1 = nn.Conv1d(embed_size, 100, 3)
        self.conv2 = nn.Conv1d(100, 100, 4)
        self.conv3 = nn.Conv1d(100, 100, 5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x1 = torch.relu(self.conv1(x)).max(dim=2)[0]
        x2 = torch.relu(self.conv2(x)).max(dim=2)[0]
        x3 = torch.relu(self.conv3(x)).max(dim=2)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        output = self.fc(x)
        return output


# Load Models
bilstm_model = BiLSTMModel(input_size=1000, hidden_size=128, output_size=3).to(device)
cnn_model = CNNModel(vocab_size=1000, embed_size=100, num_classes=3).to(device)

bilstm_model.eval()
cnn_model.eval()


# Tokenizer for Deep Learning Models
vectorizer = CountVectorizer(max_features=1000)


# Load RoBERTa and VADER models
@st.cache_resource
def load_roberta_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")


@st.cache_resource
def load_vader_model():
    return SentimentIntensityAnalyzer()


roberta_model = load_roberta_model()
vader_model = load_vader_model()


# Custom CSS Style
st.markdown('''
    <style>
        body { background-color: white; color: black; }
        h1 { text-align: center; color: green; font-size: 3rem; transition: color 0.3s; }
        h1:hover { color: white; }
        .css-18e3th9 { padding: 1rem 2rem 0rem 2rem; text-align: center; }
        .stButton>button { background-color: green; color: white; border-radius: 10px; transition: background-color 0.3s; }
        .stButton>button:hover { background-color: white; color: green; }
        .stTextInput>div>div>input { background-color: #F0F2F6; }
        img { transition: transform 0.3s; }
        img:hover { transform: scale(1.05); }
        th, td { color: green; }
    </style>
''', unsafe_allow_html=True)


st.sidebar.title("Services")
st.sidebar.write("Our services use VADER, RoBERTa, BiLSTM, and CNN models for sentiment analysis. The results are categorized as Positive, Negative, and Neutral.")
st.title("SENTIMENT ANALYZER")

analysis_type = st.sidebar.selectbox("Select Analysis Type:", ["Text Analysis", "CSV Analysis"])


def plot_sentiment_distribution(sentiments):
    sentiment_counts = sentiments.value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    st.pyplot(plt)


def plot_pie_chart(sentiments):
    sentiment_counts = sentiments.value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["#76c893", "#ff6b6b", "#ffd93d"])
    plt.title("Sentiment Distribution Pie Chart")
    st.pyplot(plt)


if analysis_type == "CSV Analysis":
    uploaded_file = st.file_uploader("Upload CSV file for analysis", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV file:")
        st.dataframe(df, width=1200, height=400)

        product_column = st.selectbox("Select Product Name Column", df.columns)
        review_column = st.selectbox("Select Review Text Column", df.columns)

        if st.button("Start Analysis"):
            texts = df[review_column].dropna().astype(str).tolist()
            products = df[product_column].dropna().astype(str).tolist()

            roberta_results = roberta_model(texts)
            vader_results = [vader_model.polarity_scores(text)['compound'] for text in texts]

            df['Combined Sentiment'] = ['Positive' if roberta_results[i]['label'] == 'POSITIVE' or vader_results[i] > 0
                                        else 'Negative' if roberta_results[i]['label'] == 'NEGATIVE' or vader_results[i] < 0
                                        else 'Neutral' for i in range(len(texts))]

            df['Product Name'] = products

            st.write("Analysis Results:")
            st.dataframe(df, width=1200, height=400)

            plot_sentiment_distribution(df['Combined Sentiment'])
            plot_pie_chart(df['Combined Sentiment'])

            for sentiment in ['Positive', 'Negative', 'Neutral']:
                sentiment_df = df[df['Combined Sentiment'] == sentiment]
                st.subheader(f"{sentiment} Reviews")
                st.write(sentiment_df)

if analysis_type == "Text Analysis":
    text_input = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Text") and text_input:
        roberta_result = roberta_model(text_input)[0]
        vader_result = vader_model.polarity_scores(text_input)['compound']
        sentiment = "Positive" if roberta_result['label'] == 'POSITIVE' or vader_result > 0 else "Negative" if roberta_result['label'] == 'NEGATIVE' or vader_result < 0 else "Neutral"
        st.write(f"Sentiment: {sentiment}")
