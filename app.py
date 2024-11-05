# app.py
import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newspaper import Article
import gradio as gr

# Download necessary NLTK data
nltk.data.path.append("./")  # Specify directory for NLTK data
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download('vader_lexicon', download_dir='./')

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Function to analyze sentiment of the text
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']
    
    # Determine sentiment based on compound score
    if compound_score >= 0.05:
        sentiment = 'Bullish'
    elif compound_score <= -0.05:
        sentiment = 'Bearish'
    else:
        sentiment = 'Neutral'
    
    return sentiment, scores

# Function to fetch article text from URL
def get_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

# Prediction function that combines article fetching and sentiment analysis
def predict_market_sentiment(url):
    # Get the article text
    article_text = get_article_text(url)
    
    # Analyze sentiment
    sentiment, scores = analyze_sentiment(article_text)
    
    # Calculate certainty percentage based on compound score
    certainty_percentage = abs(scores['compound']) * 100  # Convert to percentage
    
    # Return values instead of printing (for Gradio)
    return f"Predicted Market Sentiment: {sentiment}", f"Certainty of Prediction: {certainty_percentage:.2f}%", f"Sentiment Scores: {scores}"

# Gradio function to handle URL input and call prediction function
def gradio_predict_sentiment(url):
    prediction, certainty, scores = predict_market_sentiment(url)
    return f"{prediction}\n{certainty}\n{scores}"

# Gradio Interface setup
demo = gr.Interface(
    fn=gradio_predict_sentiment,
    inputs=gr.Textbox(label="Enter Article URL"),
    outputs="text",
    title="Crypto Market Sentiment Predictor",
    description="Enter the URL of a cryptocurrency-related article to analyze its investment sentiment."
)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()
