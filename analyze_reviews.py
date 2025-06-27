import pandas as pd
from transformers import pipeline
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import re

# Load the cleaned data from Task 1
try:
    df = pd.read_csv('bank_app_reviews.csv')
    print("Loaded bank_app_reviews.csv successfully.")
except FileNotFoundError:
    print("Error: bank_app_reviews.csv not found. Please run Task 1 first.")
    exit()

# --- 1. Sentiment Analysis ---
print("\nStarting Sentiment Analysis...")

# Load pre-trained sentiment analysis model from Hugging Face
# This model classifies text as 'POSITIVE' or 'NEGATIVE'
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to get sentiment label and score
def get_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return "neutral", 0.5 # Assign neutral for empty or missing reviews
    try:
        result = sentiment_pipeline(text)[0]
        # Map 'POSITIVE'/'NEGATIVE' to positive/negative and score
        if result['label'] == 'POSITIVE':
            return 'positive', result['score']
        elif result['label'] == 'NEGATIVE':
            return 'negative', result['score']
        else:
            return 'neutral', result['score'] # Should not happen with this model, but good practice
    except Exception as e:
        print(f"Error processing sentiment for text: {text[:50]}... Error: {e}")
        return "neutral", 0.5 # Default to neutral if there's an error

# Apply sentiment analysis
# This might take a while depending on the number of reviews and your hardware
# For a large number of reviews, consider batching or using a GPU if available
df[['sentiment_label', 'sentiment_score']] = df['review'].apply(
    lambda x: pd.Series(get_sentiment(x))
)
print("Sentiment analysis complete.")

# Aggregate by bank and rating
print("\nAggregating sentiment by bank and rating:")
# Calculate mean sentiment score for positive/negative labels
# Note: distilbert gives scores for POS/NEG. We can interpret positive scores for positive, and (1-score) for negative for consistency.
# Or simply look at the label distribution and average score per label.
sentiment_summary = df.groupby(['bank', 'rating', 'sentiment_label']).size().unstack(fill_value=0)
print(sentiment_summary)


# --- 2. Thematic Analysis ---
print("\nStarting Thematic Analysis...")

# Preprocessing for thematic analysis (using spaCy)
nlp = spacy.load("en_core_web_sm") # Load small English model

def preprocess_text_for_theme(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text)
    # Lemmatization and stop-word removal, keep only nouns and adjectives for keywords
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha and token.lemma_ not in ['app', 'bank'] # remove common generic words
        and token.pos_ in ['NOUN', 'ADJ', 'VERB'] # Focus on descriptive words
    ]
    return " ".join(tokens)

df['processed_review'] = df['review'].apply(preprocess_text_for_theme)
print("Reviews preprocessed for thematic analysis (lemmatization, stop-word removal, etc.).")


tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2)) # Consider unigrams and bigrams
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_review'])
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get top N keywords for each bank
top_n_keywords = 15
bank_keywords = defaultdict(list)

for bank in df['bank'].unique():
    print(f"\nTop keywords for {bank}:")
    bank_reviews_idx = df[df['bank'] == bank].index
    bank_tfidf_matrix = tfidf_matrix[bank_reviews_idx]

    # Sum TF-IDF scores for each feature across the bank's reviews
    feature_scores = bank_tfidf_matrix.sum(axis=0).A1 # .A1 converts matrix to 1D array
    # Get indices of top scores
    top_indices = feature_scores.argsort()[-top_n_keywords:][::-1]
    for i in top_indices:
        keyword = feature_names[i]
        score = feature_scores[i]
        bank_keywords[bank].append((keyword, score))
        print(f"- {keyword} ({score:.2f})")


theme_keywords = {
    'Account Access Issues': ['login', 'sign in', 'password', 'fingerprint', 'security', 'error'],
    'Transaction Performance': ['transfer', 'send money', 'slow', 'fast', 'transaction', 'payment', 'stuck', 'delay'],
    'User Interface & Experience': ['ui', 'interface', 'design', 'easy', 'confusing', 'user friendly', 'layout', 'bug', 'crash'],
    'Customer Support': ['support', 'customer service', 'help', 'agent', 'contact'],
    'Feature Requests': ['feature', 'add', 'option', 'update', 'new']
}

# Function to assign themes to reviews
def assign_theme(review_text, theme_keywords_map):
    themes = []
    text_lower = str(review_text).lower() # Ensure string and lowercase
    for theme, keywords in theme_keywords_map.items():
        for keyword in keywords:
            if keyword in text_lower:
                themes.append(theme)
                break # Assign theme if any keyword is found
    return ", ".join(themes) if themes else "Other"

# Apply thematic assignment
df['identified_themes'] = df['review'].apply(lambda x: assign_theme(x, theme_keywords))
print("\nReviews assigned to themes based on keyword matching.")

# Save results to a new CSV
output_filename_analysis = 'bank_app_reviews_analyzed.csv'
df[['review', 'rating', 'date', 'bank', 'source', 'sentiment_label', 'sentiment_score', 'identified_themes']].to_csv(output_filename_analysis, index=False, encoding='utf-8')
print(f"\nAnalyzed data saved to {output_filename_analysis}")

print("\nFirst 5 rows of analyzed data:")
print(df[['review', 'rating', 'sentiment_label', 'sentiment_score', 'identified_themes']].head())