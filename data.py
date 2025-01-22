import pandas as pd
import re
import string
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# Load the datasets
fake_news_df = pd.read_csv("C:/xampp1/htdocs/DataMining/Fake.csv")
true_news_df = pd.read_csv("C:/xampp1/htdocs/DataMining/True.csv")

# Label the datasets
fake_news_df['label'] = 0  # 0 for Fake News
true_news_df['label'] = 1  # 1 for True News

# Combine the datasets
data = pd.concat([fake_news_df, true_news_df], axis=0).reset_index(drop=True)

# Preprocess the titles
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

data['title'] = data['title'].apply(preprocess_text)

# Feature Engineering
# 1. Average word length
data['avg_word_length'] = data['title'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if len(x.split()) > 0 else 0)

# 2. Sentiment Polarity and Subjectivity
data['polarity'] = data['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
data['subjectivity'] = data['title'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# 3. Capitalization Ratio and Exclamation Count
data['all_caps_ratio'] = data['title'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
data['exclamation_count'] = data['title'].apply(lambda x: x.count('!'))

# 4. Title Length and Word Count
data['title_length'] = data['title'].apply(len)
data['word_count'] = data['title'].apply(lambda x: len(x.split()))

# Splitting the data
X = data['title']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Additional Features
train_features = data.loc[X_train.index, ['avg_word_length', 'polarity', 'subjectivity', 'all_caps_ratio', 'exclamation_count', 'title_length', 'word_count']]
test_features = data.loc[X_test.index, ['avg_word_length', 'polarity', 'subjectivity', 'all_caps_ratio', 'exclamation_count', 'title_length', 'word_count']]

# Combine TF-IDF features with additional features
X_train_combined = hstack([X_train_tfidf, train_features])
X_test_combined = hstack([X_test_tfidf, test_features])

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_combined, y_train)

# Evaluate the model
y_pred = model.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict if a given news title is fake or real with additional features
def predict_news(title):
    # Preprocess and vectorize the title
    title_cleaned = preprocess_text(title)
    title_tfidf = vectorizer.transform([title_cleaned])
    
    # Extract additional features for the single title
    avg_word_length = sum(len(word) for word in title_cleaned.split()) / len(title_cleaned.split()) if len(title_cleaned.split()) > 0 else 0
    polarity = TextBlob(title_cleaned).sentiment.polarity
    subjectivity = TextBlob(title_cleaned).sentiment.subjectivity
    all_caps_ratio = sum(1 for c in title_cleaned if c.isupper()) / len(title_cleaned) if len(title_cleaned) > 0 else 0
    exclamation_count = title_cleaned.count('!')
    title_length = len(title_cleaned)
    word_count = len(title_cleaned.split())
    
    # Create a sparse matrix for additional features
    additional_features = [[avg_word_length, polarity, subjectivity, all_caps_ratio, exclamation_count, title_length, word_count]]
    
    # Combine TF-IDF features with additional features
    title_combined = hstack([title_tfidf, additional_features])
    
    # Predict the label
    prediction = model.predict(title_combined)
    return "Fake News" if prediction[0] == 0 else "Real News"

# Example usage
user_title = input("Enter a news title: ")
result = predict_news(user_title)
print(f"Prediction: {result}")
