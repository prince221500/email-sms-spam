import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization (split by space)
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def train():
    print("Loading dataset...")
    try:
        # Read CSV with no header, assuming first column is label, second is text
        df = pd.read_csv('spam.csv', encoding='latin-1', header=None)
        df = df.rename(columns={0: 'label', 1: 'text'})
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    print(f"Dataset shape: {df.shape}")
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    print("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text']).toarray()
    y = df['label']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Naive Bayes
    print("Training Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

    # Save the best model (using NB for now as baseline)
    print("Saving model and vectorizer...")
    pickle.dump(nb_model, open('model.pkl', 'wb'))
    pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
    print("Done.")

if __name__ == "__main__":
    train()
