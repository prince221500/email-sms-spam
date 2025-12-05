import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded (redundant if already done, but safe)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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

def predict_message(message):
    try:
        # Load model and vectorizer
        model = pickle.load(open('model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    except FileNotFoundError:
        print("Error: Model files not found. Please run train_model.py first.")
        return

    # Preprocess and vectorize
    cleaned_text = preprocess_text(message)
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()

    # Predict
    prediction = model.predict(vectorized_text)[0]
    return prediction

if __name__ == "__main__":
    print("--- Spam Detection Test ---")
    print("Type a message to check if it's Spam or Ham.")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter message: ")
        if user_input.lower() == 'exit':
            break
        
        result = predict_message(user_input)
        print(f"Prediction: {result.upper()}")
