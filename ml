import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load spam dataset
try:
    spam_df = pd.read_csv('spam.csv', encoding='latin-1', low_memory=False)
except FileNotFoundError:
    print("Error: spam.csv file not found!")
    exit(1)

spam_df = spam_df[['v1', 'v2']]  
spam_df.columns = ['label', 'text']

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    else:
        text = ''  
    return text

spam_df['text'] = spam_df['text'].apply(preprocess_text)

# Data into features and labels
X_spam = spam_df['text']
y_spam = spam_df['label']

# Split into training and test sets
X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(X_spam, y_spam, test_size=0.2, random_state=42)

# Vectorize the text data for spam classification
vectorizer_spam = TfidfVectorizer(max_features=5000)
X_train_spam_vec = vectorizer_spam.fit_transform(X_train_spam)
X_test_spam_vec = vectorizer_spam.transform(X_test_spam)

# Train the spam detection model
spam_model = MultinomialNB()
spam_model.fit(X_train_spam_vec, y_train_spam)

# Load and preprocess toxicity dataset (Profanity dataset)
profinity_folder = 'profinity'
profinity_dfs = []

if not os.path.exists(profinity_folder):
    print(f"Folder {profinity_folder} does not exist!")
    exit(1)

for file_name in os.listdir(profinity_folder):
    if file_name.endswith('.csv'):
        try:
            file_path = os.path.join(profinity_folder, file_name)
            df = pd.read_csv(file_path, encoding='latin1', low_memory=False) 
            profinity_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

if not profinity_dfs:
    print("No valid profanity data found. Please check the 'profinity' folder.")
    exit(1)

profinity_df = pd.concat(profinity_dfs, ignore_index=True)
profinity_df = profinity_df[['comment_text', 'toxic']] 

# Preprocessing the toxicity data
profinity_df['comment_text'] = profinity_df['comment_text'].apply(preprocess_text)

# Data into features and labels
X_toxic = profinity_df['comment_text']
y_toxic = profinity_df['toxic']

# Split into training and test sets
X_train_toxic, X_test_toxic, y_train_toxic, y_test_toxic = train_test_split(X_toxic, y_toxic, test_size=0.2, random_state=42)

# Vectorize the text data for toxicity classification
vectorizer_toxic = TfidfVectorizer(max_features=5000)
X_train_toxic_vec = vectorizer_toxic.fit_transform(X_train_toxic)
X_test_toxic_vec = vectorizer_toxic.transform(X_test_toxic)

# Train the toxicity detection model
toxic_model = MultinomialNB()
toxic_model.fit(X_train_toxic_vec, y_train_toxic)

# Save models to disk in the 'models' folder
models_folder = 'models'
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

joblib.dump(spam_model, os.path.join(models_folder, 'spam_model.pkl'))
joblib.dump(vectorizer_spam, os.path.join(models_folder, 'vectorizer_spam.pkl'))
joblib.dump(toxic_model, os.path.join(models_folder, 'toxic_model.pkl'))
joblib.dump(vectorizer_toxic, os.path.join(models_folder, 'vectorizer_toxic.pkl'))

# Function for prediction based on user input
def predict_input(text_input):
    try:
        # Preprocess the input text
        text_input = preprocess_text(text_input)

        # Spam Prediction
        text_spam_vec = vectorizer_spam.transform([text_input])
        spam_prediction = spam_model.predict(text_spam_vec)[0]
        spam_label = 'Spam' if spam_prediction == 'spam' else 'Ham'

        # Profanity Prediction
        text_toxic_vec = vectorizer_toxic.transform([text_input])
        toxic_prediction = toxic_model.predict(text_toxic_vec)[0]
        toxic_label = 'Toxic' if toxic_prediction == 1 else 'Non-toxic'

        return spam_label, toxic_label
    except Exception as e:
        return 'Error processing text', 'Error processing text'

@app.route('/classify_comment', methods=['POST'])
def classify_comment():
    data = request.get_json()
    
    comment_text = data.get('comment_text', '')
    if not comment_text:
        return jsonify({'error': 'No comment_text provided'}), 400

    spam_result, toxic_result = predict_input(comment_text)

    return jsonify({
        'spam_result': spam_result,
        'toxic_result': toxic_result
    })

if __name__ == '__main__':
    app.run(debug=True)
