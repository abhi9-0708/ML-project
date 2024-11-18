
from flask import Flask, render_template, request
import pandas as pd
import textdistance
import re
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask_socketio import SocketIO, emit
import json

app = Flask(__name__)
socketio = SocketIO(app)

# Load text data and preprocess
words = []

with open('final.txt', 'r', encoding='utf-8') as f:
    data = f.read().lower()
    words = re.findall(r'\w+', data)

# Duplicate the words to increase frequency
words += words

# Create word frequency dictionary
words_freq_dict = Counter(words)
Total = sum(words_freq_dict.values())
probs = {word: freq / Total for word, freq in words_freq_dict.items()}

# Function to generate n-gram predictions
def generate_ngram_predictions(input_text, n=3):
    tokens = input_text.split()
    ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    predictions = [" ".join(ngram) for ngram in ngrams]
    return predictions

# Convert word to a fixed-length vector (precompute this)
def word_to_vector(word, max_length=20):
    vector = [ord(c) for c in word[:max_length]]
    vector += [0] * (max_length - len(vector))
    return vector

# Precompute word vectors to avoid recomputation during requests
word_vectors = np.array([word_to_vector(word) for word in words])

# Create a feature matrix and a target vector (for Decision Tree)
X = word_vectors
y = np.array([words_freq_dict[word] for word in words])

# Split the data into training, validation, and testing sets (60%, 20%, 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Evaluate the model on training, validation, and test sets
y_train_pred = dt_classifier.predict(X_train)
y_valid_pred = dt_classifier.predict(X_valid)
y_test_pred = dt_classifier.predict(X_test)

# Calculate metrics for training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

# Calculate metrics for validation set
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
valid_precision = precision_score(y_valid, y_valid_pred, average='weighted')
valid_recall = recall_score(y_valid, y_valid_pred, average='weighted')
valid_f1 = f1_score(y_valid, y_valid_pred, average='weighted')

# Calculate metrics for test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Print average metrics for training, validation, and test sets
print(f"Training Set Evaluation:")
print(f"Accuracy: {train_accuracy:.2f}")
print(f"Precision: {train_precision:.2f}")
print(f"Recall: {train_recall:.2f}")
print(f"F1 Score: {train_f1:.2f}")

print(f"\nValidation Set Evaluation:")
print(f"Accuracy: {valid_accuracy:.2f}")
print(f"Precision: {valid_precision:.2f}")
print(f"Recall: {valid_recall:.2f}")
print(f"F1 Score: {valid_f1:.2f}")

print(f"\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy:.2f}")
print(f"Precision: {test_precision:.2f}")
print(f"Recall: {test_recall:.2f}")
print(f"F1 Score: {test_f1:.2f}")

# Check for overfitting or underfitting
if train_accuracy > valid_accuracy + 0.1:
    print("\nThe model may be overfitting.")
elif train_accuracy < valid_accuracy - 0.1:
    print("\nThe model may be underfitting.")
else:
    print("\nThe model is well-fitted.")

# Jaccard similarity function
def jaccard_similarity(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    return len(set1 & set2) / len(set1 | set2)

@app.route('/')
def index():
    return render_template('index.html', suggestions=None)

@socketio.on('input_event')
def handle_input_event(json_data):
    data = json.loads(json_data)
    input_text = data.get('keyword', '').lower().strip()
    suggestions_list = []

    if input_text:
        # Extract the last word from the input text if it's a sentence
        keyword = input_text.split()[-1]

        # Generate n-gram predictions (optional, remove duplicates later)
        ngram_suggestions = generate_ngram_predictions(keyword)
        suggestions_list += ngram_suggestions

        # Check if the keyword itself exists in the word dictionary
        if keyword in words_freq_dict:
            suggestions_list.append(keyword)

        # Filter words starting with the same letter(s) as the input keyword
        filtered_words = [
            word for word in words_freq_dict.keys() 
            if word.startswith(keyword) and word != keyword
        ]

        # Fallback: If no matches, include words starting with any other letter
        if not filtered_words:
            filtered_words = [
                word for word in words_freq_dict.keys() 
                if word != keyword
            ]

        # Use the Decision Tree to predict the most frequent words
        filtered_word_vectors = np.array([word_to_vector(word) for word in filtered_words])
        predicted_frequencies = dt_classifier.predict(filtered_word_vectors)

        # Combine predicted frequencies with the filtered words
        word_freq_pairs = list(zip(filtered_words, predicted_frequencies))

        # Sort words by predicted frequency (higher frequency is better)
        sorted_word_freq_pairs = sorted(word_freq_pairs, key=lambda x: x[1], reverse=True)

        # Add sorted words to suggestions list
        suggestions_list += [pair[0] for pair in sorted_word_freq_pairs]

        # Remove duplicates from the suggestions list
        suggestions_list = list(dict.fromkeys(suggestions_list))  # Remove duplicates

        # If no exact matches, use Jaccard similarity for suggestions
        if not suggestions_list or keyword not in words_freq_dict:
            best_match = None
            best_similarity = 0
            for word in words_freq_dict.keys():
                similarity = jaccard_similarity(keyword, word)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = word

            # Add the best match from Jaccard similarity to suggestions
            if best_match:
                suggestions_list.append(best_match)

    # Emit suggestions to frontend
    emit('suggestions_response', {'suggestions': suggestions_list})

if __name__ == '__main__':
    socketio.run(app, debug=True)

